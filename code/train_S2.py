import datetime
import time
import torch
from torch import optim
from torch.utils import data
from tqdm import tqdm
import numpy as np
import argparse
import wandb
from pathlib import Path

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from data_generator import DataGenerator, Dataset
from network import StepFoldNet, LossFunc, HelixCenterMaskedPriorGenerator
from utils import *

def evaluate_model(model, feature_gen, data_loader, device, config):
    model.eval()
    feature_gen.eval()
    
    result = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating (FP32)"):
            seq_indices, legal_mask, contact_map, seq_length, padding_mask, node_set1_list = data
            
            seq_indices = seq_indices.to(device)
            legal_mask = legal_mask.to(device)
            contact_map = contact_map.to(device)
            seq_length = seq_length.to(device)
            padding_mask = padding_mask.to(device)
            
            L_max = seq_indices.shape[1]
                        
            # Generate features
            helix_prior = feature_gen(seq_indices, legal_mask) # (B, L, L, C)
            
            # Generate band masks
            band_masks = create_dynamic_start_band_masks(
                seq_length, 
                config['K_local'], config['K_global'], 
                config['min_distance'], config['max_distance'], config['start_ratio'],
                config['growth_power'], 
                L_max, device
            )

            input_data = (helix_prior, band_masks, legal_mask, padding_mask)

            pred_logits = model.inference(input_data)
            
            base_pair_prob = torch.sigmoid(pred_logits) * legal_mask.float()

            for i in range(len(seq_length)):
                L = seq_length[i].item()
                contact_map_pred_prob = base_pair_prob[i, :L, :L]
                
                # Use the same MWM post-processing as test_all.py.
                param = (contact_map_pred_prob, node_set1_list[i], config['threshold'])
                contact_map_pred = post_process_maximum_weight_matching(param)
                
                contact_map_label = contact_map[i, :L, :L]
                result.append(evaluate(contact_map_pred, contact_map_label))

    nt_exact_p, nt_exact_r, nt_exact_f1 = zip(*result)
    avg_p = np.average(nt_exact_p)
    avg_r = np.average(nt_exact_r)
    avg_f1 = np.average(nt_exact_f1)
 
    return avg_f1, avg_p, avg_r

def train(model, feature_gen, loss_fn, train_loader, optimizer, device, config):
    model.train()
    feature_gen.eval()
    
    total_loss = 0.0
    optimizer.zero_grad()
    
    if int(os.environ['LOCAL_RANK']) == 0:
        iterator = tqdm(train_loader, desc="Training (FP32)")
    else:
        iterator = train_loader
    
    for i, data in enumerate(iterator):
        seq_indices, legal_mask, contact_map, padding_mask, seq_length = data
        
        seq_indices = seq_indices.to(device)
        legal_mask = legal_mask.to(device)
        contact_map = contact_map.to(device)
        padding_mask = padding_mask.to(device)
        seq_length = seq_length.to(device)
        
        L_max = seq_indices.shape[1]
        
        helix_prior = feature_gen(seq_indices, legal_mask) # (B, L, L, C)
        
        band_masks = create_dynamic_start_band_masks(
                seq_length, 
                config['K_local'], config['K_global'], 
                config['min_distance'], config['max_distance'], config['start_ratio'],
                config['growth_power'], 
                L_max, device
                )
    
        input_data = (helix_prior, band_masks, legal_mask, padding_mask)
        predictions = model(input_data)
        
        loss = loss_fn(predictions, contact_map, band_masks)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
    return total_loss


def make_train_loader(dataset, config):
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['batch_size'] * 6,
        pin_memory=True,
        sampler=sampler,
        collate_fn=collate_train,
    )
    return loader


def make_eval_loader(dataset):
    return data.DataLoader(
        dataset, batch_size=1, num_workers=6, pin_memory=True,
        shuffle=False, collate_fn=collate_test,
    )


def run_stage(stage, epochs, model, feature_gen, loss_fn, train_loader, optimizer,
              device, config, local_rank, global_epoch, val_loader=None,
              test_loader=None, checkpoint_dir=None):
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loader.sampler.set_epoch(epoch)
        local_loss = train(model, feature_gen, loss_fn, train_loader, optimizer, device, config)
        total_loss = torch.tensor(local_loss, device=device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        avg_loss = total_loss.item() / (len(train_loader) * dist.get_world_size())

        if local_rank == 0:
            metrics = {"epoch": global_epoch + epoch, f"{stage}_train_loss": avg_loss}
            if val_loader is not None:
                f1, precision, recall = evaluate_model(
                    model.module, feature_gen, val_loader, device, config
                )
                metrics.update({f"{stage}_val_f1": f1,
                                f"{stage}_val_precision": precision,
                                f"{stage}_val_recall": recall})
            if test_loader is not None:
                f1, precision, recall = evaluate_model(
                    model.module, feature_gen, test_loader, device, config
                )
                metrics.update({f"{stage}_test_f1": f1,
                                f"{stage}_test_precision": precision,
                                f"{stage}_test_recall": recall})
            if config['save_every'] and (epoch % config['save_every'] == 0 or epoch == epochs):
                checkpoint_path = checkpoint_dir / f"{stage}_epoch_{epoch}.pt"
                torch.save(model.module.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            elapsed = time.time() - start_time
            metrics[f"{stage}_epoch_seconds"] = elapsed
            wandb.run.log(metrics, step=global_epoch + epoch)
            summary = f"{stage} epoch {epoch}/{epochs} | loss={avg_loss:.6f}"
            if val_loader is not None:
                summary += f" | val F1={metrics[f'{stage}_val_f1']:.6f}"
            if test_loader is not None:
                summary += f" | test F1={metrics[f'{stage}_test_f1']:.6f}"
            print(f"{summary} | {elapsed:.1f}s")
        dist.barrier()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture_config_path', type=str, default='configs/Architecture.json', help='Path to the architecture config file')
    parser.add_argument('--training_config_path', type=str, default='configs/S2.json', help='Path to the training config file')
    parser.add_argument('--save_every', type=int, default=None,
                        help='Save weights every N epochs of each stage (0 disables saving; default: config value)')
    
    args = parser.parse_args()
    config = process_config(args.architecture_config_path, args.training_config_path)
    config['save_every'] = config.get('save_every', 0) if args.save_every is None else args.save_every
    if not isinstance(config['save_every'], int) or config['save_every'] < 0:
        parser.error('--save_every must be a non-negative integer')
    
    init_ddp()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{local_rank}")
    
    seed_torch(2025 + local_rank)
        
    # --- Initialize WandB ---
    run_name = f"StepFold_S2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}" if local_rank == 0 else None
    run_name_list = [run_name]
    dist.broadcast_object_list(run_name_list, src=0)
    run_name = run_name_list[0]
    if local_rank == 0:
        wandb.init(project="StepFold", name=run_name, config=config)
        wandb.run.log_code("code/")
        wandb.run.log_code("configs/")
    
    # --- Load data ---
    if local_rank == 0:
        print('Loading datasets...')
    
    pretrain_data = DataGenerator(config['train_data_dir1'], 'train_max600_with_indices', mode='train')
    pretrain_data.merge(DataGenerator(config['train_data_dir2'], 'train_with_indices', mode='train'))
    pretrain_loader = make_train_loader(Dataset(pretrain_data), config)

    if local_rank == 0:
        print('Pretraining data loading done.')
    
    feature_gen = HelixCenterMaskedPriorGenerator(K=config['helices_num']).to(device)

    model = StepFoldNet(
        K_local=config['K_local'],
        K_global=config['K_global'],
        hidden_dim=config['embedding_dim'],
        helix_prior_K=config['helices_num'],
        ff_kernel_size=config['ff_kernel_size'],
        ff_expansion=config['ff_expansion'],
        ff_depth=config['ff_depth']
    ).to(device)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    loss_fn = LossFunc(
        K_total=config['K_total'], 
        alpha=config['alpha'], 
        pos_weight=config['pos_weight']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    
    if local_rank == 0:
        print(f"Model #Params Num: {sum([x.nelement() for x in model.parameters()])}")
        if config['save_every']:
            checkpoint_dir = Path(config['log_dir']) / run_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            checkpoint_dir = None

    run_stage('pretrain', config['pretrain_epochs'], model, feature_gen, loss_fn,
              pretrain_loader, optimizer, device, config, local_rank, 0,
              checkpoint_dir=checkpoint_dir if local_rank == 0 else None)

    # Fine-tune the in-memory model on PDB TR1 without selecting a validation checkpoint.
    finetune_data = DataGenerator(config['train_data_dir3'], 'TR1_with_indices', mode='train')
    finetune_loader = make_train_loader(Dataset(finetune_data), config)
    finetune_val_loader = (
        make_eval_loader(Dataset(DataGenerator(config['train_data_dir3'], 'VL1_with_indices', mode='test')))
        if local_rank == 0 else None
    )
    test_loader = (
        make_eval_loader(Dataset(DataGenerator(config['test_data_dir'], 'TS123_hard_with_indices', mode='test')))
        if local_rank == 0 else None
    )
    if local_rank == 0:
        print('Fine-tuning data loading done.')
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    run_stage('finetune', config['finetune_epochs'], model, feature_gen, loss_fn,
              finetune_loader, optimizer, device, config, local_rank,
              config['pretrain_epochs'], val_loader=finetune_val_loader,
              test_loader=test_loader,
              checkpoint_dir=checkpoint_dir if local_rank == 0 else None)

    if local_rank == 0:
        print(f"\nTraining finished.")
        wandb.finish()
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
    # Multi-GPU example: torchrun --standalone --nproc_per_node=4 code/train_S2.py
