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

def test(model, feature_gen, data_loader, device, config):
    model.eval()
    feature_gen.eval()
    
    result = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Testing (FP32)"):
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
                
                # Post-processing (MWM / HK)
                param = (contact_map_pred_prob, node_set1_list[i], config['threshold'])
                contact_map_pred = post_process_HK(param)
                
                contact_map_label = contact_map[i, :L, :L]
                result.append(evaluate(contact_map_pred, contact_map_label))

    nt_exact_p, nt_exact_r, nt_exact_f1 = zip(*result)
    avg_p = np.average(nt_exact_p)
    avg_r = np.average(nt_exact_r)
    avg_f1 = np.average(nt_exact_f1)
 
    print(f'Test Precision: {avg_p:.6f}, Recall: {avg_r:.6f}, F1: {avg_f1:.6f}')
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
        
    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture_config_path', type=str, default='configs/Architecture.json', help='Path to the architecture config file')
    parser.add_argument('--training_config_path', type=str, default='configs/S1.json', help='Path to the training config file')
    
    args = parser.parse_args()
    config = process_config(args.architecture_config_path, args.training_config_path)
    
    init_ddp()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device(f"cuda:{local_rank}")
    
    seed_torch(2025 + local_rank)
        
    # --- Initialize WandB ---
    if local_rank == 0:
        run_name = f"StepFold_S1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project="StepFold", name=run_name, config=config)
        wandb.run.log_code("code/")
        wandb.run.log_code("configs/")
    
    # --- Load data ---
    if local_rank == 0:
        print('Loading datasets...')
    
    train_data = DataGenerator(config['train_data_dir'], 'train_with_indices', mode='train')
    train_dataset = Dataset(train_data)
    
    global_max_len = train_data.max_len
    print(f"Global Max Length for Padding: {global_max_len}")
    global_mean_len = train_data.mean_len
    print(f"Global Mean Length: {global_mean_len:.2f}")
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = data.DataLoader(train_dataset, 
                                   batch_size=config['batch_size'], 
                                   num_workers=config['batch_size']*6,
                                   pin_memory=True,
                                   sampler=train_sampler, # Use the distributed sampler
                                   collate_fn=collate_train)
    
    test_data = DataGenerator(config['test_data_dir'], 'all_with_indices', mode='test')
    test_dataset = Dataset(test_data)
    test_loader = data.DataLoader(test_dataset, 
                                  batch_size=1, 
                                  num_workers=6, 
                                  pin_memory=True,
                                  shuffle=False,
                                  collate_fn=collate_test)
    
    if local_rank == 0:
        print('Data Loading Done.')
    
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
        checkpoint_dir = Path(config['log_dir']) / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        test(model.module, feature_gen, test_loader, device, config)

    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()
        
        train_loader.sampler.set_epoch(epoch)
        
        total_loss_on_proc = train(model, feature_gen, loss_fn, train_loader, optimizer, device, config)
        
        total_loss_tensor = torch.tensor(total_loss_on_proc).to(device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = total_loss_tensor.item() / len(train_loader.dataset)
        
        if local_rank == 0:
            f1, p, r = test(model.module, feature_gen, test_loader, device, config)
            
            end_time = time.time()
            epoch_time = end_time - start_time

            print(f"Epoch {epoch}/{config['epochs']} | Loss: {avg_loss:.6f} | F1: {f1:.6f} | Time: {epoch_time:.2f}s")

            # WandB Log
            wandb.run.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "epoch_time": epoch_time
            }, step=epoch)
    
            ckpt_path = checkpoint_dir / f"StepFold_E{epoch}.pt"
            torch.save(model.module.state_dict(), ckpt_path)
            print(f"New model saved at {ckpt_path}")
        
        dist.barrier()

    if local_rank == 0:
        print(f"\nTraining finished.")
        wandb.finish()
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
    # Running in the terminal: CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc_per_node=2 code/train_S1.py