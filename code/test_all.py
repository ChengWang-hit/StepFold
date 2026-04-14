import torch
from torch.utils import data
from tqdm import tqdm
import numpy as np
import json

from data_generator import DataGenerator, Dataset
from network import StepFoldNet, HelixCenterMaskedPriorGenerator
from utils import *
import os

def process_config(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)

    config['K_total'] = config['K_local'] + config['K_global']
    
    return config

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
                contact_map_pred = post_process_maximum_weight_matching(param)
                
                contact_map_label = contact_map[i, :L, :L]
                result.append(evaluate(contact_map_pred, contact_map_label))

    nt_exact_p, nt_exact_r, nt_exact_f1 = zip(*result)
    avg_p = np.average(nt_exact_p)
    avg_r = np.average(nt_exact_r)
    avg_f1 = np.average(nt_exact_f1)
 
    return avg_f1, avg_p, avg_r

def inference_data(dataset, split, ckpt, config, feature_gen, model, device):
    print(f'Dataset: {dataset} | Split: {split} | Checkpoint: {ckpt}')
    test_data = DataGenerator(os.path.join(config['data_dir'], dataset), f"{split}_with_indices", mode='test')
    test_dataset = Dataset(test_data)
    test_loader = data.DataLoader(test_dataset, 
                                  batch_size=1, 
                                  num_workers=6, 
                                  pin_memory=True,
                                  shuffle=False,
                                  collate_fn=collate_test)
    
    state_dict = torch.load(f"{config['ckpt_path']}/{ckpt}.pt", map_location=device)
    model.load_state_dict(state_dict)

    avg_f1, avg_p, avg_r = test(model, feature_gen, test_loader, device, config)
    print(f'Precision: {avg_p:.6f}, Recall: {avg_r:.6f}, F1: {avg_f1:.6f}')

def main():
    config_path = 'configs/Architecture.json'
    config = process_config(config_path)
    config['ckpt_path'] = 'ckpt'
    config['data_dir'] = 'data'
    device = torch.device(f"cuda:2")
    
    # Initialize the model
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
    print(f"Model #Params Num: {sum([x.nelement() for x in model.parameters()])}")  
    
    dataset_list = [('bpRNA_new', 'all'),
                    ('bpRNA_new', 'all'),
                    ('PDB', 'TS1'),
                    ('PDB', 'TS2'),
                    ('PDB', 'TS3'),
                    ('PDB', 'TS_hard'),
                    ('PDB', 'TS123_hard'),
                    ('ArchiveII', 'all_max600'),
                    ('bpRNA_1m', 'test'),
                    ]
    
    ckpt_list = ['S1', 'S1_aug', 'S2', 'S2', 'S2', 'S2', 'S2', 'S3', 'S4']
    
    for (dataset, split), ckpt in list(zip(dataset_list, ckpt_list))[8:9]:
        inference_data(dataset, split, ckpt, config, feature_gen, model, device)

if __name__ == '__main__':
    main()