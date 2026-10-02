import numpy as np
import torch
import networkx as nx
import random
import os
import torch
import json
import torch.distributed as dist


BASE_TO_INT = {'A': 0, 'U': 1, 'C': 2, 'G': 3, 'N': 4}

CANONICAL_PAIRS_TENSOR = torch.tensor([
    [0, 1, 0, 0, 0], # A
    [1, 0, 0, 1, 0], # U
    [0, 0, 0, 1, 0], # C
    [0, 1, 1, 0, 0], # G
    [0, 0, 0, 0, 0]  # N
], dtype=torch.bool)

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def init_ddp():
    if 'LOCAL_RANK' not in os.environ:
        print("⚠️ Debug Mode: Running on Single GPU")
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
                            backend="nccl", 
                            device_id=torch.device(f"cuda:{local_rank}")
                            )

def process_config(arch_config_path, train_config_path):
    with open(arch_config_path, 'r', encoding='utf-8') as f:
        arch_config = json.load(f)
    
    with open(train_config_path, 'r', encoding='utf-8') as f:
        train_config = json.load(f)
    
    # Merge configuration
    config = {**arch_config, **train_config}
    config['K_total'] = config['K_local'] + config['K_global']
    
    return config

def seq_to_indices(seq_list, max_len=None):
    """Convert a list of character sequences into an integer tensor of shape (B, L)."""
    if max_len is None:
        max_len = max(len(s) for s in seq_list)
    
    batch_size = len(seq_list)
    seq_tensor = torch.full((batch_size, max_len), 4, dtype=torch.long) # 4 is 'N'
    
    for i, seq in enumerate(seq_list):
        ints = [BASE_TO_INT.get(c.upper(), 4) for c in seq]
        seq_tensor[i, :len(ints)] = torch.tensor(ints, dtype=torch.long)
        
    return seq_tensor

def pairs2map(pairs, seq_len):
    if np.array(pairs).size == 0:
        return torch.zeros([seq_len, seq_len])
    contact = torch.zeros([seq_len, seq_len])
    idx = torch.LongTensor(pairs).T
    contact[idx[0], idx[1]] = 1
    return contact

# Split nodes into bipartite sets {A, G} and {C, U}
def seq2set(seq):
    set1 = {'A', 'G'}
    node_set1 = []
    for i, s in enumerate(seq):
        if s.upper() in set1:
            node_set1.append(i)
    return node_set1

def outer_concat(t1, t2):
    seq_len = t1.shape[1]
    a = t1.unsqueeze(-2).expand(-1, -1, seq_len, -1)
    b = t2.unsqueeze(-3).expand(-1, seq_len, -1, -1)
    return torch.concat((a, b), dim=-1)

show_band = True

def create_dynamic_start_band_masks(seq_lengths, K_local, K_global, min_distance, max_distance, start_ratio, power, L_max, device):
    """
    Generate band masks whose initial bandwidth is dynamically adjusted according to sequence length.
    Formula: start_distance = clamp(L * start_ratio, min_distance, max_distance)
    
    Args:
        seq_lengths: (B,) actual lengths of each sequence
        K_local: number of local layers (bandwidth grows from start_dist to L)
        K_global: number of global layers (bandwidth fixed to full length L)
        min_distance: minimum initial bandwidth
        max_distance: maximum initial bandwidth
        start_ratio: initial bandwidth ratio
        power: growth exponent (control expansion speed)
        L_max: maximum sequence length
        device: device
    """
    K_local = int(K_local)
    K_global = int(K_global)
    
    B = seq_lengths.shape[0]
    L = seq_lengths.float()
    
    # --- 1. Compute the dynamic initial bandwidth ---
    # clamp it to the range [min_distance, max_distance]
    dynamic_start = torch.clamp(L * start_ratio, min=float(min_distance), max=float(max_distance))
    safe_start = torch.minimum(dynamic_start, L)
    
    # --- 2. Compute the local layer bandwidths (B, K_local) ---
    steps = torch.arange(K_local, device=device).float() / (K_local - 1 if K_local > 1 else 1)
    progress = torch.pow(steps, power) # (K_local,)

    # The bandwidth grows from each sequence's safe_start to the full length L
    diff = L - safe_start # (B,)
    local_bandwidths = safe_start.unsqueeze(1) + diff.unsqueeze(1) * progress.unsqueeze(0)

    # --- 3. Compute the Global layer bandwidth (B, K_global) ---
    global_bandwidths = L.unsqueeze(1).expand(-1, K_global)

    # --- 4. Concatenate and generate the final mask ---
    batch_bandwidths = torch.cat([local_bandwidths, global_bandwidths], dim=1) # (B, K_total)

    # Generate the distance matrix
    indices = torch.arange(L_max, device=device).float()
    dist_matrix = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1)) 
    
    band_masks = (dist_matrix.unsqueeze(0).unsqueeze(0) <= batch_bandwidths.unsqueeze(-1).unsqueeze(-1)).float()
    
    global show_band
    if show_band:
        print("Dynamic Bandwidths (Initial Step):", batch_bandwidths.int().tolist())
        show_band = False
    
    # Padding
    valid_pos = indices.unsqueeze(0) < seq_lengths.unsqueeze(1)
    valid_matrix = (valid_pos.unsqueeze(1) & valid_pos.unsqueeze(2)).float().unsqueeze(1)
    
    return band_masks * valid_matrix

def collate_train(data_list):
    batch_size = len(data_list)
    seq_lengths = [d[3] for d in data_list]
    max_len = max(seq_lengths)
    
    # Padding (4 is 'N')
    seq_batch = torch.full((batch_size, max_len), 4, dtype=torch.long)
    mask_matrix_batch = torch.zeros((batch_size, max_len, max_len), dtype=torch.float)
    contact_map_batch = torch.zeros((batch_size, max_len, max_len), dtype=torch.float)
    padding_mask_batch = torch.zeros((batch_size, max_len, max_len), dtype=torch.float)

    for i, (seq_ints, mask_matrix, contact_map, L) in enumerate(data_list):
        seq_batch[i, :L] = seq_ints
        mask_matrix_batch[i, :L, :L] = mask_matrix
        contact_map_batch[i, :L, :L] = contact_map
        padding_mask_batch[i, :L, :L] = 1.0
        
    return (seq_batch, 
            mask_matrix_batch, 
            contact_map_batch,
            padding_mask_batch, 
            torch.tensor(seq_lengths))

def collate_test(data_list):
    batch_size = len(data_list)
    seq_lengths = [d[3] for d in data_list]
    max_len = max(seq_lengths)
    
    node_set1_list = [d[4] for d in data_list]

    # Padding value = 4 for 'N'
    seq_batch = torch.full((batch_size, max_len), 4, dtype=torch.long)
    mask_matrix_batch = torch.zeros((batch_size, max_len, max_len), dtype=torch.float)
    contact_map_batch = torch.zeros((batch_size, max_len, max_len), dtype=torch.float)
    padding_mask_batch = torch.zeros((batch_size, max_len, max_len), dtype=torch.float)

    for i, (seq_ints, mask_matrix, contact_map, L, _) in enumerate(data_list):
        seq_batch[i, :L] = seq_ints
        mask_matrix_batch[i, :L, :L] = mask_matrix
        contact_map_batch[i, :L, :L] = contact_map
        padding_mask_batch[i, :L, :L] = 1.0

    return (
        seq_batch,          
        mask_matrix_batch,  
        contact_map_batch,  
        torch.tensor(seq_lengths, dtype=torch.long), 
        padding_mask_batch,
        node_set1_list
    )

def evaluate(pred_a, true_a, eps=1e-11):
    tp = torch.sign(torch.Tensor(pred_a)*torch.Tensor(true_a)).sum()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = (tp + eps)/(tp+fn+eps)
    precision = (tp + eps)/(tp+fp+eps)
    f1_score = (2*tp + eps)/(2*tp + fp + fn + eps)
    return precision.item(), recall.item(), f1_score.item()

def post_process_argmax(p, threshold=0.5):
        # Keep the largest value in each row
        max_values, max_indices = torch.max(p, dim=1)
        p = torch.where(p == max_values.view(-1, 1), p, torch.zeros_like(p))

        # Keep values greater than the threshold
        p = torch.where(p > threshold, torch.ones_like(p), torch.zeros_like(p))
        return p.to(p.device)

# hopcroft_karp algorithm
def post_process_HK(param):
    mat, node_set1, threshold = param

    # Keep values greater than the threshold
    mat = torch.where(mat > threshold, mat, torch.zeros_like(mat))
    n = mat.size(-1)
    G = nx.convert_matrix.from_numpy_array(np.array(mat.data.cpu()))
    # top_nodes = [v for i,v in enumerate(G.nodes) if bipartite_label[i] == 0]
    pairings = nx.bipartite.maximum_matching(G, top_nodes=node_set1)
    y_out = torch.zeros_like(mat)
    for (i, j) in pairings.items():
            if i>n and j>n:
                    continue
            y_out[i%n, j%n] = 1
            y_out[j%n, i%n] = 1
    return y_out.to(mat.device)

def post_process_maximum_weight_matching(param):
    mat, node_set1, threshold = param
    mat_np = mat.data.cpu().numpy()
    mat_np[mat_np <= threshold] = 0
    
    # Keep only the upper triangular part for MWM
    triu_indices = np.triu_indices_from(mat_np, k=1)
    G = nx.convert_matrix.from_numpy_array(mat_np)
    pairings_set = nx.max_weight_matching(G, maxcardinality=False, weight='weight')

    y_out = torch.zeros_like(mat)
    n = mat.size(-1)

    for u, v in pairings_set:
        if 0 <= u < n and 0 <= v < n:
            y_out[u, v] = 1
            y_out[v, u] = 1
    
    return y_out.to(mat.device)