import os
import _pickle as cPickle
from torch.utils import data
from utils import BASE_TO_INT, pairs2map, seq2set
import torch
import numpy as np

class DataGenerator(object):
    """
    StepFold data generator: loads RNA sequences, secondary structures.
    """
    def __init__(self, data_dir, split, mode='train', family_fold=False, data_num=999999):
        self.data_dir = data_dir
        self.split = split
        self.mode = mode
        if family_fold:
            self.mask_dir = self.data_dir.replace('/family_fold', '')
        else:
            self.mask_dir = self.data_dir
        self.mask_dir = os.path.join(self.mask_dir, 'mask_matrix')
        self.max_len = 0
        self.load_data(data_num)

    def load_data(self, data_num):
        data_dir = self.data_dir

        file_name = f'{self.split}_with_stability.pickle' 
        if not os.path.exists(os.path.join(data_dir, file_name)):
             file_name = f'{self.split}.pickle'

        with open(os.path.join(data_dir, file_name), 'rb') as f:
            self.data = cPickle.load(f)

        self.seq = [seq.upper() for seq in self.data['seq'][:data_num]]
        self.ss = self.data['ss'][:data_num]
        raw_idx = self.data.get('mask_matrix_idx', [None] * len(self.seq))[:data_num] 
        self.mask_matrix_idx = [(idx, self.mask_dir) for idx in raw_idx]
        
        self.len = len(self.seq)
        self.seq_len = [len(s) for s in self.seq]
        self.max_len = max(self.seq_len)
        self.mean_len = sum(self.seq_len) / self.len
        print(f"Loaded {self.len} samples from {file_name}. Mode: {self.mode}")

    def merge(self, augmentation):
        self.seq = self.seq + augmentation.seq
        self.ss = self.ss + augmentation.ss
        self.mask_matrix_idx = self.mask_matrix_idx + augmentation.mask_matrix_idx
        self.seq_len = self.seq_len + augmentation.seq_len
        self.max_len = max(self.seq_len)
        self.mean_len = sum(self.seq_len) / len(self.seq_len)
        self.len = len(self.seq)
        return self

    def get_one_sample(self, index):
        seq = self.seq[index]
        seq_length = len(seq)
        seq_ints = [BASE_TO_INT.get(s.upper(), 4) for s in seq]
        
        # Ground-truth contact map (G)
        cm_pairs = np.array(self.ss[index])
        # if len(cm_pairs) == 0:
        #     cm_pairs = np.array([[0, seq_length-1], [seq_length-1, 0]])
        contact_map = pairs2map(cm_pairs, seq_length) # Ground-truth secondary structure matrix G
        
        mask_idx, mask_dir = self.mask_matrix_idx[index]
        with open(os.path.join(mask_dir, f'{mask_idx}.pickle'), 'rb') as f:
            pred_pairs = torch.LongTensor(cPickle.load(f)).T # (2, N), asymmetric
            mask_matrix = torch.zeros((seq_length, seq_length))
            if pred_pairs.shape[0] > 0:
                mask_matrix[pred_pairs[0, :], pred_pairs[1, :]] = 1
            mask_matrix = mask_matrix + mask_matrix.T

        node_set1 = seq2set(seq)

        if self.mode == 'train':
            result = (
                torch.LongTensor(seq_ints),      # 0: Integer-encoded sequence
                torch.Tensor(mask_matrix),      # 1: Mask Matrix (L x L) - constraint matrix
                torch.Tensor(contact_map),      # 2: Ground-truth Contact Map (G)
                seq_length,                     # 3: Sequence length
            )
        else:
            result = (
                torch.LongTensor(seq_ints),      # 0: Integer-encoded sequence
                torch.Tensor(mask_matrix),      # 1: Mask Matrix (L x L) - constraint matrix
                torch.Tensor(contact_map),      # 2: Ground-truth contact map (G)
                seq_length,                     # 3: Sequence length
                node_set1,                      # 4: Bipartite node set (used for MWM)
            )
        
        return result

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data):
        'Initialization'
        self.data = data

    def __len__(self):
        'Denotes the total number of samples'
        return self.data.len

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data.get_one_sample(index)