import os
import pickle
import numpy as np
from tqdm import tqdm

# List of dataset groups, each group is a list of pickle file paths
INPUT_PICKLE_PATHS = [
    [
        'data/PDB/TR1.pickle',
        'data/PDB/VL1.pickle',
        'data/PDB/TS1.pickle',
        'data/PDB/TS2.pickle',
        'data/PDB/TS3.pickle',
        'data/PDB/TS_hard.pickle',
    ],
    [
        'data/ArchiveII/all_max600.pickle',
    ],
    [
        'data/bpRNA_1m/train.pickle',
        'data/bpRNA_1m/val.pickle',
        'data/bpRNA_1m/test.pickle',
    ],
    [
        'data/bpRNA_new/all.pickle',
    ],
    [
        'data/RNAStralign/train_max600.pickle',
        'data/RNAStralign/val_max600.pickle',
        'data/RNAStralign/test_max600.pickle',
    ],
]

# Output directory for the new pickle files with mask indices
OUTPUT_DIRS = [
    'data/PDB/',
    'data/ArchiveII/',
    'data/bpRNA_1m/',
    'data/bpRNA_new/',
    'data/RNAStralign/',
]

# Directory for storing individual mask_matrix pickle files
MASK_MATRIX_SAVE_DIRS = [
    'data/PDB/mask_matrix/',
    'data/ArchiveII/mask_matrix/',
    'data/bpRNA_1m/mask_matrix/',
    'data/bpRNA_new/mask_matrix/',
    'data/RNAStralign/mask_matrix/',
]

def create_mask_matrix(sequence, min_loop_length=3):
    """
    Generate allowed base-pair coordinates (upper triangle only) for an RNA sequence.

    Allowed pairs:
        A-U, U-A, G-C, C-G, G-U, U-G

    Args:
        sequence (str): RNA sequence.
        min_loop_length (int): Minimum loop length constraint.

    Returns:
        np.ndarray: Array of shape (N, 2), where each row is [i, j].
    """
    sequence = sequence.upper().replace('T', 'U')
    L = len(sequence)

    if L == 0:
        return np.empty((0, 2), dtype=np.int32)

    seq_array = np.array(list(sequence))
    seq_row = seq_array.reshape(L, 1)
    seq_col = seq_array.reshape(1, L)

    mask_AU = (seq_row == 'A') & (seq_col == 'U')
    mask_GC = (seq_row == 'G') & (seq_col == 'C')
    mask_GU = (seq_row == 'G') & (seq_col == 'U')

    pairing_mask = mask_AU | mask_GC | mask_GU
    full_pairing_mask = pairing_mask | pairing_mask.T

    indices = np.arange(L)
    dist_matrix = indices.reshape(1, L) - indices.reshape(L, 1)
    loop_constraint_mask = dist_matrix > min_loop_length

    final_mask = full_pairing_mask & loop_constraint_mask

    indices_i, indices_j = np.where(final_mask)
    pairing_coordinates = np.stack([indices_i, indices_j], axis=1)

    return pairing_coordinates.astype(np.int32)


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def normalize_group(group):
    """
    Normalize one dataset group into a list of file paths.
    Accepts either a string or a list/tuple of strings.
    """
    if isinstance(group, str):
        return [group]
    if isinstance(group, (list, tuple)):
        return list(group)
    raise TypeError(f"Unsupported group type: {type(group)}")


def process_dataset_group(input_files, output_dir, mask_matrix_save_dir, min_loop_length=3):
    """
    Process one dataset group:
      1. Collect all sequences from all pickle files in the group
      2. Generate and save mask matrices individually
      3. Create new pickle files with 'mask_matrix_idx'

    Args:
        input_files (list[str]): List of pickle file paths for one dataset group
        output_dir (str): Directory to save new pickle files
        mask_matrix_save_dir (str): Directory to save mask matrix pickle files
        min_loop_length (int): Minimum loop length constraint
    """
    ensure_dir(output_dir)
    ensure_dir(mask_matrix_save_dir)

    print("=" * 70)
    print(f"Processing dataset group")
    print(f"Output dir: {output_dir}")
    print(f"Mask dir:   {mask_matrix_save_dir}")
    print("=" * 70)

    # --------------------------------------------------------
    # Stage 1: Collect sequences and record file metadata
    # --------------------------------------------------------
    print("\n[Stage 1] Collecting sequences...")

    all_sequences = []
    file_metadata = []
    current_index = 0

    for file_path in tqdm(input_files, desc="Collecting sequences"):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if 'seq' not in data:
            raise KeyError(f"'seq' key not found in {file_path}")

        sequences_in_file = data['seq']
        num_sequences = len(sequences_in_file)

        all_sequences.extend(sequences_in_file)

        file_metadata.append({
            'path': file_path,
            'start_index': current_index,
            'num_sequences': num_sequences,
        })

        current_index += num_sequences

    print(f"Collected {len(all_sequences)} sequences in total.")

    # --------------------------------------------------------
    # Stage 2: Generate and save mask matrices
    # --------------------------------------------------------
    print("\n[Stage 2] Generating and saving mask matrices...")

    for idx, seq in enumerate(tqdm(all_sequences, desc="Generating masks")):
        mask_matrix_np = create_mask_matrix(seq, min_loop_length=min_loop_length)
        mask_matrix_list = mask_matrix_np.tolist()

        save_path = os.path.join(mask_matrix_save_dir, f"{idx}.pickle")
        with open(save_path, 'wb') as f:
            pickle.dump(mask_matrix_list, f)

    print(f"All mask matrices have been saved to: {mask_matrix_save_dir}")

    # --------------------------------------------------------
    # Stage 3: Create new pickle files with mask_matrix_idx
    # --------------------------------------------------------
    print("\n[Stage 3] Creating new pickle files with mask indices...")

    for meta in tqdm(file_metadata, desc="Creating new pickle files"):
        original_path = meta['path']
        start_idx = meta['start_index']
        num_seqs = meta['num_sequences']

        with open(original_path, 'rb') as f:
            original_data = pickle.load(f)

        mask_indices_for_file = list(range(start_idx, start_idx + num_seqs))
        original_data['mask_matrix_idx'] = mask_indices_for_file

        base_name = os.path.basename(original_path)
        file_name, file_ext = os.path.splitext(base_name)
        new_file_name = f"{file_name}_with_indices{file_ext}"
        new_file_path = os.path.join(output_dir, new_file_name)

        with open(new_file_path, 'wb') as f:
            pickle.dump(original_data, f)

    print("Finished processing this dataset group.\n")
    

if __name__ == "__main__":
    # if not (len(INPUT_PICKLE_PATHS) == len(OUTPUT_DIRS) == len(MASK_MATRIX_SAVE_DIRS)):
    #     raise ValueError(
    #         "INPUT_PICKLE_PATHS, OUTPUT_DIRS, and MASK_MATRIX_SAVE_DIRS "
    #         "must have the same length."
    #     )

    # for group_idx, (input_group, output_dir, mask_dir) in enumerate(
    #     zip(INPUT_PICKLE_PATHS, OUTPUT_DIRS, MASK_MATRIX_SAVE_DIRS)
    # ):
    #     input_files = normalize_group(input_group)

    #     print(f"\nStarting dataset group {group_idx + 1}/{len(INPUT_PICKLE_PATHS)}")
    #     process_dataset_group(
    #         input_files=input_files,
    #         output_dir=output_dir,
    #         mask_matrix_save_dir=mask_dir,
    #         min_loop_length=3,
    #     )

    # print("=" * 70)
    # print("All dataset groups have been processed successfully.")
    # print("Original files were kept unchanged.")
    # print("New pickle files with 'mask_matrix_idx' have been created.")
    # print("=" * 70)
    
    # merge PDB test sets.
    import _pickle as cPickle
    import os

    data_list = [
        'data/PDB/TS1_with_indices.pickle',
        'data/PDB/TS2_with_indices.pickle',
        'data/PDB/TS3_with_indices.pickle',
        'data/PDB/TS_hard_with_indices.pickle',
    ]

    output_path = 'data/PDB/TS123_hard_with_indices.pickle'

    merged_data = {
        'seq': [],
        'ss': [],
        'mask_matrix_idx': [],
    }

    for data_path in data_list:
        with open(data_path, 'rb') as f:
            data = cPickle.load(f)

        required_keys = ['seq', 'ss', 'mask_matrix_idx']

        seq_list = data['seq']
        ss_list = data['ss']
        mask_idx_list = data['mask_matrix_idx']

        merged_data['seq'].extend(seq_list)
        merged_data['ss'].extend(ss_list)
        merged_data['mask_matrix_idx'].extend(mask_idx_list)

    with open(output_path, 'wb') as f:
        cPickle.dump(merged_data, f)

    print(f'Merged file saved to: {output_path}')
    print(f"Total sequences: {len(merged_data['seq'])}")
    print(f"Total structures: {len(merged_data['ss'])}")
    print(f"Total mask indices: {len(merged_data['mask_matrix_idx'])}")