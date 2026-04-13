import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

from network import StepFoldNet, HelixCenterMaskedPriorGenerator
from utils import create_dynamic_start_band_masks

os.chdir('/home/wangcheng/project/RNA/StepFold/github_version/StepFold_open_source')

ARCH_CONFIG_PATH = "configs/Architecture.json"

CHECKPOINT_PATH = "ckpt/training_all.pt"

BASE_TO_INT = {"A": 0, "U": 1, "C": 2, "G": 3, "N": 4}

CANONICAL_PAIRS_TENSOR = torch.tensor(
    [
        [0, 1, 0, 0, 0],  # A
        [1, 0, 0, 1, 0],  # U
        [0, 0, 0, 1, 0],  # C
        [0, 1, 1, 0, 0],  # G
        [0, 0, 0, 0, 0],  # N
    ],
    dtype=torch.bool,
)

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["K_total"] = config["K_local"] + config["K_global"]

    return config


def load_model_weights(model, ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)


def make_seq_dir(root, name):
    candidate = root / name
    if not candidate.exists():
        return candidate


def parse_fasta(fasta_path):
    records = []
    name = None
    seq_chunks = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if name is not None:
                    seq = "".join(seq_chunks).upper().replace("T", "U")
                    records.append((name, seq))
                name = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)

    if name is not None:
        seq = "".join(seq_chunks).upper().replace("T", "U")
        records.append((name, seq))

    return records


def validate_sequence(seq):
    seq = seq.upper().replace("T", "U")
    valid = {"A", "U", "C", "G", "N"}
    cleaned = []
    for ch in seq:
        cleaned.append(ch if ch in valid else "N")
    return "".join(cleaned)


def sequence_to_tensor_and_legal_mask(seq, min_loop_length, device):
    seq = validate_sequence(seq)
    L = len(seq)

    seq_ints = [BASE_TO_INT.get(ch, 4) for ch in seq]
    seq_tensor = torch.tensor([seq_ints], dtype=torch.long, device=device)  # (1, L)

    row = seq_tensor.unsqueeze(2).expand(1, L, L)
    col = seq_tensor.unsqueeze(1).expand(1, L, L)

    canonical_mask = CANONICAL_PAIRS_TENSOR.to(device)[row, col]

    idx = torch.arange(L, device=device)
    dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    loop_mask = dist > min_loop_length

    legal_mask = (canonical_mask & loop_mask.unsqueeze(0)).float()  # (1, L, L)

    return seq_tensor, legal_mask


def maximum_weight_matching_postprocess(prob_matrix, threshold):
    """
    prob_matrix: (L, L), torch.Tensor on any device
    return: binary symmetric contact map (L, L), torch.Tensor on same device
    """
    device = prob_matrix.device
    mat = prob_matrix.detach().cpu().numpy()
    L = mat.shape[0]

    G = nx.Graph()
    G.add_nodes_from(range(L))

    for i in range(L):
        for j in range(i + 1, L):
            w = float(mat[i, j])
            if w > threshold:
                G.add_edge(i, j, weight=w)

    matching = nx.max_weight_matching(G, maxcardinality=False, weight="weight")

    binary = np.zeros((L, L), dtype=np.float32)
    for i, j in matching:
        binary[i, j] = 1.0
        binary[j, i] = 1.0

    return torch.tensor(binary, dtype=torch.float32, device=device)


def contact_map_to_bpseq(seq, binary_contact_map, out_path):
    L = len(seq)
    partners = np.zeros(L, dtype=np.int32)

    for i in range(L):
        paired = np.where(binary_contact_map[i] > 0.5)[0]
        if len(paired) > 1:
            # This should not happen after matching, but keep a safe fallback
            best_j = paired[np.argmax(binary_contact_map[i, paired])]
            partners[i] = best_j + 1
        elif len(paired) == 1:
            partners[i] = paired[0] + 1
        else:
            partners[i] = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, base in enumerate(seq, start=1):
            f.write(f"{i} {base} {partners[i - 1]}\n")


def extract_pairs_from_binary_map(binary_contact_map):
    pairs = []
    L = binary_contact_map.shape[0]
    for i in range(L):
        for j in range(i + 1, L):
            if binary_contact_map[i, j] > 0.5:
                pairs.append((i, j))
    return pairs


def save_probability_heatmap(prob_matrix, out_path, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(prob_matrix, cmap="viridis", origin="upper", aspect="auto")
    plt.colorbar(label="Pairing probability")
    plt.title(title)
    plt.gca().xaxis.tick_top()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_arc_plot(seq, binary_contact_map, out_path, title):
    pairs = extract_pairs_from_binary_map(binary_contact_map)
    L = len(seq)

    fig_width = max(10, min(24, L / 8))
    fig_height = 6
    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()

    x = np.arange(1, L + 1)

    # Baseline
    ax.plot([1, L], [0, 0], linewidth=1.2)

    # Nodes
    ax.scatter(x, np.zeros_like(x), s=12)

    # Labels
    if L <= 120:
        for i, base in enumerate(seq, start=1):
            ax.text(i, -0.06 * max(1, L / 20), base, ha="center", va="top", fontsize=8)

    # Draw arcs
    max_radius = 1.0
    for i, j in pairs:
        xi = i + 1
        xj = j + 1
        center = (xi + xj) / 2.0
        radius = (xj - xi) / 2.0
        t = np.linspace(0, np.pi, 200)
        xs = center + radius * np.cos(t)
        ys = radius * np.sin(t)
        ax.plot(xs, ys, linewidth=1.4)
        max_radius = max(max_radius, radius)

    # ax.set_xlim(0, L + 1)
    ax.set_ylim(-0.15 * max_radius, 1.1 * max_radius)
    ax.set_title(title)
    # ax.set_xlabel("Sequence position")
    # ax.set_ylabel("Arc height")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    
    ax.set_xticks([])
    ax.set_yticks([])

    # if L > 120:
    #     tick_step = max(10, L // 10)
    #     ax.set_xticks(np.arange(1, L + 1, tick_step))
    # else:
    #     tick_step = max(5, L // 12 if L >= 12 else 1)
    #     ax.set_xticks(np.arange(1, L + 1, tick_step))
    
    # 标 5' 和 3'
    label_y = -0.08 * max_radius
    ax.text(1, label_y, "5′", ha="center", va="top", fontsize=12)
    ax.text(L, label_y, "3′", ha="center", va="top", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_sequence_fasta(name, seq, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i + 80] + "\n")


@torch.no_grad()
def predict_one_sequence(seq, model, feature_gen, device, config):
    seq = validate_sequence(seq)
    L = len(seq)

    seq_tensor, legal_mask = sequence_to_tensor_and_legal_mask(
        seq=seq,
        min_loop_length=3,
        device=device
    )

    padding_mask = torch.ones((1, L, L), dtype=torch.float32, device=device)
    seq_length = torch.tensor([L], dtype=torch.long, device=device)

    helix_prior = feature_gen(seq_tensor, legal_mask)

    band_masks = create_dynamic_start_band_masks(
        seq_lengths=seq_length,
        K_local=config["K_local"],
        K_global=config["K_global"],
        min_distance=config["min_distance"],
        max_distance=config["max_distance"],
        start_ratio=config["start_ratio"],
        power=config["growth_power"],
        L_max=L,
        device=device,
    )

    input_data = (helix_prior, band_masks, legal_mask, padding_mask)
    pred_logits = model.inference(input_data)      # (1, L, L)
    prob_matrix = torch.sigmoid(pred_logits) * legal_mask.float()
    prob_matrix = prob_matrix[0, :L, :L]

    binary_contact_map = maximum_weight_matching_postprocess(
        prob_matrix=prob_matrix,
        threshold=config["threshold"],
    )

    return prob_matrix.detach().cpu().numpy(), binary_contact_map.detach().cpu().numpy()


def build_model_and_feature_generator(config, device):
    feature_gen = HelixCenterMaskedPriorGenerator(K=config["helices_num"]).to(device)

    model = StepFoldNet(
        K_local=config["K_local"],
        K_global=config["K_global"],
        hidden_dim=config["embedding_dim"],
        helix_prior_K=config["helices_num"],
        ff_kernel_size=config["ff_kernel_size"],
        ff_expansion=config["ff_expansion"],
        ff_depth=config["ff_depth"],
    ).to(device)

    load_model_weights(model, CHECKPOINT_PATH, device)

    model.eval()
    feature_gen.eval()

    return model, feature_gen


def process_fasta_file(fasta_path, output_root, device):
    config = load_config(ARCH_CONFIG_PATH)
    model, feature_gen = build_model_and_feature_generator(config, device)

    records = parse_fasta(fasta_path)
    if len(records) == 0:
        raise ValueError(f"No FASTA records found in: {fasta_path}")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(records)} sequence(s) from {fasta_path}")
    print(f"Results will be saved to: {output_root.resolve()}")
    print(f"Using checkpoint: {CHECKPOINT_PATH}")
    print(f"Using device: {device}")

    for idx, (name, seq) in enumerate(records, start=1):
        seq = validate_sequence(seq)
        seq_dir = make_seq_dir(output_root, name)
        seq_dir.mkdir(parents=True, exist_ok=False)

        print(f"\n[{idx}/{len(records)}] Processing: {name}")
        print(f"Sequence length: {len(seq)}")

        prob_matrix, binary_contact_map = predict_one_sequence(
            seq=seq,
            model=model,
            feature_gen=feature_gen,
            device=device,
            config=config,
        )

        # Save raw sequence
        save_sequence_fasta(name=name, seq=seq, out_path=str(seq_dir / "sequence.fasta"))

        # Save matrices
        np.save(seq_dir / "probability_matrix.npy", prob_matrix)
        np.save(seq_dir / "binary_contact_map.npy", binary_contact_map)

        # Save BPSEQ
        contact_map_to_bpseq(
            seq=seq,
            binary_contact_map=binary_contact_map,
            out_path=str(seq_dir / "prediction.bpseq"),
        )

        # Save figures
        save_probability_heatmap(
            prob_matrix=prob_matrix,
            out_path=str(seq_dir / "probability_heatmap.png"),
            title=f"{name} - Pairing Probability Matrix",
        )

        save_arc_plot(
            seq=seq,
            binary_contact_map=binary_contact_map,
            out_path=str(seq_dir / "structure_arc.png"),
            title=f"{name} - Predicted Secondary Structure",
        )

        # Save summary
        num_pairs = int(np.sum(np.triu(binary_contact_map, k=1)))
        summary = {
            "name": name,
            "length": len(seq),
            "num_pairs": num_pairs,
            "threshold": config["threshold"],
            "checkpoint_path": CHECKPOINT_PATH,
            "arch_config_path": ARCH_CONFIG_PATH,
            "files": {
                "fasta": "sequence.fasta",
                "bpseq": "prediction.bpseq",
                "probability_matrix_npy": "probability_matrix.npy",
                "binary_contact_map_npy": "binary_contact_map.npy",
                "probability_heatmap_png": "probability_heatmap.png",
                "structure_arc_png": "structure_arc.png",
            },
        }

        with open(seq_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Saved results to: {seq_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run StepFold inference on a FASTA file.")
    parser.add_argument("--fasta", type=str, default="inference_demo/demo.fasta", help="Input FASTA file")
    parser.add_argument("--output", type=str, default="inference_demo/output", help="Output root directory")
    parser.add_argument("--device", type=str, default="cuda:4", help='Device, e.g. "cuda:0" or "cpu"')
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    process_fasta_file(
        fasta_path=args.fasta,
        output_root=args.output,
        device=device,
    )


if __name__ == "__main__":
    main()