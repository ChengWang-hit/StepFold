# StepFold: A Progressive Local-to-Global Generation Framework for RNA Secondary Structure Prediction

[![Zenodo DOI](https://img.shields.io/badge/Zenodo-10.5281/zenodo.19556209-blue.svg)](https://doi.org/10.5281/zenodo.19556209)
[![Code Ocean DOI](https://img.shields.io/badge/Code%20Ocean-10.24433/CO.9487656.v1-blue.svg)](https://doi.org/10.24433/CO.9487656.v1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation for the paper: **''Cheng Wang, Haozhuo Zheng, Gaurav Sharma, Maozu Guo, Quan Zou and Yang Liu. StepFold: A Progressive Local-to-Global Generation Framework for RNA Secondary Structure Prediction.''**

![StepFold Architecture](/StepFold.png)

**StepFold** is a novel, multi-step generation framework designed for accurate and efficient RNA secondary structure prediction. Unlike conventional "one-step" deep learning methods, StepFold mimics the hierarchical nature of RNA folding by predicting structures progressively—from local substructures to global long-range interactions.

-----

## ⚡ Quick Reproducibility via Code Ocean

For an immediate, zero-setup experience, we provide an interactive compute capsule on Code Ocean: **[https://doi.org/10.24433/CO.9487656.v1](https://doi.org/10.24433/CO.9487656.v1)**.

This capsule enables rapid reproduction of paper results and allows users to perform RNA secondary structure prediction on new sequences using the pre-trained parameters directly within your browser.

-----

## 📑 Table of Contents

- [Preparation Workflow](#-preparation-workflow)
  - [1. Clone & Environment Setup](#1-clone--environment-setup)
  - [2. Download Datasets & Checkpoints](#2-download-datasets--checkpoints)
  - [3. Preprocessing](#3-preprocessing)
- [Reproducing Paper Results](#-reproducing-paper-results)
- [Inference on New Sequences](#-inference-on-new-sequences)
- [Training from Scratch](#-training-from-scratch)

-----

## 🛠 Preparation Workflow

### 1\. Clone & Environment Setup

**Author's Implementation:**
* **OS:** Ubuntu 22.04 (Linux kernel 6.8.0)
* **GPU & CUDA:** NVIDIA GPU with Driver 560.35.03, CUDA 12.6

First, clone the repository to your local machine and navigate into the project directory:

```bash
git clone https://github.com/ChengWang-hit/StepFold.git
cd StepFold
```

We use [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package and project manager, to configure the environment.

If you don't have `uv` installed yet, you can install it quickly using `pip` (or refer to their [official installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other methods):

```bash
pip install uv
```

Once `uv` is installed, run the following command in the project root to synchronize and install all dependencies specified in `pyproject.toml`:

```bash
# Install all required dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```

### 2\. Download Datasets & Checkpoints

The complete datasets and pre-trained model checkpoints are hosted on Zenodo.

  * **Download Link:** [10.5281/zenodo.19556209](https://doi.org/10.5281/zenodo.19556209)

After downloading and extracting the archives, please organize them in the root directory following this exact structure:

```text
StepFold/
├── ckpt/
│   ├── S1.pt
│   ├── S1_aug.pt
│   ├── S2.pt
│   ├── S3.pt
│   ├── S4.pt
│   └── training_all.pt
├── code/
├── configs/
├── data/
│   ├── ArchiveII/
│   ├── bpRNA_1m/
│   ├── bpRNA_new/
│   ├── PDB/
│   └── RNAStralign/
├── pyproject.toml           
└── README.md                
```

### 3\. Preprocessing

Before running training or inference, you must generate the mask matrices required by the model. Run the following script:

```bash
python code/generate_mask_matrix.py
```

-----

## 📊 Reproducing Paper Results

To reproduce the comparative results presented in our paper for all scenarios (**S1 including data augmentation, S2, S3, and S4**), you can simply execute the test script:

```bash
python code/test_all.py
```

This script will automatically load the checkpoints, evaluate them against the test datasets, and print out the F1-scores, precision, and recall metrics reported in the manuscript.

-----

## 🧬 Inference on New Sequences

You can easily use our pre-trained model to predict the secondary structure of your own custom RNA sequences.

**Step 1:** Add your RNA sequences to the demo FASTA file located at `inference_demo/demo.fasta`.

**Step 2:** Run the inference script:

```bash
python code/inference_fasta.py
```

**Step 3:** The predicted secondary structure results will be saved and output to the `inference_demo/output/` directory.

-----

## 🚀 Training from Scratch

If you wish to retrain the models from scratch (S1 through S4) to reproduce the training process, execute the corresponding training scripts:

```bash
# Train Stage 1
python code/train_S1.py

# Train Stage 2
python code/train_S2.py

# Train Stage 3
python code/train_S3.py

# Train Stage 4
python code/train_S4.py
```