# IGCN-BERT Baseline Models — Running Instructions

This project implements multiple baseline models for **Academic Paper Review Acceptance Prediction**, including Transformer-based pretrained models, Graph Neural Networks, and the proposed **IGCN-BERT** model.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Model List](#model-list)
- [General Usage](#general-usage)
- [Model-by-Model Instructions](#model-by-model-instructions)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [Results Log](#results-log)

---

## Environment Setup

### Conda Environment

It is recommended to use your own conda environment:

```bash
conda activate your_env
```

If you need to create a new environment from scratch, here are the core dependencies:

| Dependency | Version | Notes |
|------------|---------|-------|
| Python | 3.10+ | |
| PyTorch | 2.0+ | Match your CUDA version |
| transformers | 4.30+ | Pretrained model loading & tokenization |
| torch_geometric | 2.3+ | *DGCBERT only* |
| scikit-learn | 1.2+ | Metric computation |
| tqdm | 4.65+ | Progress bars |
| pandas | 2.0+ | Data processing |
| numpy | 1.24+ | Numerical computation |

### Quick Install

```bash
conda create -n your_env python=3.10
conda activate your_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers scikit-learn tqdm pandas numpy
# Only if you plan to run DGCBERT:
pip install torch_geometric
```

---

## Datasets

The project supports two academic paper review datasets. Both are **binary classification tasks** (accept / reject).

| Dataset | Train | Val | Test | Total | Classes | Max Seq Len |
|---------|-------|-----|------|-------|---------|-------------|
| **AAPR** | 26,980 | 3,373 | 3,373 | 33,726 | 2 | ... |
| **PeerRead** | 9,377 | 498 | 542 | 10,417 | 2 | ... |

### Data Directory Structure

```
data/
├── AAPR/
│   ├── train_contents.list      # Training set text
│   ├── train_labels.list        # Training set labels (0/1)
│   ├── train_indexes.list       # Training set indices
│   ├── val_contents.list        # Validation set text
│   ├── val_labels.list          # Validation set labels
│   ├── val_indexes.list         # Validation set indices
│   ├── test_contents.list       # Test set text
│   ├── test_labels.list         # Test set labels
│   ├── test_indexes.list        # Test set indices
│   ├── vocab.pth                # Vocabulary (auto-generated on first run)
│   └── cache/                   # Preprocessing cache (auto-generated)
│       ├── train_256.pth
│       ├── val_256.pth
│       └── test_256.pth
├── PeerRead/
│   └── (same structure as AAPR)
└── bert/
    ├── scibert-scivocab-uncased/   # SciBERT pretrained model
    └── bert-base-uncased/          # BERT pretrained model (optional)
```

### Cache Cleanup

Cache files are auto-generated on first run. If you switch datasets or upgrade the `transformers` version, clean the old cache:

```bash
# Windows PowerShell
del data\AAPR\cache\*.pth
del data\AAPR\vocab.pth
del data\PeerRead\cache\*.pth
del data\PeerRead\vocab.pth
```

---

## Model List

| Model | Script | Config Dir | Pretrained Model | Key Feature |
|-------|--------|------------|------------------|-------------|
| **BERT_CLS** | `run_bert_cls.py` | `configs/bert_cls/` | SciBERT | Standard BERT/SciBERT + linear classifier baseline |
| **SciBERT_CLS** | `run_scibert_cls.py` | `configs/scibert_cls/` | SciBERT | SciBERT fine-tuning baseline |
| **SciBERT_Concat** | `run_scibert_concat.py` | `configs/scibert_concat/` | SciBERT | Multi-layer output concatenation |
| **SciBERT_Max** | `run_scibert_max.py` | `configs/scibert_max/` | SciBERT | Multi-layer max-pooling fusion |
| **SciBERT_Gate** | `run_scibert_gate.py` | `configs/scibert_gate/` | SciBERT | Multi-layer gating fusion |
| **Transformer** | `run_transformer.py` | `configs/transformer/` | BERT tokenizer | Pure Transformer encoder (trained from scratch) |
| **BERT_GCN** | `run_bert_gcn.py` | `configs/bert_gcn/` | SciBERT | BERT features + Graph Convolutional Network |
| **BERT_MHAN** | `run_bert_mhan.py` | `configs/bert_mhan/` | BERT-base | Multi-Hop Attention Network |
| **TextGCN** | `run_textgcn.py` | `configs/textgcn/` | — | Heterogeneous graph text classification (no pretraining) |
| **DGCBERT** | `run_dgc_bert.py` | `configs/DGCBERT/` | SciBERT | Dynamic Graph Conv + BERT (requires torch_geometric) |
| **IGCN-BERT** | `run_igcn_bert.py` | `configs/IGCN-BERT/` | SciBERT | **Proposed in this work**: Interactive GCN + BERT |

---

## General Usage

### Train Mode

All models share a unified CLI interface:

```bash
conda activate your_env
python scripts/run_<model_name>.py --mode train --model <config_dir_name> [--data_source AAPR|PeerRead]
```

### Test Mode

```bash
python scripts/run_<model_name>.py --mode test --model <config_dir_name> [--data_source AAPR|PeerRead] [--model_path <checkpoint_path>]
```

### Parameter Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | Run mode: `train` or `test` | `train` |
| `--model` | Config directory name (under `configs/`) | — |
| `--data_source` | Dataset: `AAPR` or `PeerRead` | Uses value in config file |
| `--model_path` | Test mode only: model checkpoint path | `checkpoints/<model>/best_model/model_best.pth` |

### Output Directory

After training, model checkpoints and logs are saved to `checkpoints/<model>/`:

```
checkpoints/
├── IGCN-BERT/
│   ├── best_model/
│   │   └── model_best.pth        # Best model (by val accuracy)
│   ├── training_log.csv          # Per-batch log (loss/acc/val_loss/val_acc)
│   └── train.log                 # Text log (from setup_logger)
└── ... (same for other models)
```

---

## Model-by-Model Instructions

### 1. BERT_CLS — Baseline Pretrained Model Classification

**Summary**: Uses SciBERT's <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> embedding through a linear layer for classification. The simplest baseline.

**Run Command**:

```bash
# AAPR dataset
python scripts/run_bert_cls.py --mode train --model bert_cls --data_source AAPR

# PeerRead dataset
python scripts/run_bert_cls.py --mode train --model bert_cls --data_source PeerRead

# Test
python scripts/run_bert_cls.py --mode test --model bert_cls --data_source AAPR
```

**Config File**: `configs/bert_cls/config.json`

```json
{
    "data_source": "AAPR",
    "num_class": 2,
    "max_seq_length": 256,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "dropout": 0.3,
    "hidden_size": 768,
    "model_type": "../data/bert/scibert-scivocab-uncased",
    "model_name": "BERT_CLS",
    "optimizer": "ADAMW",
    "scheduler": "cosine",
    "seed": 42,
    "device": "cuda"
}
```

**Key Parameters**:
- `dropout`: Classification head dropout rate
- `hidden_size`: BERT hidden dimension (SciBERT = 768)
- `model_type`: Pretrained model path (local path)

---

### 2. SciBERT_CLS — SciBERT Fine-tuning

**Summary**: Same structure as BERT_CLS but explicitly labeled as a SciBERT model. Uses SciBERT for standard fine-tuning.

**Run Command**:

```bash
# AAPR
python scripts/run_scibert_cls.py --mode train --model scibert_cls --data_source AAPR

# PeerRead
python scripts/run_scibert_cls.py --mode train --model scibert_cls --data_source PeerRead

# Test
python scripts/run_scibert_cls.py --mode test --model scibert_cls --data_source AAPR
```

**Config File**: `configs/scibert_cls/config.json` (same structure as BERT_CLS)

---

### 3. SciBERT_Concat — Multi-Layer Output Concatenation

**Summary**: Concatenates outputs from all SciBERT transformer layers, then feeds to the classifier to leverage multi-layer semantic information.

**Run Command**:

```bash
python scripts/run_scibert_concat.py --mode train --model scibert_concat --data_source AAPR
python scripts/run_scibert_concat.py --mode test --model scibert_concat --data_source AAPR
```

**Config File**: `configs/scibert_concat/config.json`

```json
{
    "data_source": "AAPR",
    "num_class": 2,
    "max_seq_length": 256,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "dropout": 0.3,
    "hidden_size": 768,
    "num_layers": 13,
    "model_type": "allenai/scibert-scivocab-uncased",
    "model_name": "SciBERT_Concat",
    "optimizer": "ADAMW",
    "scheduler": "cosine",
    "seed": 42,
    "device": "cuda"
}
```

**Key Parameter**:
- `num_layers`: Number of layers to concatenate (SciBERT has 13: 12 transformer + 1 embedding layer)

---

### 4. SciBERT_Max — Multi-Layer Output Max-Pooling

**Summary**: Applies cross-layer max-pooling on token representations from all SciBERT layers, then classifies.

**Run Command**:

```bash
python scripts/run_scibert_max.py --mode train --model scibert_max --data_source AAPR
python scripts/run_scibert_max.py --mode test --model scibert_max --data_source AAPR
```

**Config File**: `configs/scibert_max/config.json` (same structure as SciBERT_Concat)

---

### 5. SciBERT_Gate — Multi-Layer Gated Fusion

**Summary**: Uses a learnable gating mechanism to dynamically fuse outputs from all SciBERT layers, learning weights for different layers.

**Run Command**:

```bash
python scripts/run_scibert_gate.py --mode train --model scibert_gate --data_source AAPR
python scripts/run_scibert_gate.py --mode test --model scibert_gate --data_source AAPR
```

**Config File**: `configs/scibert_gate/config.json` (same structure as SciBERT_Concat)

---

### 6. Transformer — Pure Transformer Encoder

**Summary**: Trains a Transformer encoder from scratch (no pretraining). Uses BERT tokenizer for tokenization.

**Run Command**:

```bash
python scripts/run_transformer.py --mode train --model transformer --data_source AAPR
python scripts/run_transformer.py --mode test --model transformer --data_source AAPR
```

**Config File**: `configs/transformer/config.json`

```json
{
    "data_source": "AAPR",
    "num_class": 2,
    "max_seq_length": 256,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "dropout": 0.5,
    "dim_model": 768,
    "num_head": 12,
    "hidden": 3072,
    "num_encoder": 6,
    "model_type": "bert-base-uncased",
    "model_name": "Transformer",
    "optimizer": "ADAMW",
    "scheduler": "cosine",
    "seed": 42,
    "device": "cuda"
}
```

**Key Parameters**:
- `dim_model`: Model dimension d_model
- `num_head`: Number of attention heads
- `hidden`: FFN hidden layer dimension
- `num_encoder`: Number of Transformer encoder layers

---

### 7. BERT_GCN — BERT + Graph Convolutional Network

**Summary**: First uses BERT to extract text features, then builds a document-word bipartite graph, and applies GCN for classification on the graph.

**Run Command**:

```bash
python scripts/run_bert_gcn.py --mode train --model bert_gcn --data_source AAPR
python scripts/run_bert_gcn.py --mode test --model bert_gcn --data_source AAPR
```

**Config File**: `configs/bert_gcn/config.json`

```json
{
    "dataset": "mr",
    "nfeat": 768,
    "nhid": 200,
    "nclass": 2,
    "dropout": 0.5,
    "batch_size": 2,
    "learning_rate": 2e-5,
    "epochs": 30,
    "warmup_ratio": 0.1,
    "model_type": "./data/bert/scibert-scivocab-uncased",
    "model_name": "BERT_GCN",
    "optimizer": "ADAMW",
    "scheduler": "cosine",
    "seed": 42,
    "device": "cuda",
    "bert_freeze": false
}
```

**Key Parameters**:
- `nfeat`: Node feature dimension (BERT output = 768)
- `nhid`: GCN hidden layer dimension
- `bert_freeze`: Whether to freeze BERT parameters (false = end-to-end fine-tuning)

---

### 8. BERT_MHAN — Multi-Hop Attention Network

**Summary**: Uses Multi-Hop Attention Network for multi-step reasoning at both sentence-level and document-level.

**Run Command**:

```bash
python scripts/run_bert_mhan.py --mode train --model bert_mhan --data_source AAPR
python scripts/run_bert_mhan.py --mode test --model bert_mhan --data_source AAPR
```

**Config File**: `configs/bert_mhan/config.json`

```json
{
    "data_source": "AAPR",
    "num_class": 2,
    "max_seq_length": 256,
    "max_sentences": 50,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "dropout": 0.3,
    "hidden_size": 768,
    "model_type": "./data/bert/bert-base-uncased",
    "model_name": "BERT_MHAN",
    "optimizer": "ADAMW",
    "scheduler": "cosine",
    "seed": 42,
    "device": "cuda",
    "bert_freeze": false
}
```

**Key Parameters**:
- `max_sentences`: Maximum number of sentences per document
- `bert_freeze`: Whether to freeze BERT parameters

---

### 9. TextGCN — Text Graph Convolutional Network

**Summary**: Builds a heterogeneous word-document graph (document nodes + word nodes with PMI edge weights), and uses a 2-layer GCN for classification. **No pretrained model**.

**Run Command**:

```bash
python scripts/run_textgcn.py --mode train --model textgcn
python scripts/run_textgcn.py --mode test --model textgcn
```

**Config File**: `configs/textgcn/config.json`

```json
{
    "dataset": "mr",
    "nhid": 200,
    "max_epoch": 200,
    "dropout": 0.5,
    "val_ratio": 0.1,
    "early_stopping": 10,
    "lr": 0.02,
    "seed": 42,
    "graph_path": "../data/graph",
    "text_dataset_path": "../data/text_dataset"
}
```

**Key Parameters**:
- `dataset`: Graph dataset name (mr/R52/R8/ohsumed), must match files under `data/text_dataset/`
- `nhid`: GCN hidden layer dimension
- `early_stopping`: Early-stopping patience
- `graph_path` / `text_dataset_path`: Paths to graph data and text data

**Note**: TextGCN uses an independent data pipeline and requires files under `data/graph/` and `data/text_dataset/`. It **does NOT** use the `.list` files of AAPR/PeerRead.

---

### 10. DGCBERT — Dynamic Graph Conv + BERT

**Summary**: Dynamically builds document graphs, uses Top-K attention to select key nodes, applies APPNP convolution on the graph, and combines with BERT representations.

**Prerequisite**: Ensure `torch_geometric` is installed:

```bash
pip install torch_geometric
```

**Run Command**:

```bash
python scripts/run_dgc_bert.py --mode train --model DGCBERT --data_source AAPR
python scripts/run_dgc_bert.py --mode test --model DGCBERT --data_source AAPR
```

**Config File**: `configs/DGCBERT/config.json`

```json
{
    "data_source": "AAPR",
    "num_class": 2,
    "max_seq_length": 256,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "keep_prob": 0.3,
    "hidden_size": 768,
    "predict_dim": 128,
    "k": 4,
    "alpha": 0.5,
    "top_rate": 0.3,
    "model_type": "./data/bert/scibert-scivocab-uncased",
    "model_name": "DGCBERT",
    "mode": "top_biaffine+softmax",
    "optimizer": "ADAMW",
    "scheduler": "cosine",
    "seed": 123
}
```

**Key Parameters**:
- `predict_dim`: Feature dimension after graph convolution
- `k` / `alpha`: APPNP propagation steps (K) and teleport probability
- `top_rate`: Top-K attention node selection ratio
- `mode`: Fusion mode: `top_biaffine+softmax` / `top_normal`

---

### 11. IGCN-BERT — Interactive GCN + BERT (Proposed Model)

**Summary**: **Proposed in this work**. Extracts multi-layer lexical/semantic representations and attention matrices from SciBERT, builds multi-channel adjacency matrices, and applies stacked **IGCN (Interactive GCN)** layers for joint dynamic update of node embeddings and adjacency matrices.

**Core Modules**:
- **WANU (Weight-Aware Node Update)**: Aggregates neighbor information using the current adjacency matrix to update node embeddings
- **NAWU (Node-Aware Adjacency Weight Update)**: Dynamically updates adjacency matrix weights using updated node embeddings
- **IGCN Layer**: Alternating WANU + NAWU execution, stacked with multiple layers

**Run Command**:

```bash
# Train on AAPR
python scripts/run_igcn_bert.py --mode train --model IGCN-BERT --data_source AAPR

# Train on PeerRead
python scripts/run_igcn_bert.py --mode train --model IGCN-BERT --data_source PeerRead

# Test on AAPR
python scripts/run_igcn_bert.py --mode test --model IGCN-BERT --data_source AAPR

# Test on PeerRead
python scripts/run_igcn_bert.py --mode test --model IGCN-BERT --data_source PeerRead
```

**Config File**: `configs/IGCN-BERT/config.json`

```json
{
    "data_source": "AAPR",
    "num_class": 2,
    "max_seq_length": 256,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "keep_prob": 0.3,
    "hidden_size": 768,
    "l_layers": 3,
    "d_A": 32,
    "M_igcn": 3,
    "model_type": "./data/bert/scibert-scivocab-uncased",
    "model_name": "IGCNBERT",
    "optimizer": "ADAMW",
    "scheduler": "cosine",
    "seed": 123
}
```

**Key Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `l_layers` | Number of layers selected from SciBERT (first l layers = lexical, last l layers = semantic) | 3 |
| `d_A` | Number of channels in multi-channel adjacency matrix | 32 |
| `M_igcn` | Number of stacked IGCN layers | 3 |
| `keep_prob` | Dropout rate | 0.3 |
| `hidden_size` | SciBERT hidden layer dimension | 768 |
| `batch_size` | Recommended ≤ 8 (IGCN processes seq dimension in a loop, memory/speed sensitive) | 8 |

**GPU Memory Reference**:
- 256 sequence length + batch_size=8 + M=3: approximately 10-12 GB
- If OOM occurs, reduce `batch_size` or `max_seq_length`

**Training Speed Reference**:
- AAPR dataset (26,980 samples) + RTX 3090: approx 30-40 min/epoch
- PeerRead dataset (9,377 samples): approx 10-15 min/epoch

---

## FAQ / Troubleshooting

### Q1: `ModuleNotFoundError: No module named 'transformers.models.bert.tokenization_bert_fast'`

**Cause**: Data cache files were generated by an older version of transformers and are incompatible with the current version.

**Fix**: Clean cache and re-run:

```bash
del data\AAPR\cache\*.pth
del data\AAPR\vocab.pth
```

### Q2: `ModuleNotFoundError: No module named 'torch_geometric'`

**Cause**: torch_geometric dependency missing (only affects DGCBERT).

**Fix**:

```bash
pip install torch_geometric
```

### Q3: CUDA out of memory

**Cause**: batch_size too large or sequence length too long.

**Fix**: Modify `configs/<model>/config.json`:

```json
{
    "batch_size": 4,          // Reduce from 32 to 4
    "max_seq_length": 128     // Reduce from 256 to 128
}
```

### Q4: First run data preprocessing is slow

Normal. On first run, the code will:
1. Iterate all documents to build the vocabulary (~10-30 seconds)
2. Tokenize all text and build PyTorch Datasets (~1-2 minutes)

After processing, cache files are generated; subsequent runs load from cache and are fast.

### Q5: How to switch to the PeerRead dataset

Simply specify via `--data_source PeerRead`:

```bash
python scripts/run_igcn_bert.py --mode train --model IGCN-BERT --data_source PeerRead
```

Alternatively, modify the `data_source` field in the config file directly.

### Q6: Where are the training logs

Training logs for each model are under `checkpoints/<model>/`:
- `train.log`: Text-format log
- `training_log.csv`: CSV format — can be imported into Excel or pandas for analysis

### Q7: How to reproduce experimental results

All models use a fixed `seed`. Results should be reproducible within the same environment. For exact reproducibility, ensure:
- Same Python / PyTorch / transformers versions
- Same hardware (GPU model)
- Identical data file contents

---

## Results Log

### AAPR Dataset

| Model | Test Accuracy | Training Time | Notes |
|-------|---------------|---------------|-------|
| BERT_CLS | — | ~15 min/epoch | Basic baseline |
| SciBERT_CLS | — | ~15 min/epoch | SciBERT baseline |
| SciBERT_Concat | — | ~20 min/epoch | Multi-layer concat |
| SciBERT_Max | — | ~20 min/epoch | Multi-layer max-pool |
| SciBERT_Gate | — | ~20 min/epoch | Gated fusion |
| Transformer | — | — | Trained from scratch |
| BERT_GCN | — | — | BERT + GCN |
| BERT_MHAN | — | — | Multi-hop attention |
| TextGCN | — | — | Graph-only method |
| DGCBERT | — | ~25 min/epoch | Dynamic graph conv |
| **IGCN-BERT** | — | ~35 min/epoch | **This work** |

### PeerRead Dataset

| Model | Test Accuracy | Training Time | Notes |
|-------|---------------|---------------|-------|
| BERT_CLS | — | ~5 min/epoch | |
| SciBERT_CLS | — | ~5 min/epoch | |
| ... | ... | ... | ... |
| **IGCN-BERT** | — | ~12 min/epoch | **This work** |

> **Note**: Fill in the table after actual runs. Training curves can be viewed via `checkpoints/<model>/training_log.csv`.

---

## Project Directory Structure Quick Reference

```
IGCN-BERT/
├── scripts/                      # Per-model run scripts
│   ├── run_bert_cls.py
│   ├── run_scibert_cls.py
│   ├── run_scibert_concat.py
│   ├── run_scibert_max.py
│   ├── run_scibert_gate.py
│   ├── run_transformer.py
│   ├── run_bert_gcn.py
│   ├── run_bert_mhan.py
│   ├── run_textgcn.py
│   ├── run_dgc_bert.py
│   └── run_igcn_bert.py          # IGCN-BERT
├── models/                       # Model implementations
│   ├── bert_cls/
│   ├── scibert_cls/
│   ├── scibert_concat/
│   ├── scibert_max/
│   ├── scibert_gate/
│   ├── transformer/
│   ├── bert_gcn/
│   ├── bert_mhan/
│   ├── textgcn/
│   ├── dgc_bert/
│   └── igcn_bert/                # IGCN-BERT
├── configs/                      # Model configurations
│   ├── bert_cls/config.json
│   ├── ...
│   ├── DGCBERT/config.json
│   └── IGCN-BERT/config.json     # IGCN-BERT
├── dataloader/                   # Data loading
│   ├── dataloader.py             # Universal DataLoader
│   └── dataset.py                # Dataset definition
├── train/                        # Trainer
│   └── trainer.py                # Universal Trainer (BERT-style + GCN-style)
├── utils/                        # Utilities
│   └── utils.py                  # setup_seed, setup_logger
├── data/                         # Datasets & pretrained models
│   ├── AAPR/
│   ├── PeerRead/
│   ├── graph/                    # TextGCN graph data
│   ├── text_dataset/             # TextGCN text data
│   └── bert/
│       ├── scibert-scivocab-uncased/
│       └── bert-base-uncased/
├── checkpoints/                  # Training output (generated at runtime)
│   ├── IGCN-BERT/
│   ├── DGCBERT/
│   └── ...
└── references/                   # Reference implementations (dev-only)
```

---

## Quick Start: One-Shot IGCN-BERT

```bash
# 1. Activate environment
conda activate openltm

# 2. Clean old cache (required on first run or env change)
del data\AAPR\cache\*.pth
del data\AAPR\vocab.pth

# 3. Train (AAPR dataset)
python scripts\run_igcn_bert.py --mode train --model IGCN-BERT --data_source AAPR

# 4. Test
python scripts\run_igcn_bert.py --mode test --model IGCN-BERT --data_source AAPR

# 5. Review training log
# checkpoints\IGCN-BERT\training_log.csv
```

**Expected Time** (single GPU):
- AAPR 5 epochs: approx 2.5-3 hours
- PeerRead 5 epochs: approx 1 hour

Good luck with your experiments!
