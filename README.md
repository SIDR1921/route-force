<div align="center">

# 🛡️ A.E.G.I.S.

### **A**utonomous **E**go-centric **G**raph **I**ntelligence **S**ystem

*Multi-modal pedestrian trajectory prediction via SE(2)-equivariant graph neural networks & continuous Bézier parameterization*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyG-2.4+-3C2179?style=for-the-badge&logo=pyg&logoColor=white)](https://pyg.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 🌟 Project Overview

**A.E.G.I.S.** predicts where pedestrians and vehicles will go next by modeling social interactions as equivariant graphs and outputting smooth, physically plausible trajectories. 

The primary objectives achieved by this system are:
1. **Processing temporal sequence data**: Ingesting historical `(x, y)` coordinate sequences and converting them to latent embeddings representing agent kinematics.
2. **Account for "Social Context"**: Transforming scenes into graphs where pedestrians avoid each other using ego-centric proximity (10-meter radius edges).
3. **Multi-modal prediction**: Outputting the 3 most likely future paths per-agent, using a mathematically continuous parameterization (Bézier curves). 

Unlike standard neural networks that suffer from "mode collapse" (averaging all futures into a single line) and memorized orientations, A.E.G.I.S. integrates **Minimum-of-K Loss** to foster multi-modal diversity and **E(2)-Equivariant Message Passing** to ensure predictions are robust regardless of the direction the agents are facing.

---

## 🏗️ Model Architecture

The model processes inputs through highly specific modules to ensure spatial invariance and continuous geometry.

```
┌─────────────────────────────────────────────────────────────────┐
│                     A.E.G.I.S. Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ nuScenes │───▶│  Ego-Centric │───▶│ Social Graph │          │
│  │ Raw Data │    │ Normalization│    │ Construction │          │
│  └──────────┘    └──────────────┘    └──────┬───────┘          │
│                                              │                  │
│              ┌───────────────────────────────┘                  │
│              ▼                                                  │
│  ┌───────────────────┐    ┌───────────────────┐                │
│  │   EGNN Encoder    │───▶│  Bézier Decoder   │                │
│  │  3× E(2) Layers   │    │   3 Modal Paths   │                │
│  │  SiLU + LayerNorm │    │  4 Control Points │                │
│  │  [N, 10] → [N,128]│    │  [N,128] → [N,3,  │                │
│  └───────────────────┘    │     12, 2]        │                │
│                           └─────────┬─────────┘                │
│                                     │                           │
│              ┌──────────────────────┘                           │
│              ▼                                                  │
│  ┌───────────────────┐    ┌───────────────────┐                │
│  │ Min-of-K Huber    │    │  TTA Inference    │                │
│  │   Winner-Takes-   │    │  4× Rotation      │                │
│  │   All Loss        │    │  Ensemble          │                │
│  └───────────────────┘    └───────────────────┘                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Core Architecture Components
1. **Ego-Centric Normalization** strips absolute map coordinates so the model learns relative kinematics (the agent's current location becomes origin `[0,0]`).
2. **SE(2) Equivariant Graph Neural Network (Encoder)** mathematically guarantees that rotated inputs produce rotated outputs, drastically reducing the dataset size needed for generalization.
3. **Continuous Bézier Trajectories (Decoder)** predicts 4 cubic control points per mode instead of predicting individual time steps. This algebraic approach guarantees smooth predictions and parameter-efficiency.
4. **Minimum-of-K Loss** ensures true topological diversity. The model evaluates Huber loss (Smooth L1) across 3 modes but only backpropagates the penalty on the *best* performing mode, freeing the others to explore different directions without punishment.
5. **Test-Time Augmentation (TTA)** leverages a final spatial ensemble covering `0°/90°/180°/270°` rotations at inference to squeeze out extra accuracy without added parameter count.

---

## 📂 Dataset Used

The model expects the standard **nuScenes prediction mini dataset (`v1.0-mini`)**. 
The A.E.G.I.S. custom data extractor (`aegis/data/pipeline.py`) performs the following native extractions:

- **Parsing Annotations**: Traverses JSON metadata linked-lists to reconstruct temporally contiguous `x, y` sequences per agent.
- **Categorical Filtering**: Tracks 7 core human and vehicle actor classes (e.g., `human.pedestrian.adult`, `vehicle.car`).
- **History/Future Windows**: At `2Hz`, the model looks `2` seconds backward (4 steps + current) and predicts `6` seconds forward (12 steps).
- **Train/Val Splits**: The 10 scenes in `v1.0-mini` are partitioned cleanly with scenes 0-7 comprising the training split (196 snapshots), and scenes 8-9 forming the validation split (48 snapshots).

---

## ⚙️ Setup & Installation Instructions

This codebase was specifically designed to run out-of-the-box on CPU without requiring finicky C++ graph compilation. 

### 1. Clone & Environment Setup
```bash
git clone https://github.com/SIDR1921/route-force.git
cd route-force

python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
# CPU-only PyTorch setup (recommended for fast installation)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Essential dependencies (PyG fallback avoids torch-cluster compilation issues)
pip install torch-geometric pyyaml tqdm matplotlib
```

### 3. Prepare the nuScenes Dataset
Place your nuScenes mini tarball within the repository, and unpack it so the metadata JSONs reside correctly in `data/v1.0-mini`:
```bash
mkdir -p data
tar -xf v1.0-mini.tar -C data --include='v1.0-mini/*'
```
*(File structure should look like `route-force/data/v1.0-mini/sample_annotation.json`)*

---

## 🚀 How to Run the Code

Configuration parameters (model dimensions, batch size, sliding windows) are centralized in `configs/default.yaml`.

### Training the Network 
To initiate the training pipeline with AdamW and the bespoke `OneCycleLR` learning rate scheduler:
```bash
python -m aegis.train --config configs/default.yaml
```

Checkpoints will be automatically streamed directly to `./checkpoints/`. The training engine manages tracking the Best Val ADE checkpoint. 

### Model Inference (TTA Output)
To execute trajectory evaluations on the validation split leveraging Test-Time Augmentation (TTA) ensembles:
```bash
python -m aegis.inference --config configs/default.yaml --checkpoint checkpoints/best.pt
```

---

## 📈 Example Outputs / Results

Training convergence occurs around 50~80 Epochs. During Hackathon execution across the `v1.0-mini` snapshot set, the custom architecture immediately stabilized its Best-of-K multi-modal choices. 

**Hackathon Baseline Profile (Validation Segment):**
| Metric | Score | Note |
|:---|:---:|:---|
| **Best-of-3 ADE** *(Avg Displacement)* | **2.51 m** | Captures sequential coordinate offset |
| **Best-of-3 FDE** *(Final Displacement)* | **5.28 m** | End-destination accuracy |
| GPU Requirements | None | Full CPU portability achieved |

**Example Training Log Output:**
```
============================================================
  A.E.G.I.S. — Training Pipeline
  Device: cpu
============================================================

📊 Building data loaders...
   Train samples: 196
   Val samples:   48
🧠 Model: 346,333 parameters

🚀 Starting training for 80 epochs...

Ep   1 | T.ADE 4.590 T.FDE 8.335 | V.ADE 4.688 V.FDE 8.707 | LR 0.000044 🏆
...
Ep  16 | T.ADE 2.442 T.FDE 4.557 | V.ADE 2.886 V.FDE 5.716 | LR 0.000763 🏆
...
Ep  38 | T.ADE 2.169 T.FDE 4.535 | V.ADE 2.510 V.FDE 5.278 | LR 0.000852 🏆

============================================================
  Training complete!
  Best Val ADE: 2.5095
  Checkpoints saved to: checkpoints
============================================================
```

> **A.E.G.I.S.** natively addresses the requested core objectives—modeling social awareness by encoding radius proximity edges inside the `PyG` data pipeline, processing kinematic histories securely with rotational equivariance, and outputting explicit multi-modal predictions driven by a continuous mathematics backend. 

---

## 📄 License
MIT License — see [LICENSE](LICENSE) for details.
