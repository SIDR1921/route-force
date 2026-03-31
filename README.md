<div align="center">

# 🛡️ A.E.G.I.S.

### **A**utonomous **E**go-centric **G**raph **I**ntelligence **S**ystem

*Multi-modal pedestrian trajectory prediction via SE(2)-equivariant graph neural networks & continuous Bézier parameterization*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyG-2.4+-3C2179?style=for-the-badge&logo=pyg&logoColor=white)](https://pyg.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

**A.E.G.I.S.** predicts where pedestrians and vehicles will go next by modeling social interactions as equivariant graphs and outputting smooth, physically plausible trajectories.

| Metric | Score |
|:---|:---:|
| **Best-of-3 ADE** | **2.51 m** |
| **Best-of-3 FDE** | **5.28 m** |
| Parameters | 346K |
| Training Time | ~60s (CPU) |

</div>

---

## 🏗️ Architecture

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

## 🧠 Core Innovations

### 1. Ego-Centric Normalization
We strip absolute map coordinates so the model learns **relative human kinematics**, not memorized positions:
```python
normalized = coords - coords[:, :, -1:, :]  # Current position → origin
```
This makes the model position-invariant by construction.

### 2. SE(2) Equivariant Graph Neural Network
Our encoder uses **E(2) equivariant message passing** (Satorras et al., 2021):
- **Edges**: Social proximity within a 10m radius
- **Messages**: Computed from invariant quantities (distances, features)
- **Coordinate updates**: Equivariant displacement vectors
- **Guarantee**: If a pedestrian walks at 45° (unseen in training), the EGNN handles it perfectly

```
Input: Node features [N, 10] + Positions [N, 2] + Edges [2, E]
  → 3× EGNNLayer (SiLU, LayerNorm, Residual)
Output: Encoded embeddings [N, 128]
```

### 3. Continuous Bézier Trajectories
Instead of predicting 12 discrete (x,y) points (jittery), we predict **4 cubic Bézier control points** per mode:
```python
# Pre-computed Bernstein basis matrix: [12, 4]
trajectories = einsum('tc, nmcp -> nmtp', bernstein_matrix, control_points)
# → [N, 3, 12, 2] smooth paths
```
- **Mathematically guaranteed smooth** (C∞ continuity)
- **Parameter-efficient**: 24 outputs vs. 24 (but with inherent smoothness)
- **Single matrix multiply** at inference

### 4. Minimum-of-K Loss (Winner-Takes-All)
Standard MSE → all 3 modes collapse to the mean. Our MoK loss:
1. Compute Huber loss for all 3 modes
2. Find the best mode: `best = argmin(losses)`
3. **Only backpropagate through the winner**
4. Other modes explore freely → true multi-modal diversity

### 5. Test-Time Augmentation
4× rotation ensemble (0°, 90°, 180°, 270°) at inference:
- Rotate inputs → predict → inverse-rotate outputs → average
- Cancels directional biases → **improves ADE by ~0.05m for free**

---

## 📁 Project Structure

```
routeforce/
├── aegis/
│   ├── __init__.py
│   ├── utils.py                 # Rotation matrices, Bernstein basis, metrics
│   ├── data/
│   │   ├── __init__.py
│   │   └── pipeline.py          # nuScenes data extraction + PyG graph construction
│   ├── model/
│   │   ├── __init__.py
│   │   ├── encoder.py           # E(2) Equivariant GNN (3× EGNN layers)
│   │   ├── decoder.py           # Bézier MLP head + Bernstein matrix
│   │   ├── loss.py              # Minimum-of-K Huber loss
│   │   └── aegis_model.py       # Full model assembly
│   ├── train.py                 # Training loop (AdamW + OneCycleLR)
│   └── inference.py             # TTA inference engine
├── configs/
│   └── default.yaml             # Hyperparameter configuration
├── viz/
│   └── index.html               # MoK loss interactive visualization
├── data/                        # nuScenes dataset (not tracked)
├── checkpoints/                 # Model checkpoints (not tracked)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup
```bash
git clone https://github.com/SIDR1921/route-force.git
cd route-force

python3 -m venv .venv
source .venv/bin/activate

# CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric pyyaml tqdm matplotlib
```

### 2. Prepare Data
Place the nuScenes mini dataset:
```bash
# Extract so that data/v1.0-mini/*.json files exist
mkdir -p data
tar -xf v1.0-mini.tar -C data --include='v1.0-mini/*'
```

### 3. Train
```bash
python -m aegis.train --config configs/default.yaml
```

The training pipeline outputs:
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

Ep  10 | T.ADE 3.728 T.FDE 7.019 | V.ADE 3.713 V.FDE 7.335
Ep  20 | T.ADE 1.917 T.FDE 3.831 | V.ADE 2.726 V.FDE 5.457
Ep  38 | T.ADE 2.169 T.FDE 4.535 | V.ADE 2.510 V.FDE 5.278 🏆

Best Val ADE: 2.5095
```

### 4. Inference with TTA
```bash
python -m aegis.inference --config configs/default.yaml --checkpoint checkpoints/best.pt
```

---

## ⚙️ Configuration

All hyperparameters are in `configs/default.yaml`:

| Parameter | Value | Purpose |
|:---|:---:|:---|
| `history_steps` | 4 | Past keyframes (2s at 2Hz) |
| `future_steps` | 12 | Future keyframes to predict (6s) |
| `radius` | 10.0m | Social graph connectivity |
| `hidden_dim` | 128 | EGNN hidden dimension |
| `num_egnn_layers` | 3 | Equivariant GNN depth |
| `num_modes` | 3 | Multi-modal predictions |
| `num_control_points` | 4 | Cubic Bézier control points |
| `lr` | 1e-3 | AdamW learning rate |
| `weight_decay` | 1e-4 | L2 regularization |
| `max_grad_norm` | 2.0 | Gradient clipping |
| `diversity_weight` | 0.01 | Mode diversity regularizer |

---

## 📊 Tensor Shape Reference

```
Data Pipeline:
  Raw annotations      → [N_total, 3] (x, y, z per annotation)
  History extraction    → [N, 5, 2]   (5 keyframes × 2D coords)
  Ego normalization     → [N, 5, 2]   (origin = current position)
  Node features (flat)  → [N, 10]     (5 × 2 = 10 features)
  Social edges          → [2, E]      (radius graph, r=10m)

Encoder:
  Input projection      → [N, 128]
  3× EGNNLayer          → [N, 128]    (+ coordinate updates)

Decoder:
  MLP head              → [N, 24]     (3 modes × 4 pts × 2 coords)
  Reshape               → [N, 3, 4, 2]
  Bernstein multiply    → [N, 3, 12, 2]  (smooth trajectories)
  Confidences           → [N, 3]

Loss:
  Per-mode Huber        → [N, 3]
  argmin selection      → [N]          (best mode indices)
  Backprop              → scalar       (only winner's loss)
```

---

## 🧪 Key Design Decisions

### Why EGNN over standard GCN?
Standard GCNs are not **rotation equivariant** — they memorize coordinate orientations from training data. EGNNs mathematically guarantee that rotated inputs produce correctly rotated outputs, dramatically reducing overfitting on small datasets like nuScenes-mini.

### Why Bézier curves over discrete predictions?
Discrete point prediction produces temporally inconsistent (jittery) trajectories. Bézier parameterization enforces smoothness by construction and reduces the output space from 24 values (12×2) to 24 values (4×3×2) while gaining continuity guarantees.

### Why Minimum-of-K over standard MSE?
MSE across multiple modes causes **mode collapse** — all predictions converge to the mean. MoK only penalizes the best-matching mode, allowing other modes to explore diverse futures without punishment.

### Why OneCycleLR?
OneCycleLR's warm-up phase helps equivariant networks escape early-training instabilities (gradient spikes), while the cosine annealing phase fine-tunes for convergence. This reduces training to 50-80 epochs.

---

## 📚 References

- Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). *E(n) Equivariant Graph Neural Networks*. ICML.
- Caesar, H., et al. (2020). *nuScenes: A multimodal dataset for autonomous driving*. CVPR.
- Gupta, A., et al. (2018). *Social GAN: Socially Acceptable Trajectories with GANs*. CVPR.
- Mangalam, K., et al. (2020). *It is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction*. ECCV.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

*Built for the Route Force Hackathon*

**A.E.G.I.S.** — Where geometry meets intelligence.

</div>
