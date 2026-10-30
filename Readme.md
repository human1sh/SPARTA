# Physics-Constrained Orbital Conjunction Risk Model (SciML)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade Scientific Machine Learning (SciML) pipeline designed to evaluate satellite collision avoidance maneuver risks across LEO (Low Earth Orbit) conjunction events. The system combines high-capacity **Residual Neural Networks**, **autograd physics gradient constraints** ($\partial P_{\text{risk}} / \partial r_{\text{miss}} \le 0$), and **L-BFGS Temperature Scaling** for probability calibration.

---

## 📌 Project Overview

Operational satellite collision avoidance relies on estimating maneuver necessity from noisy conjunction assessment data. Standard deep learning classifiers often suffer from two failure modes:

1. **Unphysical Predictions:** Predicting *higher* risk as miss distance widens.
2. **Overconfident Logits:** Uncalibrated probabilities leading to poor risk-quantification decisions.

`orbital-risk-pinn` solves both issues by enforcing a soft physics penalty ($\frac{\partial P_{\text{risk}}}{\partial r_{\text{miss}}} \le 0$) via PyTorch autograd during training and applying **L-BFGS Temperature Scaling** to optimize probability outputs for operational decision boundaries.

---

## 📁 Repository Structure

```text
orbital-risk-pinn/
├── configs/
│   └── config.yaml          # Pipeline hyperparameter and path configurations
├── dat/
│   └── telemetry.csv        # Telemetry dataset
├── outputs/
│   ├── 01_calibration_and_roc.png
│   ├── 02_physics_monotonicity.png
│   ├── 03_loss_trajectories.png
│   ├── 04_drag_risk_heatmap.png
│   ├── 05_risk_probability_distribution.png
│   ├── pinn_model.pt        # Saved PyTorch model checkpoint
│   └── predictions.csv      # Output inference predictions
├── src/
│   ├── data.py              # Data ingestion, engineering, and PyTorch DataLoaders
│   ├── loss.py              # Autograd soft physics monotonicity penalty
│   ├── models.py            # Residual PINN architecture & TemperatureScaler
│   ├── plots.py             # Diagnostic and physics verification figure generation
│   ├── predict.py           # Batch inference script
│   ├── train.py             # Dedicated model training CLI
│   └── utils.py             # Evaluation metrics and calibration utilities
├── tests/
│   └── test_pipeline.py     # Automated pytest test suite
├── main.py                  # Root unified entry point script
├── pyproject.toml           # Package configuration & pytest settings
└── README.md                # Technical project documentation
```

---

## ⚙️ Mathematical Methodology

### 1. Physics Monotonicity Constraint
Physical collision risk must non-increase as the total miss distance ($r_{\text{miss}}$) between objects increases. To enforce this physical law during gradient descent, an autograd soft penalty is calculated over each training batch:

$$\mathcal{L}_{\text{physics}} = \lambda \cdot \frac{1}{N} \sum_{i=1}^{N} \max\left(0, \frac{\partial P_{\text{risk}}^{(i)}}{\partial r_{\text{miss}}^{(i)}}\right)$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{physics}}$$

Where $\frac{\partial P_{\text{risk}}}{\partial r_{\text{miss}}}$ is calculated directly via PyTorch autograd graph tracking.

### 2. Post-Hoc Temperature Scaling Calibration
Standard neural network outputs trained with Binary Cross-Entropy often suffer from overconfidence. We optimize a single scalar parameter $T > 0$ on the validation set using **L-BFGS** to calibrate predicted probabilities:

$$\hat{q}_i = \sigma\left(\frac{z_i}{T}\right)$$

---

## ⚡ Quickstart & Installation

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/human1sh/orbital-risk-pinn.git
cd orbital-risk-pinn

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies and editable package
pip install -e .
```

---

## 🚀 Usage

### End-to-End Training & Evaluation Pipeline
Run the full training, calibration, and plotting pipeline:
```bash
python main.py --config configs/config.yaml
```

### Dedicated Training Script
Train the Residual PINN and save model checkpoints:
```bash
python src/train.py --config configs/config.yaml
```

### Production Batch Inference
Run probability predictions on new satellite telemetry data:
```bash
python src/predict.py \
  --input_csv dat/leo_satellite_conjunction_telemetry_enriched_2026.csv \
  --output_csv outputs/predictions.csv \
  --checkpoint outputs/checkpoints/pinn_model.pt
```

---

## 🧪 Testing

Run the automated test suite to verify tensor shapes and autograd physics penalties:
```bash
pytest tests/
```

---

## 📊 Benchmark Diagnostics & Results

| Metric | Benchmark Value |
| :--- | :--- |
| **ROC-AUC Score** | **0.9321** |
| **Macro F1-Score** | **0.9428** |
| **Balanced Accuracy** | **0.9161** |
| **Brier Score Loss** | **0.0338** |
| **Calibrated Temperature ($T$)** | **1.8767** |
| **Expected Calibration Error (ECE)** | **0.037** |

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.