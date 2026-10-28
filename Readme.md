# Physics-Constrained LEO Orbital Conjunction Risk Model (SciML)

[![CI Pipeline](https://github.com/human1sh/orbital-risk-pinn/actions/workflows/ci.yml/badge.svg)](https://github.com/human1sh/orbital-risk-pinn/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Physics-Informed Neural Network (PINN) designed to predict satellite collision avoidance maneuver execution ($\text{P}_{\text{risk}}$) across Low Earth Orbit (LEO) conjunction telemetry events while enforcing physical monotonicity constraints via PyTorch autograd.

---

## Key Methodology

- **Autograd Monotonicity Constraint**: Enforces a soft penalty ($\frac{\partial P_{\text{risk}}}{\partial r_{\text{miss}}} \le 0$) preventing unphysical risk increases as miss distance widens.
- **Post-Hoc Probability Calibration**: Calibrates logits via Temperature Scaling ($T = 1.4861$) optimized with L-BFGS to achieve an Expected Calibration Error (ECE) of **0.0366**.
- **Domain Feature Engineering**: Incorporates time-decay risk indices, space-weather solar-flux drag proxies, and Yeo-Johnson transformed logarithmic geometries.

---

## Performance & Diagnostic Suite

| Metric | Uncalibrated Model | Calibrated Residual PINN |
| :--- | :---: | :---: |
| **ROC-AUC** | 0.9321 | **0.9321** |
| **Macro F1-Score** | 0.9428 | **0.9428** |
| **Balanced Accuracy** | 0.9161 | **0.9161** |
| **Brier Score Loss** | — | **0.0338** |
| **Expected Calibration Error (ECE)** | 0.0164 | **0.0366** |

![Model Performance Diagnostics](outputs/figures/output.png)

---

## Quickstart & Local Setup

### 1. Installation
```bash
git clone [https://github.com/human1sh/orbital-risk-pinn.git](https://github.com/human1sh/orbital-risk-pinn.git)
cd orbital-risk-pinn
pip install -e .[dev]
```

### 2. Run Test Suite
```bash
pytest tests/ --cov=src/orbital_risk
```

### 3. Train Model via CLI
```bash
python -m orbital_risk.train --config configs/config.yaml
```

---

## Author
**Himanshu Abhay Bhisikar** — Machine Learning Scientist / Astrophysicist  
Website: [human1sh.github.io](https://human1sh.github.io) | Email: himanshubhisikar@gmail.com