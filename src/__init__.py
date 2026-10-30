"""
Orbital Risk PINN Package.

Physics-Constrained Neural Network pipeline for Low Earth Orbit (LEO)
satellite conjunction risk assessment and probability calibration.
"""

from src.data import ConjunctionDataset, prepare_data_pipeline
from src.loss import MonotonicityPhysicsLoss
from src.models import AstrodynamicPINN, TemperatureScaler
from src.plots import generate_all_plots
from src.utils import calculate_ece, evaluate_predictions

__version__ = "0.1.0"

__all__ = [
    "ConjunctionDataset",
    "prepare_data_pipeline",
    "MonotonicityPhysicsLoss",
    "AstrodynamicPINN",
    "TemperatureScaler",
    "generate_all_plots",
    "calculate_ece",
    "evaluate_predictions",
]