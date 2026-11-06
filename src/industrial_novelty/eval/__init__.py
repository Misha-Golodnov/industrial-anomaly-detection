"""Общая оценка моделей на одном пороге и одних окнах."""

from industrial_novelty.eval.calibration import run_maxima, threshold_at_run_risk
from industrial_novelty.eval.detection import (
    DetectionReport,
    evaluate_model,
    threshold_at_fpr,
)

__all__ = [
    "DetectionReport",
    "evaluate_model",
    "run_maxima",
    "threshold_at_fpr",
    "threshold_at_run_risk",
]
