"""Normal-only virtual sensors and multiscale tests for single-duct AHUs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from industrial_novelty.data.lbnl import FEEDBACK
from industrial_novelty.models.subspace import ledoit_wolf

ZONE = tuple(f"ZONE_TEMP_{i}" for i in range(1, 6))
SENSORS = {
    "OA_TEMP": ("MA_TEMP", "RA_TEMP", "OA_DMPR_DM", "SF_CS", *ZONE),
    "MA_TEMP": ("OA_TEMP", "RA_TEMP", "OA_DMPR_DM", "RA_DMPR_DM", "SF_CS"),
    "SA_TEMP": ("MA_TEMP", "OA_TEMP", "CHWC_VLV_DM", "SF_CS", "SA_TEMPSPT"),
    "CHWC_VLV_DM": ("MA_TEMP", "OA_TEMP", "SA_TEMPSPT", "SF_CS", *ZONE),
}
REGRESSION_FLOORS = (0.3, 0.15, 0.15, 0.01)
PHYSICS_FLOORS = (0.15, 0.15, 0.15, 0.15)
SCALES = (1, 2, 4)


def windows(values: np.ndarray, ends: np.ndarray, width: int) -> np.ndarray:
    """Create (window, time, feature) arrays at previously validated endpoints."""
    return values[ends[:, None] + np.arange(1 - width, 1)]


def virtual_regressor(*, min_samples_leaf: int = 35) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=120,
        max_leaf_nodes=23,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=1.0,
        early_stopping=False,
    )


class AHUResiduals:
    def __init__(self, *, feedback: bool = False):
        self.feedback = feedback

    def fit(self, frame: pd.DataFrame) -> AHUResiduals:
        self.virtual = {
            name: virtual_regressor().fit(
                frame.loc[:, inputs].to_numpy(), frame[name].to_numpy()
            )
            for name, inputs in SENSORS.items()
        }
        contrast = frame["OA_TEMP"] - frame["RA_TEMP"]
        valid = contrast.abs() >= 5
        self.mixing = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        self.mixing.fit(
            frame.loc[valid, "OA_DMPR_DM"],
            ((frame["MA_TEMP"] - frame["RA_TEMP"])[valid] / contrast[valid]),
            sample_weight=contrast[valid] ** 2,
        )
        return self

    def regression(self, frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                frame[name].to_numpy()
                - self.virtual[name].predict(frame.loc[:, inputs].to_numpy())
                for name, inputs in SENSORS.items()
            ]
        )

    def physics(self, frame: pd.DataFrame) -> np.ndarray:
        fraction = self.mixing.predict(frame["OA_DMPR_DM"].to_numpy())
        outdoor, ret, mixed, supply, setpoint = (
            frame[column].to_numpy()
            for column in ("OA_TEMP", "RA_TEMP", "MA_TEMP", "SA_TEMP", "SA_TEMPSPT")
        )
        errors = [
            mixed - ret - fraction * (outdoor - ret),
            supply - setpoint,
            np.maximum(np.minimum(outdoor, ret) - mixed, 0)
            + np.maximum(mixed - np.maximum(outdoor, ret), 0),
            np.where(frame["CHWC_VLV_DM"].to_numpy() < 0.01, mixed - supply, 0),
        ]
        if self.feedback:
            errors.extend(
                (frame[name] - frame[f"{name}_DM"]).to_numpy() for name in FEEDBACK
            )
        return np.column_stack(errors)


def regime(frame: pd.DataFrame) -> np.ndarray:
    """Three weather bands from the equipment's documented economizer thresholds."""
    return np.searchsorted([33.8, 60.0], frame["OA_TEMP"].to_numpy())


class ResidualScan:
    """Shrinkage Gaussian scores for signed mean shifts of unknown duration."""

    def __init__(self, floors, *, conditional: bool = True, scales=SCALES):
        self.floors = np.asarray(floors)
        self.conditional = conditional
        self.scales = scales

    def _regimes(self, modes: np.ndarray) -> np.ndarray:
        return modes if self.conditional else np.zeros_like(modes)

    def fit(self, patches: np.ndarray, modes: np.ndarray) -> ResidualScan:
        modes = self._regimes(modes)
        self.geometry = {}
        for width in self.scales:
            means = patches[:, -width:].mean(axis=1)
            for band in (None, *np.unique(modes)):
                subset = means if band is None else means[modes == band]
                if len(subset) < 80:
                    continue
                location = np.median(subset, axis=0)
                scale = np.maximum(
                    1.4826 * np.median(abs(subset - location), axis=0), self.floors
                )
                normal = np.clip((subset - location) / scale, -6, 6)
                covariance, _ = ledoit_wolf(normal)
                # On simulated telemetry some normal relations are almost exact.
                covariance.flat[:: len(scale) + 1] += 0.25
                precision = np.linalg.inv(covariance)
                projected = normal @ precision / np.sqrt(np.diag(precision))
                dense = np.einsum("ij,ij->i", normal @ precision, normal)
                tails = np.maximum(
                    np.quantile(
                        np.column_stack([np.max(abs(projected), axis=1) ** 2, dense]),
                        0.95,
                        axis=0,
                    ),
                    1.0,
                )
                self.geometry[width, band] = location, scale, precision, tails
            if (width, None) not in self.geometry:
                raise ValueError("Need at least 80 normal reference windows.")
        return self

    def score(self, patches: np.ndarray, modes: np.ndarray) -> dict[str, np.ndarray]:
        modes = self._regimes(modes)
        sparse = np.zeros(len(patches))
        dense = np.zeros(len(patches))
        diagonal = np.zeros(len(patches))
        for width in self.scales:
            means = patches[:, -width:].mean(axis=1)
            for band in np.unique(modes):
                mask = modes == band
                center, scale, precision, tails = self.geometry.get(
                    (width, band), self.geometry[width, None]
                )
                x = (means[mask] - center) / scale
                projected = x @ precision
                sparse[mask] = np.maximum(
                    sparse[mask],
                    (projected**2 / np.diag(precision)).max(axis=1) / tails[0],
                )
                dense[mask] = np.maximum(
                    dense[mask], np.einsum("ij,ij->i", projected, x) / tails[1]
                )
                diagonal[mask] = np.maximum(diagonal[mask], abs(x).max(axis=1) ** 2)
        return {
            "scan": np.maximum(sparse, dense),
            "sparse": sparse,
            "dense": dense,
            "diagonal": diagonal,
        }
