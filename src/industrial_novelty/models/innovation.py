"""Causal autoregressive innovations and multiscale tests of their moments."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from industrial_novelty.models.subspace import tail_scale

MOMENTS = ("mean", "variance", "lag_correlation")


def lagged(runs: np.ndarray, order: int) -> np.ndarray:
    """At each target, concatenate only preceding observations of the same run."""
    return np.concatenate(
        [runs[:, order - lag : runs.shape[1] - lag] for lag in range(1, order + 1)],
        axis=2,
    )


def moving_mean(values: np.ndarray, width: int) -> np.ndarray:
    total = np.cumsum(values, axis=1, dtype=np.float64)
    total[:, width:] -= total[:, :-width]
    return total[:, width - 1 :] / width


def rolling_moments(innovations: np.ndarray, width: int) -> dict[str, np.ndarray]:
    """Each vector uses the last width innovation pairs and their marginal moments."""
    current, previous = innovations[:, 1:], innovations[:, :-1]
    mean = moving_mean(current, width)
    past_mean = moving_mean(previous, width)
    var = np.maximum(moving_mean(current**2, width) - mean**2, 0)
    past_var = np.maximum(moving_mean(previous**2, width) - past_mean**2, 0)
    lag_cov = moving_mean(current * previous, width) - mean * past_mean
    return {
        "mean": mean,
        "variance": np.log(var + 0.05),
        "lag_correlation": lag_cov / np.sqrt((var + 0.25) * (past_var + 0.25)),
    }


def moment_scores(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    squared = ((x - center) / scale) ** 2
    return np.stack([squared.max(axis=-1), squared.mean(axis=-1)], axis=-1)


class InnovationScan:
    """Ridge VAR with frozen coefficients, followed by empirical score calibration."""

    def __init__(self, *, order: int = 4, widths: tuple[int, ...] = (8, 16, 32, 64)):
        self.order = order
        self.widths = widths

    def fit(self, fit: np.ndarray, reference: np.ndarray) -> InnovationScan:
        features = lagged(fit, self.order)
        self.model = Ridge(alpha=100.0).fit(
            features.reshape(-1, features.shape[-1]),
            fit[:, self.order :].reshape(-1, fit.shape[-1]),
        )
        errors = self.residuals(reference)
        pooled = errors.reshape(-1, errors.shape[-1])
        self.location = np.median(pooled, axis=0)
        self.spread = np.maximum(
            1.4826 * np.median(abs(pooled - self.location), axis=0), 0.005
        )
        z = (errors - self.location) / self.spread
        self.reference = {}
        for width in self.widths:
            for name, values in rolling_moments(z, width).items():
                rows = values.reshape(-1, values.shape[-1])
                center = np.median(rows, axis=0)
                scale = np.maximum(1.4826 * np.median(abs(rows - center), axis=0), 0.02)
                raw = moment_scores(rows, center, scale)
                tails = np.array([tail_scale(column, 0.01) for column in raw.T])
                self.reference[width, name] = center, scale, tails
        return self

    def residuals(self, runs: np.ndarray) -> np.ndarray:
        features = lagged(runs, self.order)
        targets = runs[:, self.order :]
        predicted = self.model.predict(features.reshape(-1, features.shape[-1]))
        return targets - predicted.reshape(targets.shape)

    def score_mean(self, runs: np.ndarray) -> np.ndarray:
        """Return the mean-shift branch without calculating the other moments."""
        z = (self.residuals(runs) - self.location) / self.spread
        result = np.zeros(runs.shape[:2])
        for width in self.widths:
            center, scale, tails = self.reference[width, "mean"]
            means = moving_mean(z[:, 1:], width)
            values = (moment_scores(means, center, scale) / tails).max(axis=-1)
            target = result[:, self.order + width :]
            np.maximum(target, values, out=target)
        return result

    def score(self, runs: np.ndarray) -> dict[str, np.ndarray]:
        z = (self.residuals(runs) - self.location) / self.spread
        result = {name: np.zeros(runs.shape[:2]) for name in MOMENTS}
        for width in self.widths:
            for name, x in rolling_moments(z, width).items():
                center, scale, tails = self.reference[width, name]
                values = (moment_scores(x, center, scale) / tails).max(axis=-1)
                target = result[name][:, self.order + width :]
                np.maximum(target, values, out=target)
        result["joint"] = np.maximum.reduce(list(result.values()))
        return result
