"""Refit the three normal-only detectors on a common target-domain window grid."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

from industrial_novelty.models.canonical import CanonicalVariateIndex
from industrial_novelty.models.chart import accumulate_runs
from industrial_novelty.models.chiller_directions import FeatureGaussian
from industrial_novelty.models.chiller_precision import ReferencePrecision
from industrial_novelty.models.lbnl import REGRESSION_FLOORS, SENSORS, virtual_regressor
from industrial_novelty.models.slow import SlowFeatureIndex
from industrial_novelty.models.subspace import WindowMomentIndex, tail_scale

SINGLES = ("tep_dynamic", "ashrae_precision", "lbnl_virtual")
MAX_FUSIONS = {
    "max_dynamic_precision": SINGLES[:2],
    "max_dynamic_virtual": (SINGLES[0], SINGLES[2]),
    "max_precision_virtual": SINGLES[1:],
    "max_all": SINGLES,
}


class VirtualSensors:
    """Predict each target from other simultaneous channels, using only fit normal."""

    def __init__(self, channels: tuple[str, ...], *, ahu_graph: bool):
        self.channels = channels
        self.ahu_graph = ahu_graph

    def fit(self, normal: np.ndarray) -> VirtualSensors:
        if self.ahu_graph:
            targets = [self.channels.index(name) for name in SENSORS]
            parents = [
                [self.channels.index(name) for name in names]
                for names in SENSORS.values()
            ]
            self.floors = np.asarray(REGRESSION_FLOORS)
        else:
            targets = list(range(normal.shape[1]))
            # Absolute Spearman similarity is invariant to channel units.
            ranks = rankdata(normal, axis=0)
            centered = ranks - ranks.mean(axis=0)
            norms = np.linalg.norm(centered, axis=0)
            unit = centered / np.where(norms > 0, norms, 1)
            similarity = abs(unit.T @ unit)
            varying = np.flatnonzero(norms > 0)
            parents = []
            for target in targets:
                candidates = varying[varying != target]
                order = np.argsort(-similarity[target, candidates], kind="stable")
                parents.append(candidates[order[:6]].tolist())
            self.floors = np.maximum(
                0.01 * normal.std(axis=0),
                1e-5 * np.maximum(1, np.median(abs(normal), axis=0)),
            )
        self.targets = targets
        self.parents = parents
        self.defaults = np.median(normal[:, targets], axis=0)
        min_leaf = max(8, min(35, len(normal) // 20))
        self.models = [
            virtual_regressor(min_samples_leaf=min_leaf).fit(
                normal[:, inputs], normal[:, target]
            )
            if inputs and np.ptp(normal[:, target]) > 0
            else None
            for target, inputs in zip(targets, parents, strict=True)
        ]
        return self

    def residuals(self, values: np.ndarray) -> np.ndarray:
        predictions = np.column_stack(
            [
                model.predict(values[:, inputs])
                if model is not None
                else np.full(len(values), default)
                for model, inputs, default in zip(
                    self.models, self.parents, self.defaults, strict=True
                )
            ]
        )
        return values[:, self.targets] - predictions

    def graph(self) -> dict[str, list[str]]:
        return {
            self.channels[target]: [self.channels[parent] for parent in inputs]
            for target, inputs in zip(self.targets, self.parents, strict=True)
        }


class TransferSuite:
    def __init__(self, channels, *, ahu_graph=False, pre_scaled=False):
        self.channels = tuple(channels)
        self.pre_scaled = pre_scaled
        self.virtual = VirtualSensors(self.channels, ahu_graph=ahu_graph)
        self.precision = ReferencePrecision("raw")
        self.dynamics = (
            CanonicalVariateIndex(),
            WindowMomentIndex(),
            SlowFeatureIndex(),
        )

    def _scaled(self, windows: np.ndarray) -> np.ndarray:
        if self.pre_scaled:
            return windows
        return self.scaler.transform(windows.reshape(-1, windows.shape[-1])).reshape(
            windows.shape
        )

    def _dynamic_raw(self, scaled: np.ndarray) -> np.ndarray:
        return np.maximum.reduce(
            [
                detector.score(scaled) / scale
                for detector, scale in zip(
                    self.dynamics, self.dynamic_scales, strict=True
                )
            ]
        )

    def fit(self, fit, reference, ref_group, ref_end):
        if not self.pre_scaled:
            self.scaler = StandardScaler().fit(fit.reshape(-1, fit.shape[-1]))
        scaled = self._scaled(fit)
        self.dynamic_scales = []
        for detector in self.dynamics:
            detector.fit(scaled)
            self.dynamic_scales.append(tail_scale(detector.score(scaled), 0.01))
        raw = self._dynamic_raw(scaled)
        self.dynamic_center, self.dynamic_spread = raw.mean(), raw.std()
        self.precision.fit(fit, reference)
        self.virtual.fit(fit[:, -1])
        residuals = self.virtual.residuals(reference[:, -1])
        self.virtual_center = np.median(residuals, axis=0)
        self.virtual_scale = np.maximum(
            1.4826 * np.median(abs(residuals - self.virtual_center), axis=0),
            self.virtual.floors,
        )
        normalized = (residuals - self.virtual_center) / self.virtual_scale
        self.residual_precision = FeatureGaussian("raw").fit(normalized[:, None])
        ref_scores = self._single_scores(reference, ref_group, ref_end, normalized)
        self.fusion_scales = {}
        for name in SINGLES:
            center = np.median(ref_scores[name])
            spread = tail_scale(np.maximum(ref_scores[name] - center, 0), 0.01)
            self.fusion_scales[name] = (float(center), spread)
        return self

    def _single_scores(self, windows, group, end, normalized):
        return {
            "tep_dynamic": accumulate_runs(
                self._dynamic_raw(self._scaled(windows)),
                group,
                end,
                lam=0.2,
                center=self.dynamic_center,
                spread=self.dynamic_spread,
            ),
            "ashrae_precision": self.precision.score(windows),
            "lbnl_virtual": abs(normalized).max(axis=1) ** 2,
            "virtual_cov": self.residual_precision.score(normalized[:, None]),
        }

    def score(self, windows, group, end):
        residuals = self.virtual.residuals(windows[:, -1])
        normalized = (residuals - self.virtual_center) / self.virtual_scale
        scores = self._single_scores(windows, group, end, normalized)
        evidence = {
            name: np.log1p(np.maximum((scores[name] - center) / spread, 0))
            for name, (center, spread) in self.fusion_scales.items()
        }
        scores.update(
            {
                name: np.maximum.reduce([evidence[key] for key in members])
                for name, members in MAX_FUSIONS.items()
            }
        )
        scores["mean_all"] = np.mean([evidence[name] for name in SINGLES], axis=0)
        matrix = np.column_stack(list(scores.values()))
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite transfer scores.")
        return matrix, list(scores)
