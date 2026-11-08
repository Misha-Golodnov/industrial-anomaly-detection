"""Severity-profiled physical fault signatures with a normal calibration gate."""

from __future__ import annotations

import numpy as np

from industrial_novelty.models.chiller_directions import (
    ConditionalResidual,
    FeatureGaussian,
)


class FaultSignatureClassifier:
    """Gaussian GLR for positive unknown severity, or a two-direction subspace.

    Ray model: r ~ N(a*d_f, I), a >= 0; maximizing over a yields
    evidence_f = max(r @ d_f, 0)^2 / ||d_f||^2. A normal calibration set
    supplies the rejection threshold. Whitening is estimated, so neither an
    exact Gaussian null nor a target-domain false-alarm guarantee is claimed.
    """

    def __init__(self, *, kind="thermo", conditional=False, subspace=False, alpha=0.05):
        self.kind, self.conditional, self.subspace = kind, conditional, subspace
        self.alpha = alpha

    def fit(self, X, y, reference, calibration):
        normal = X[y == 0]
        self.normal_ = (
            ConditionalResidual(self.kind, degree=1)
            if self.conditional
            else FeatureGaussian(self.kind)
        )
        self.normal_.fit(normal, reference)
        residual = self.normal_.residual(X)
        self.prototypes_ = []
        for label in range(1, 8):
            values = residual[y == label]
            if not len(values):
                raise ValueError(
                    "Need training data for all seven known fault classes."
                )
            if self.subspace:
                _, _, vt = np.linalg.svd(values, full_matrices=False)
                self.prototypes_.append(vt[:2].T)
            else:
                mean = values.mean(axis=0)
                self.prototypes_.append(mean / max(np.linalg.norm(mean), 1e-12))
        cal_scores = self.evidence(calibration).max(axis=1)
        rank = int(np.ceil((len(cal_scores) + 1) * (1 - self.alpha)))
        self.threshold_ = (
            float(np.partition(cal_scores, rank - 1)[rank - 1])
            if rank <= len(cal_scores)
            else float("inf")
        )
        return self

    def evidence(self, X):
        residual = self.normal_.residual(X)
        if self.subspace:
            return np.column_stack(
                [np.sum((residual @ basis) ** 2, axis=1) for basis in self.prototypes_]
            )
        return np.maximum(residual @ np.stack(self.prototypes_).T, 0) ** 2

    def predict(self, X):
        evidence = self.evidence(X)
        label = evidence.argmax(axis=1) + 1
        return np.where(evidence.max(axis=1) > self.threshold_, label, 0)
