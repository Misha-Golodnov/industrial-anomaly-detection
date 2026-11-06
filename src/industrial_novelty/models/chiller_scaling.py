"""Unit-equivariant scaling ablation for mixed-unit physical indicators."""

from __future__ import annotations

import numpy as np

from industrial_novelty.models.chiller_directions import window_features


class UnitScaledPhysics:
    """Standardize every nonconstant physical coordinate without a .05 floor.

    Exactly constant channels use scale 1, with a numerical covariance ridge.
    For nonconstant features, changing feature units cancels from standardized
    coordinates, up to floating-point accuracy and the 1e-12 numerical cutoff.
    """

    def __init__(self, *, eos=False, covariance="lw"):
        self.kind = "thermo_eos" if eos else "thermo"
        self.covariance = covariance

    def fit(self, X, reference):
        from sklearn.covariance import OAS, LedoitWolf

        features = window_features(X, self.kind)
        self.center_ = features.mean(axis=0)
        spread = features.std(axis=0)
        self.scale_ = np.where(spread > 1e-12, spread, 1)
        z = (features - self.center_) / self.scale_
        ref = (window_features(reference, self.kind) - self.center_) / self.scale_
        self.constant_coordinates_ = int((spread <= 1e-12).sum())
        if self.covariance == "cv":
            sample = z.T @ z / len(z) + 1e-6 * np.eye(z.shape[1])
            candidates = []
            for lam in [1e-5, 1e-4, 0.001, 0.01, 0.05, 0.1, 0.3, 0.7, 1]:
                covariance = (1 - lam) * sample + lam * np.diag(np.diag(sample))
                eig, vec = np.linalg.eigh(covariance)
                eig = np.maximum(eig, 1e-10)
                precision = (vec * (1 / eig)) @ vec.T
                nll = 0.5 * (
                    np.log(eig).sum()
                    + np.einsum("ni,ij,nj->n", ref, precision, ref).mean()
                )
                candidates.append((nll, lam, precision))
            _, self.selected_parameter_, self.precision_ = min(
                candidates, key=lambda c: c[0]
            )
        else:
            metric = (LedoitWolf() if self.covariance == "lw" else OAS()).fit(z)
            self.selected_parameter_ = float(metric.shrinkage_)
            self.precision_ = metric.precision_
        return self

    def score(self, X, meta=None):
        z = (window_features(X, self.kind) - self.center_) / self.scale_
        return np.maximum(np.einsum("ni,ij,nj->n", z, self.precision_, z), 0)
