"""Normal-reference selection of regularized Gaussian precision matrices."""

from __future__ import annotations

import warnings

import numpy as np

from industrial_novelty.models.chiller_directions import window_features


class ReferencePrecision:
    """Choose covariance regularization by Gaussian NLL on NORMAL reference.

    Density validation is allowed to use normal reference, not calibration or
    fault test. Correlated reference windows provide a development criterion,
    not an independent predictive-likelihood guarantee.
    """

    def __init__(self, kind="raw", *, policy="shrink", fixed=None):
        self.kind, self.policy, self.fixed = kind, policy, fixed

    def fit(self, X, reference):
        from sklearn.covariance import graphical_lasso
        from sklearn.exceptions import ConvergenceWarning

        features = window_features(X, self.kind)
        self.center_ = features.mean(axis=0)
        self.scale_ = np.maximum(features.std(axis=0), 0.05)
        z = (features - self.center_) / self.scale_
        ref = (window_features(reference, self.kind) - self.center_) / self.scale_
        # A numerical ridge also covers physical validity flags that are exactly
        # constant on a fit split; graphical lasso cannot invert a zero diagonal.
        sample = z.T @ z / len(z) + 1e-6 * np.eye(z.shape[1])
        grid = (
            [self.fixed]
            if self.fixed is not None
            else [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.3, 0.7, 1]
            if self.policy == "shrink"
            else [0.001, 0.01, 0.05, 0.1, 0.3, 1]
        )
        trials, candidates = [], []
        for parameter in grid:
            try:
                if self.policy == "shrink":
                    covariance = (1 - parameter) * sample + parameter * np.diag(
                        np.diag(sample)
                    )
                else:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        covariance, _ = graphical_lasso(
                            sample, alpha=parameter, max_iter=200, tol=1e-3
                        )
                    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                        trials.append(
                            {"parameter": parameter, "status": "not_converged"}
                        )
                        continue
                eigenvalues, vectors = np.linalg.eigh(covariance)
                eigenvalues = np.maximum(eigenvalues, 1e-10)
                precision = (vectors * (1 / eigenvalues)) @ vectors.T
                nll = 0.5 * (
                    np.log(eigenvalues).sum()
                    + np.einsum("ni,ij,nj->n", ref, precision, ref).mean()
                )
                if not np.isfinite(nll):
                    raise ValueError("Nonfinite reference NLL")
                trials.append(
                    {
                        "parameter": parameter,
                        "status": "ok",
                        "normal_reference_nll": float(nll),
                    }
                )
                candidates.append((nll, parameter, precision))
            except (FloatingPointError, ValueError, np.linalg.LinAlgError) as error:
                trials.append({"parameter": parameter, "status": type(error).__name__})
        if not candidates:
            raise ValueError("No valid precision candidate.")
        nll, parameter, self.precision_ = min(candidates, key=lambda t: t[0])
        self.diagnostics_ = {
            "selected_parameter": parameter,
            "normal_reference_nll": float(nll),
            "invalid_candidates": sum(t["status"] != "ok" for t in trials),
            "trials": trials,
        }
        return self

    def score(self, X, meta=None):
        z = (window_features(X, self.kind) - self.center_) / self.scale_
        return np.maximum(np.einsum("ni,ij,nj->n", z, self.precision_, z), 0)
