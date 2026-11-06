"""Literature-guided RP-1043 extensions on raw, unstandardized windows.

Physical formulas: Bezyan et al., Electronics 2026, Table 3.
These compact models are adaptations, not replicas of FDSD or VMD-KPCA.
Every transformation is confined to an already completed input window.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from industrial_novelty.data.ashrae import CHANNELS

INDEX = {name: i for i, name in enumerate(CHANNELS)}
PHYSICAL_NAMES = (
    "delta_evap_water",
    "delta_cond_water",
    "evap_approach",
    "cond_approach",
    "subcooling",
    "suction_superheat",
    "discharge_superheat",
    "epsilon_subcool",
    "epsilon_superheat",
    "lmtd_evap",
    "lmtd_cond",
    "ua_evap_proxy",
    "ua_cond_proxy",
    "inverse_cop_proxy",
    "heat_ratio_proxy",
    "flow_evap",
    "flow_cond",
    "power",
    "pressure_lift",
    "oil_net_pressure",
    "oil_temperature_difference",
    "amps",
    "evap_in_sensor_disagreement",
    "evap_out_sensor_disagreement",
    "invalid_evap_lmtd",
    "invalid_cond_lmtd",
)


def logarithmic_mean(a, b):
    """Stable logarithmic mean for positive endpoints; 0 + flag otherwise.

    No silent removal of invalid measurements, and no invented continuation of
    a steady-state heat-exchanger law into an invalid transient state.
    """
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    valid = (a > 1e-6) & (b > 1e-6)
    average = (np.where(valid, a, 1) + np.where(valid, b, 1)) / 2
    z = np.where(valid, (a - b) / (2 * average), 0)
    ratio = np.ones_like(z)
    small = np.abs(z) < 1e-4
    ratio[small] = 1 - z[small] ** 2 / 3 - 4 * z[small] ** 4 / 45
    ratio[~small] = z[~small] / np.arctanh(z[~small])
    return np.where(valid, average * ratio, 0), ~valid


def protected_ratio(a, b, floor=0.5):
    denominator = np.where(np.abs(b) < floor, np.copysign(floor, b), b)
    return a / denominator


@lru_cache(maxsize=20000)
def r134a_saturation_psig(temperature_f):
    """R134a saturation at Fahrenheit T, returned in psig at 1 atm reference."""
    from CoolProp.CoolProp import PropsSI

    kelvin = (float(temperature_f) - 32) * 5 / 9 + 273.15
    return float(
        PropsSI("P", "T", kelvin, "Q", 1, "R134a") / 6894.757293168 - 14.6959488
    )


def physical_series(X, *, eos=False):
    """No absolute temperature levels. gpm*F proxies are not SI heat rates.

    TRC_sub is already a temperature DIFFERENCE, not an absolute liquid-line T.
    Saturation residuals additionally assume the R134a/psig/F schema and are
    exposed as a separate ablation, not as an unquestioned hard constraint.
    """
    X = np.asarray(X, dtype=float)
    d = {name: X[..., i] for name, i in INDEX.items()}
    de, dc = d["TWEI"] - d["TWEO"], d["TWCO"] - d["TWCI"]
    le, invalid_e = logarithmic_mean(d["TWEI"] - d["TRE"], d["TWEO"] - d["TRE"])
    lc, invalid_c = logarithmic_mean(d["TRC"] - d["TWCI"], d["TRC"] - d["TWCO"])
    qe, qc = d["FWE"] * de, d["FWC"] * dc
    values = [
        de,
        dc,
        d["TEO"] - d["TRE"],
        d["TRC"] - d["TCO"],
        d["TRC_sub"],
        d["Tsh_suc"],
        d["Tsh_dis"],
        protected_ratio(d["TRC_sub"], d["TRC"] - d["TWCI"]),
        protected_ratio(d["Tsh_dis"], d["TR_dis"] - d["TWCO"]),
        le,
        lc,
        np.where(invalid_e, 0, protected_ratio(qe, le)),
        np.where(invalid_c, 0, protected_ratio(qc, lc)),
        protected_ratio(d["kW"], qe, floor=10),
        protected_ratio(qc, qe, floor=10),
        d["FWE"],
        d["FWC"],
        d["kW"],
        d["PRC"] - d["PRE"],
        d["PO_net"],
        d["TO_sump"] - d["TO_feed"],
        d["Amps"],
        d["TEI"] - d["TWEI"],
        d["TEO"] - d["TWEO"],
        invalid_e.astype(float),
        invalid_c.astype(float),
    ]
    if eos:
        for temperature, pressure in [("TRE", "PRE"), ("TRC", "PRC")]:
            unique, indices = np.unique(d[temperature], return_inverse=True)
            predicted = np.array([r134a_saturation_psig(float(v)) for v in unique])[
                indices
            ].reshape(d[temperature].shape)
            values.append(d[pressure] - predicted)
    result = np.stack(values, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite physical features.")
    return result


def invariant_series(X):
    """Translate water/refrigerant temperatures by their respective inlet/setpoint."""
    d = {name: np.asarray(X[..., i], float) for name, i in INDEX.items()}
    values = [
        d[name] - d["TWE_set"]
        for name in ("TEI", "TEO", "TWEI", "TWEO", "TRE", "T_suc")
    ]
    values += [d[name] - d["TWCI"] for name in ("TCI", "TCO", "TWCO", "TRC")]
    values += [d["TR_dis"] - d["TRC"], d["TO_sump"] - d["TO_feed"]]
    values += [
        d[name]
        for name in (
            "kW",
            "FWC",
            "FWE",
            "PRE",
            "PRC",
            "TRC_sub",
            "Tsh_suc",
            "Tsh_dis",
            "Amps",
            "PO_feed",
            "PO_net",
        )
    ]
    return np.stack(values, axis=-1)


def spectral_features(series):
    """Window level plus two spectral energies and a trend; no full-file FFT."""
    centered = series - series.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2 / series.shape[1] ** 2
    split = max(2, spectrum.shape[1] // 3)
    low = np.sqrt(spectrum[:, 1:split].sum(axis=1))
    high = np.sqrt(spectrum[:, split:].sum(axis=1))
    trend = series[:, -1] - series[:, 0]
    return np.column_stack([series.mean(axis=1), low, high, trend])


def window_features(X, kind):
    if kind == "raw":
        return X.mean(axis=1).astype(float)
    if kind == "raw_moments":
        return np.column_stack([X.mean(axis=1), X.std(axis=1)])
    if kind == "invariant":
        return invariant_series(X).mean(axis=1)
    if kind in ("thermo", "thermo_eos", "thermo_moments"):
        series = physical_series(X, eos=kind == "thermo_eos")
        if kind == "thermo_moments":
            return np.column_stack([series.mean(axis=1), series.std(axis=1)])
        return series.mean(axis=1)
    if kind == "spectral":
        return spectral_features(X.astype(float))
    if kind == "thermo_spectral":
        return spectral_features(physical_series(X))
    if kind == "raw_thermo_moments":
        return np.column_stack(
            [window_features(X, "raw_moments"), window_features(X, "thermo_moments")]
        )
    raise ValueError(kind)


class FeatureGaussian:
    """Shrinkage Gaussian on fixed features, or its retained/residual subspace."""

    def __init__(self, kind="thermo", *, covariance="lw", pca_residual=False):
        self.kind, self.covariance, self.pca_residual = kind, covariance, pca_residual

    def fit(self, X, reference=None):
        from sklearn.covariance import OAS, LedoitWolf

        # Match the original baseline's channel scaling exactly when kind=raw.
        raw = X.reshape(-1, X.shape[-1]).astype(float)
        self.raw_mean_, self.raw_scale_ = (
            raw.mean(axis=0),
            np.maximum(raw.std(axis=0), 1e-6),
        )
        features = window_features(X, self.kind)
        self.center_ = features.mean(axis=0)
        self.scale_ = (
            self.raw_scale_
            if self.kind == "raw"
            else np.maximum(features.std(axis=0), 0.05)
        )
        z = (features - self.center_) / self.scale_
        estimator = LedoitWolf() if self.covariance == "lw" else OAS()
        self.metric_ = estimator.fit(z)
        eigenvalues, vectors = np.linalg.eigh(self.metric_.covariance_)
        self.whitener_ = vectors * (1 / np.sqrt(np.maximum(eigenvalues, 1e-10)))
        if self.pca_residual:
            _, singular, vt = np.linalg.svd(z, full_matrices=False)
            rank = int(
                np.searchsorted(np.cumsum(singular**2) / np.sum(singular**2), 0.95) + 1
            )
            self.components_ = vt[: min(rank, z.shape[1] - 1)].T
        return self

    def residual(self, X):
        z = (window_features(X, self.kind) - self.center_) / self.scale_
        return (z - self.metric_.location_) @ self.whitener_

    def score(self, X, meta=None):
        if self.pca_residual:
            z = (window_features(X, self.kind) - self.center_) / self.scale_
            return np.maximum(
                np.sum(z * z, axis=1) - np.sum((z @ self.components_) ** 2, axis=1), 0
            )
        return np.sum(self.residual(X) ** 2, axis=1)


class ConditionalResidual:
    """Polynomial or GP mean with predictive uncertainty and reference covariance.

    Only external context TWE_set/TWCI is used. No measured flow/power/load is
    conditioned away. GP hyperparameters are fixed before fault evaluation.
    """

    def __init__(self, kind="thermo", *, mean_model="poly", degree=1, uncertainty=True):
        self.kind, self.mean_model = kind, mean_model
        self.degree, self.uncertainty = degree, uncertainty

    def _target(self, X):
        features = window_features(X, self.kind)
        return features[:, 2:] if self.kind == "raw" else features

    def fit(self, X, reference):
        from sklearn.covariance import LedoitWolf
        from sklearn.preprocessing import PolynomialFeatures

        c = X.mean(axis=1)[:, :2].astype(float)
        self.context_mean_, self.context_scale_ = (
            c.mean(axis=0),
            np.maximum(c.std(axis=0), 1e-6),
        )
        c = (c - self.context_mean_) / self.context_scale_
        y = self._target(X)
        self.y_mean_, self.y_scale_ = y.mean(axis=0), np.maximum(y.std(axis=0), 0.05)
        y = (y - self.y_mean_) / self.y_scale_
        if self.mean_model == "gp":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel

            self.model_ = GaussianProcessRegressor(
                kernel=Matern(length_scale=1.5, nu=1.5) + WhiteKernel(0.05),
                optimizer=None,
                alpha=1e-8,
            ).fit(c, y)
        else:
            self.poly_ = PolynomialFeatures(self.degree)
            design = self.poly_.fit_transform(c)
            penalty = np.eye(design.shape[1])
            penalty[0, 0] = 1e-8
            self.inverse_ = np.linalg.inv(design.T @ design + penalty)
            self.coefficients_ = self.inverse_ @ design.T @ y
        residual = self._raw_residual(reference)
        self.metric_ = LedoitWolf().fit(residual)
        eigenvalues, vectors = np.linalg.eigh(self.metric_.covariance_)
        self.whitener_ = vectors * (1 / np.sqrt(np.maximum(eigenvalues, 1e-10)))
        return self

    def _raw_residual(self, X):
        c = (X.mean(axis=1)[:, :2] - self.context_mean_) / self.context_scale_
        y = (self._target(X) - self.y_mean_) / self.y_scale_
        if self.mean_model == "gp":
            mu, std = self.model_.predict(c, return_std=True)
            if std.ndim == 1:
                std = std[:, None]
            spread = np.sqrt(1 + std**2) if self.uncertainty else 1
        else:
            design = self.poly_.transform(c)
            mu = design @ self.coefficients_
            leverage = np.einsum("ni,ij,nj->n", design, self.inverse_, design)
            spread = np.sqrt(1 + leverage[:, None]) if self.uncertainty else 1
        return (y - mu) / spread

    def residual(self, X):
        return (self._raw_residual(X) - self.metric_.location_) @ self.whitener_

    def score(self, X, meta=None):
        return np.sum(self.residual(X) ** 2, axis=1)


def contiguous_groups(meta, stride=6):
    """Respect artificial blocks AND gaps from invalid observations."""
    block = meta.block.to_numpy()
    ends = meta.row_end.to_numpy()
    cut = np.r_[True, (block[1:] != block[:-1]) | (ends[1:] - ends[:-1] != stride)]
    return np.split(np.arange(len(meta)), np.flatnonzero(cut)[1:])


class SequentialResidual:
    """Vector MEWMA or multiscale accumulation, reset at every block/gap.

    These are empirical online scores. Overlapping windows are dependent; the
    formulas do NOT imply a chi-square or anytime false-alarm guarantee.
    """

    def __init__(self, inner, *, mode="mewma", lam=0.2):
        self.inner, self.mode, self.lam = inner, mode, lam

    def fit(self, X, reference):
        self.inner.fit(X, reference)
        return self

    def score(self, X, meta=None):
        if meta is None:
            raise ValueError("Sequential scoring requires block/row metadata.")
        residual = self.inner.residual(X)
        result = np.empty(len(X))
        for indices in contiguous_groups(meta):
            values = residual[indices]
            if self.mode == "mewma":
                state = np.zeros(values.shape[1])
                for i, row in enumerate(values):
                    state = (1 - self.lam) * state + self.lam * row
                    result[indices[i]] = state @ state
            else:
                sums = np.vstack([np.zeros(values.shape[1]), np.cumsum(values, axis=0)])
                for i in range(len(values)):
                    result[indices[i]] = max(
                        np.sum((sums[i + 1] - sums[max(0, i + 1 - l)]) ** 2)
                        / min(l, i + 1)
                        for l in (1, 2, 4, 8)
                    )
        return result
