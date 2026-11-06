"""Exploratory conditional detectors; no claim of a reproduced RP-1043 SOTA.

Input arrays use the allowlisted CHANNELS in data.ashrae. Scaling is external
and fitted on normal fit blocks only. Conditional models use window means.
"""

from __future__ import annotations

import numpy as np


def tangent_energy(residual, jacobian, *, parallel_weight=0.0):
    """Whitened nuisance projection, with Moore-Penrose rank truncation.

    residual: (n,d), jacobian: (n,d,k). Both must already be whitened in
    the same metric. A parallel fault is unidentifiable after full projection.
    """
    r = np.asarray(residual, dtype=float)
    j = np.asarray(jacobian, dtype=float)
    if r.ndim != 2 or j.ndim != 3 or j.shape[:2] != r.shape:
        raise ValueError("Expected residual (n,d), jacobian (n,d,k).")
    if (
        not 0 <= parallel_weight <= 1
        or not np.isfinite(r).all()
        or not np.isfinite(j).all()
    ):
        raise ValueError("Invalid projection inputs.")
    u, s, _ = np.linalg.svd(j, full_matrices=False)
    retained = s > np.maximum(s[:, :1] * 1e-8, 1e-12)
    parallel = np.sum((np.einsum("nd,ndk->nk", r, u) * retained) ** 2, axis=1)
    total = np.sum(r * r, axis=1)
    return np.maximum(total - (1 - parallel_weight) * parallel, 0)


class ConditionalRBF:
    """Normal RBF-ridge mean, independent reference covariance, optional tangent.

    This is a proposed ablation, not a replication of LFD/FDSD or CFlow-cf.
    Conditioning variables occupy the first two channels.
    """

    def __init__(self, *, seed=0, centers=64, ridge=1.0, parallel_weight=1.0):
        self.seed, self.centers = seed, centers
        self.ridge, self.parallel_weight = ridge, parallel_weight

    def _features(self, c):
        distance = np.sum((c[:, None] - self.centers_[None]) ** 2, axis=2)
        return np.column_stack([np.ones(len(c)), c, np.exp(-self.gamma_ * distance)])

    def _predict(self, c):
        return self._features(c) @ self.coef_

    def fit(self, X, reference):
        from sklearn.covariance import LedoitWolf

        means = X.mean(axis=1).astype(float)
        c, y = means[:, :2], means[:, 2:]
        rng = np.random.default_rng(self.seed)
        self.centers_ = c[rng.choice(len(c), min(self.centers, len(c)), replace=False)]
        distances = np.sum((self.centers_[:, None] - self.centers_[None]) ** 2, axis=2)
        nonzero = distances[distances > 1e-10]
        self.gamma_ = 1 / (np.median(nonzero) if len(nonzero) else 1.0)
        phi = self._features(c)
        penalty = np.eye(phi.shape[1]) * self.ridge
        penalty[0, 0] = 1e-8
        self.coef_ = np.linalg.solve(phi.T @ phi + penalty, phi.T @ y)
        ref = reference.mean(axis=1).astype(float)
        residuals = ref[:, 2:] - self._predict(ref[:, :2])
        covariance = LedoitWolf().fit(residuals)
        self.bias_ = covariance.location_
        eig, vec = np.linalg.eigh(covariance.covariance_)
        self.whitener_ = vec * (
            1 / np.sqrt(np.maximum(eig, max(eig.max() * 1e-8, 1e-12)))
        )
        return self

    def score(self, X):
        means = X.mean(axis=1).astype(float)
        c = means[:, :2]
        residual = (means[:, 2:] - self._predict(c) - self.bias_) @ self.whitener_
        if self.parallel_weight == 1:
            return np.sum(residual * residual, axis=1)
        jacobian = []
        for k in range(2):
            delta = np.zeros_like(c)
            delta[:, k] = 1e-3
            derivative = (self._predict(c + delta) - self._predict(c - delta)) / 2e-3
            jacobian.append(derivative @ self.whitener_)
        return tangent_energy(
            residual, np.stack(jacobian, axis=2), parallel_weight=self.parallel_weight
        )


class TorchMeanDetector:
    """Small AE or conditional affine flow on means; fixed normal-only training.

    No early stopping or hyperparameter selection using fault data. Log-density
    is in fit-standardized response coordinates. This is a compact comparator,
    not an implementation of a named chiller Transformer paper.
    """

    def __init__(self, *, kind="flow", epochs=60, seed=0):
        if kind not in ("flow", "ae"):
            raise ValueError(kind)
        self.kind, self.epochs, self.seed = kind, epochs, seed

    def fit(self, X):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        data = torch.as_tensor(X.mean(axis=1), dtype=torch.float32)
        self.dimension_ = data.shape[1]
        d = self.dimension_ - 2

        class Coupling(nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.register_buffer("mask", ((torch.arange(d) + layer) % 2).float())
                self.net = nn.Sequential(
                    nn.Linear(d + 2, 64),
                    nn.SiLU(),
                    nn.Linear(64, 64),
                    nn.SiLU(),
                    nn.Linear(64, 2 * d),
                )
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)

            def forward(self, y, c):
                s, t = self.net(torch.cat([y * self.mask, c], dim=1)).chunk(2, dim=1)
                s = 1.5 * torch.tanh(s) * (1 - self.mask)
                t = t * (1 - self.mask)
                z = y * self.mask + (1 - self.mask) * (y - t) * torch.exp(-s)
                return z, -s.sum(dim=1)

        if self.kind == "ae":
            self.net_ = nn.Sequential(
                nn.Linear(self.dimension_, 64),
                nn.SiLU(),
                nn.Linear(64, 8),
                nn.SiLU(),
                nn.Linear(8, 64),
                nn.SiLU(),
                nn.Linear(64, self.dimension_),
            )
        else:
            self.net_ = nn.ModuleList([Coupling(i) for i in range(4)])
        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=1e-3, weight_decay=1e-4
        )
        rng = torch.Generator().manual_seed(self.seed)
        self.losses_ = []
        self.net_.train()
        for _ in range(self.epochs):
            losses = []
            for indices in torch.randperm(len(data), generator=rng).split(128):
                loss = self._loss(data[indices]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite neural training loss.")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 5)
                optimizer.step()
                losses.append(loss.detach().item())
            self.losses_.append(float(np.mean(losses)))
        self.net_.eval()
        return self

    def _loss(self, x):
        import torch

        if self.kind == "ae":
            return (self.net_(x) - x).square().mean(dim=1)
        y, c = x[:, 2:], x[:, :2]
        logdet = torch.zeros(len(x), device=x.device)
        for layer in self.net_:
            y, increment = layer(y, c)
            logdet = logdet + increment
        return 0.5 * (y.square() + np.log(2 * np.pi)).sum(dim=1) - logdet

    def score(self, X):
        import torch

        with torch.no_grad():
            data = torch.as_tensor(X.mean(axis=1), dtype=torch.float32)
            return np.concatenate(
                [self._loss(part).numpy() for part in data.split(2048)]
            )
