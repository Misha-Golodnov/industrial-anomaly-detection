"""Causal, compact PaAno adaptation for aligned AHU windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_novelty.data.lbnl import INTERVAL
from industrial_novelty.models.neural import _torch


class PatchMemory:
    """Triplet and time-offset pretext training, followed by cosine 3-NN."""

    def __init__(self, *, steps: int = 300, seed: int = 0):
        self.steps = steps
        self.seed = seed

    def fit(self, patches: np.ndarray, timestamps: pd.DatetimeIndex) -> PatchMemory:
        torch = _torch()
        nn = torch.nn
        normalize = nn.functional.normalize
        width = patches.shape[1]
        day = (timestamps - INTERVAL).normalize()
        usable = (timestamps[width:] - timestamps[:-width] == width * INTERVAL) & (
            day[width:] == day[:-width]
        )
        anchors = np.flatnonzero(usable) + width
        if len(anchors) < 128:
            raise ValueError("Need 128 patches with causal positive and pretext pairs.")
        data = torch.from_numpy(
            np.asarray(patches.transpose(0, 2, 1), dtype=np.float32)
        )
        rng = np.random.default_rng(self.seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.encoder = nn.Sequential(
                nn.Conv1d(data.shape[1], 32, 3, padding=1, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1),
            )
            projection = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 64))
            offset_head = nn.Linear(64, 1)
            optimizer = torch.optim.AdamW(
                [
                    *self.encoder.parameters(),
                    *projection.parameters(),
                    *offset_head.parameters(),
                ],
                lr=1e-4,
            )
            for step in range(self.steps):
                selected = rng.choice(anchors, size=128, replace=False)
                joined = torch.cat(
                    (data[selected], data[selected - 1], data[selected - width])
                )
                anchor, positive, previous = self.encoder(joined).chunk(3)
                cosine = (
                    normalize(projection(anchor)) @ normalize(projection(positive)).T
                )
                nearby = torch.from_numpy(
                    abs(selected[:, None] - selected[None, :]) <= width
                )
                hard_negative = cosine.masked_fill(nearby, -torch.inf).amax(dim=1)
                loss = torch.relu(0.1 + hard_negative - cosine.diagonal()).mean() / 10
                weight = max(1 - 5 * step / self.steps, 0)
                if weight:
                    unrelated = previous.roll(1, dims=0)
                    aligned = offset_head(torch.cat((anchor, previous), dim=1))
                    shuffled = offset_head(torch.cat((anchor, unrelated), dim=1))
                    loss = loss + weight * (
                        nn.functional.softplus(-aligned).mean()
                        + nn.functional.softplus(shuffled).mean()
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.encoder.eval()
        sample = rng.choice(len(patches), size=min(4096, len(patches)), replace=False)
        self.memory = self._embedding(data[sample])
        return self

    def _embedding(self, tensor):
        torch = _torch()
        with torch.inference_mode():
            return torch.nn.functional.normalize(self.encoder(tensor), dim=1)

    def score(self, patches: np.ndarray) -> np.ndarray:
        torch = _torch()
        data = torch.from_numpy(
            np.asarray(patches.transpose(0, 2, 1), dtype=np.float32)
        )
        with torch.inference_mode():
            chunks = [
                (1 - (self._embedding(batch) @ self.memory.T).topk(3).values).mean(1)
                for batch in data.split(512)
            ]
        return torch.cat(chunks).numpy().clip(0)
