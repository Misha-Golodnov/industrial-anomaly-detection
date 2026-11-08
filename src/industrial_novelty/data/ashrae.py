"""Explicit RP-1043 derivative schema and conservative within-file windows.

Never infer independent experiments from rounded timestamps. Blocks below are
artificial holdouts of a single trajectory, not replicated experimental runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Controller state, fault codes, time, rig valves and derived COP/heat balances
# are excluded by an allowlist. Water flows remain responses: conditioning on
# them would suppress the reduced-flow faults being sought.
CONTEXT = ("TWE_set", "TWCI")
RESPONSES = (
    "TEI",
    "TEO",
    "TWEI",
    "TWEO",
    "TCI",
    "TCO",
    "TWCO",
    "kW",
    "FWC",
    "FWE",
    "TRE",
    "PRE",
    "TRC",
    "PRC",
    "TRC_sub",
    "T_suc",
    "Tsh_suc",
    "TR_dis",
    "Tsh_dis",
    "Amps",
    "TO_sump",
    "TO_feed",
    "PO_feed",
    "PO_net",
)
CHANNELS = CONTEXT + RESPONSES
FAULTS = ("cf", "eo", "fwc", "fwe", "nc", "rl", "ro")


@dataclass
class ChillerWindows:
    X: np.ndarray
    meta: pd.DataFrame

    def take(self, indices):
        return ChillerWindows(
            self.X[indices], self.meta.iloc[indices].reset_index(drop=True)
        )


def chiller_windows(
    path: str | Path,
    *,
    window: int = 12,
    stride: int = 6,
    block_size: int = 180,
    embargo: int = 12,
) -> ChillerWindows:
    """Exclude OFF/invalid samples by sensor rules; never bridge removed rows.

    Embargo discards both edges of every artificial block. Windows cannot
    cross blocks, invalid observations or nominal setpoint bands. Temperatures
    are the source Fahrenheit values; no implicit SI conversion is performed.
    """
    if (
        min(window, stride, block_size) < 1
        or embargo < 0
        or block_size <= 2 * embargo + window
    ):
        raise ValueError("Invalid window/block/embargo settings.")
    path = Path(path)
    frame = pd.read_csv(path)
    missing = sorted(set(CHANNELS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing ASHRAE columns: {missing}")
    values = frame.loc[:, list(CHANNELS)].to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    # Fixed, broad engineering envelope. -99.4 is an observed sentinel; tiny
    # nonzero values are retained in audit and cannot count as running power.
    temperature = [c for c in CHANNELS if c.startswith("T")]
    valid &= frame[temperature].gt(-10).all(axis=1).to_numpy()
    valid &= frame[temperature].lt(200).all(axis=1).to_numpy()
    valid &= frame.kW.gt(5).to_numpy()
    valid &= frame.FWC.gt(10).to_numpy() & frame.FWE.gt(10).to_numpy()
    bands = (5 * np.rint(frame.TWE_set.to_numpy() / 5)).astype(int)
    blocks = np.arange(len(frame)) // block_size
    windows, records = [], []
    for block in np.unique(blocks):
        start = int(block * block_size + embargo)
        stop = int(min((block + 1) * block_size, len(frame)) - embargo)
        for left in range(start, stop - window + 1, stride):
            right = left + window
            if valid[left:right].all() and np.all(bands[left:right] == bands[left]):
                windows.append(values[left:right])
                records.append(
                    {
                        "file": path.name,
                        "block": int(block),
                        "row_start": left,
                        "row_end": right - 1,
                        "temperature_band": int(bands[left]),
                    }
                )
    if not windows:
        raise ValueError(f"No valid windows in {path}.")
    return ChillerWindows(np.asarray(windows, dtype=np.float32), pd.DataFrame(records))


def normal_split(batch: ChillerWindows, seed: int, *, held_out_band: int | None = None):
    """Disjoint block assignments with an explicit source-domain calibration.

    For interpolation, stratify blocks by their majority temperature band.
    For extrapolation, discard any block touching the held-out band from fit,
    reference and calibration. All its retained windows form the normal test.
    """
    meta = batch.meta
    rng = np.random.default_rng(seed)
    block_band = meta.groupby("block").temperature_band.agg(lambda x: x.mode().iloc[0])
    assignments = {}
    if held_out_band is not None:
        target = set(meta.loc[meta.temperature_band == held_out_band, "block"])
        for block in target:
            assignments[int(block)] = "test"
        block_band = block_band.loc[~block_band.index.isin(target)]
    for band in sorted(block_band.unique()):
        blocks = rng.permutation(block_band.index[block_band == band].to_numpy())
        if len(blocks) < 5:
            raise ValueError(
                "Too few blocks to form fit/reference/calibration/test splits."
            )
        n_ref = max(1, int(len(blocks) * 0.2))
        n_cal = max(1, int(len(blocks) * 0.2))
        n_test = 0 if held_out_band is not None else max(1, int(len(blocks) * 0.2))
        for i, block in enumerate(blocks):
            role = (
                "reference"
                if i < n_ref
                else "calibration"
                if i < n_ref + n_cal
                else "test"
                if i < n_ref + n_cal + n_test
                else "fit"
            )
            assignments[int(block)] = role
    roles = meta.block.map(assignments)
    parts = {
        role: batch.take(np.flatnonzero(roles.to_numpy() == role))
        for role in ("fit", "reference", "calibration", "test")
    }
    if held_out_band is not None:
        target_test = parts["test"]
        parts["test"] = target_test.take(
            np.flatnonzero(
                target_test.meta.temperature_band.to_numpy() == held_out_band
            )
        )
    if any(not len(p.X) for p in parts.values()):
        raise ValueError("An empty split was produced.")
    return parts, assignments
