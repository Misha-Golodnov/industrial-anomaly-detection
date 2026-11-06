"""Strict reader and shared calendar folds for the LBNL single-duct AHU."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd

BASIC = (
    "CHWC_VLV_DM",
    "MA_TEMP",
    "OA_DMPR_DM",
    "OA_TEMP",
    "RA_DMPR_DM",
    "RA_TEMP",
    "RF_CS",
    "SA_TEMP",
    "SA_TEMPSPT",
    "SF_CS",
    "ZONE_TEMP_1",
    "ZONE_TEMP_2",
    "ZONE_TEMP_3",
    "ZONE_TEMP_4",
    "ZONE_TEMP_5",
)
FEEDBACK = ("CHWC_VLV", "OA_DMPR", "RA_DMPR")
STATUS = ("SYS_CTL", "SF_SPD_DM", "RF_SPD_DM")
CHANNELS = (*BASIC, *FEEDBACK, *STATUS)
INTERVAL = pd.Timedelta(minutes=15)
SCENARIOS = {"AHU_annual.csv": ("normal", 0)}
for family, prefix, levels in (
    ("sa_bias", "coi_bias", (-4, -2, 2, 4)),
    ("oa_bias", "oa_bias", (-4, -2, 2, 4)),
    ("coil_leak", "coi_leakage", ("010", "025", "040", "050")),
    ("coil_stuck", "coi_stuck", ("010", "025", "050", "075")),
    ("damper_stuck", "damper_stuck", ("010", "025", "075", "100")),
):
    for level in levels:
        suffix = "_short" if level == "100" else ""
        SCENARIOS[f"{prefix}_{level}_annual{suffix}.csv"] = (family, int(level))
NORMAL = "AHU_annual.csv"


def read_sdahu(archive: Path) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Return complete 15-minute means indexed by the end of each interval."""
    with ZipFile(archive) as source:
        members = {}
        for member in source.namelist():
            name = PurePosixPath(member).name
            if name.startswith("sa_bias_"):
                name = name.replace("sa_bias_", "coi_bias_", 1)
            if name not in SCENARIOS:
                continue
            if name in members:
                raise ValueError(f"Duplicate SDAHU scenario: {name}")
            members[name] = member
        if missing := SCENARIOS.keys() - members.keys():
            raise ValueError(f"Missing SDAHU scenarios: {sorted(missing)}")
        unique = {}
        manifest = []
        for name in SCENARIOS:
            member = members[name]
            info = source.getinfo(member)
            with source.open(info) as raw:
                digest = hashlib.file_digest(raw, "sha256").hexdigest()
            canonical = unique.setdefault(digest, name)
            manifest.append(
                {
                    "scenario": name,
                    "member": member,
                    "canonical": canonical,
                    "sha256_csv": digest,
                    "crc32": f"{info.CRC:08x}",
                    "family": SCENARIOS[name][0],
                    "filename_level": SCENARIOS[name][1],
                }
            )
            if SCENARIOS[canonical][0] != SCENARIOS[name][0]:
                raise ValueError(
                    f"Same CSV with different fault labels: {name}, {canonical}"
                )
        frames = {name: _aggregate(source, members[name]) for name in unique.values()}
        return frames, manifest


def _aggregate(source: ZipFile, member: str) -> pd.DataFrame:
    with source.open(member) as raw:
        frame = pd.read_csv(
            raw,
            usecols=["Datetime", *CHANNELS],
            dtype={column: "float32" for column in CHANNELS},
        )
    dates = pd.DatetimeIndex(pd.to_datetime(frame.pop("Datetime"), format="ISO8601"))
    if (
        dates.hasnans
        or not dates.is_monotonic_increasing
        or not dates.is_unique
        or not dates.equals(dates.floor("min"))
    ):
        raise ValueError(f"Invalid minute timestamps: {member}")
    frame.index = dates
    frame = frame.loc[:, list(CHANNELS)].replace([np.inf, -np.inf], np.nan)
    buckets = frame.resample(INTERVAL, label="right")
    return buckets.mean().loc[buckets.count().eq(15).all(axis=1)]


def week_index(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
    """Assign intervals by their start date, including those ending at midnight."""
    return np.asarray(((index - INTERVAL).normalize() - origin).days // 7)


def fold_roles(weeks: np.ndarray, fold: int) -> dict[str, np.ndarray]:
    """Four disjoint test folds with normal-only fit, reference and calibration."""
    if fold not in range(4):
        raise ValueError("Expected a fold from 0 to 3.")
    phase = (weeks - 2 * fold) % 8
    return {
        "test": phase < 2,
        "fit": (phase >= 2) & (phase < 6),
        "reference": phase == 6,
        "calibration": phase == 7,
    }


def occupied_windows(frame: pd.DataFrame, length: int) -> np.ndarray:
    """Positions of full causal windows in one day of scheduled fan operation."""
    if length < 1:
        raise ValueError("Window length must be positive.")
    if len(frame) < length:
        return np.empty(0, dtype=int)
    index = frame.index
    day = (index - INTERVAL).normalize()
    on = frame["SYS_CTL"].eq(1).to_numpy() & frame["SF_SPD_DM"].eq(1).to_numpy()
    continuous = np.r_[
        True, (index[1:] - index[:-1] == INTERVAL) & (day[1:] == day[:-1])
    ]
    valid = np.lib.stride_tricks.sliding_window_view(on, length).all(axis=1)
    if length > 1:
        valid &= np.lib.stride_tricks.sliding_window_view(
            continuous[1:], length - 1
        ).all(axis=1)
    return np.flatnonzero(valid) + length - 1
