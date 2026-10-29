"""Окна, масштаб и разметка без чтения RData."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from industrial_novelty.data.scaling import ChannelScaler
from industrial_novelty.data.schema import resolve_channels
from industrial_novelty.data.windows import windows_from_frame


def series_frame(
    *,
    run: int,
    fault: int,
    values: np.ndarray,
    pressure: np.ndarray | None = None,
    samples: np.ndarray | None = None,
) -> pd.DataFrame:
    values = np.asarray(values, dtype=np.float64)
    if samples is None:
        samples = np.arange(1, len(values) + 1)
    if pressure is None:
        pressure = np.full(len(values), 2700.0)
    return pd.DataFrame(
        {
            "faultNumber": np.full(len(values), fault, dtype=np.int32),
            "simulationRun": np.full(len(values), run, dtype=np.int32),
            "sample": np.asarray(samples, dtype=np.int32),
            "xmeas_1": values,
            "xmeas_7": np.asarray(pressure, dtype=np.float64),
        }
    )


class WindowTests(unittest.TestCase):
    def test_scaler_uses_only_training_rows(self) -> None:
        train = series_frame(run=1, fault=0, values=np.zeros(12))
        val = series_frame(run=2, fault=0, values=np.full(12, 100.0))
        scaler = ChannelScaler().fit(train, ("xmeas_1",))
        train_windows = windows_from_frame(
            train,
            channels=("xmeas_1",),
            window_size=4,
            stride=4,
            scaler=scaler,
            fault_onset_sample=None,
            shutdown_pressure=2990.0,
            split="train",
        )
        val_windows = windows_from_frame(
            val,
            channels=("xmeas_1",),
            window_size=4,
            stride=4,
            scaler=scaler,
            fault_onset_sample=None,
            shutdown_pressure=2990.0,
            split="val",
        )
        self.assertTrue(np.allclose(train_windows.X, 0.0))
        self.assertTrue(np.allclose(val_windows.X, 100.0))

    def test_straddling_windows_are_dropped_and_labels_follow_onset(self) -> None:
        frame = series_frame(run=1, fault=4, values=np.arange(180, dtype=np.float64))
        scaler = ChannelScaler().fit(frame, ("xmeas_1",))
        batch = windows_from_frame(
            frame,
            channels=("xmeas_1",),
            window_size=10,
            stride=1,
            scaler=scaler,
            fault_onset_sample=161,
            shutdown_pressure=2990.0,
            split="faulty_test",
        )
        starts = set(batch.meta["sample_start"].tolist())
        self.assertIn(151, starts)
        self.assertNotIn(152, starts)
        self.assertNotIn(160, starts)
        self.assertIn(161, starts)
        normal = batch.meta.loc[batch.meta["sample_start"] == 151].iloc[0]
        fault = batch.meta.loc[batch.meta["sample_start"] == 161].iloc[0]
        self.assertEqual(int(normal["sample_end"]), 160)
        self.assertEqual(int(normal["label"]), 0)
        self.assertEqual(int(fault["sample_end"]), 170)
        self.assertEqual(int(fault["label"]), 1)
        self.assertEqual(batch.X.shape[1:], (10, 1))

    def test_shutdown_flag_starts_at_the_trip_sample(self) -> None:
        pressure = np.full(30, 2700.0)
        pressure[19:] = 3000.0
        frame = series_frame(run=1, fault=6, values=np.zeros(30), pressure=pressure)
        scaler = ChannelScaler().fit(frame, ("xmeas_1",))
        batch = windows_from_frame(
            frame,
            channels=("xmeas_1",),
            window_size=5,
            stride=5,
            scaler=scaler,
            fault_onset_sample=None,
            shutdown_pressure=2990.0,
            split="faulty_test",
        )
        flags = dict(
            zip(batch.meta["sample_start"], batch.meta["post_shutdown"], strict=True)
        )
        self.assertEqual(flags[16], False)
        self.assertEqual(flags[21], True)
        self.assertEqual(flags[26], True)

    def test_gaps_in_sample_are_rejected(self) -> None:
        frame = series_frame(
            run=1,
            fault=0,
            values=np.zeros(3),
            samples=np.array([1, 2, 4]),
        )
        scaler = ChannelScaler().fit(frame, ("xmeas_1",))
        with self.assertRaises(ValueError):
            windows_from_frame(
                frame,
                channels=("xmeas_1",),
                window_size=2,
                stride=1,
                scaler=scaler,
                fault_onset_sample=None,
                shutdown_pressure=2990.0,
                split="train",
            )

    def test_channel_names(self) -> None:
        self.assertEqual(len(resolve_channels("xmeas")), 41)
        self.assertEqual(len(resolve_channels("xmv")), 11)
        self.assertEqual(len(resolve_channels("all")), 52)
        self.assertEqual(resolve_channels(("xmeas_9", "xmv_10")), ("xmeas_9", "xmv_10"))
        with self.assertRaises(ValueError):
            resolve_channels("xmeas_1")
        with self.assertRaises(ValueError):
            resolve_channels(())


if __name__ == "__main__":
    unittest.main()
