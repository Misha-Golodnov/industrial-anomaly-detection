"""Разбиение прогонов и дымовая проверка на скачанном TEP."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from industrial_novelty.data.experiment import (
    ExperimentConfig,
    build_experiment,
    split_run_ids,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


class SplitTests(unittest.TestCase):
    def test_split_is_disjoint_and_repeatable(self) -> None:
        runs = np.arange(1, 11)
        train, val = split_run_ids(runs, val_fraction=0.2, seed=0)
        again_train, again_val = split_run_ids(runs, val_fraction=0.2, seed=0)
        self.assertTrue(np.array_equal(train, again_train))
        self.assertTrue(np.array_equal(val, again_val))
        self.assertTrue(set(train).isdisjoint(val))
        self.assertEqual(set(train) | set(val), set(runs))
        self.assertEqual(len(val), 2)
        self.assertTrue(np.all(train[:-1] <= train[1:]))

    def test_zero_validation_keeps_every_run_for_training(self) -> None:
        train, val = split_run_ids(np.array([3, 1, 2]), val_fraction=0, seed=1)
        self.assertEqual(train.tolist(), [1, 2, 3])
        self.assertEqual(len(val), 0)

    def test_config_rejects_a_single_column_name(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentConfig(channels="xmeas_1")


@unittest.skipUnless((RAW / "TEP_FaultFree_Training.RData").is_file(), "нет датасета")
class ExperimentSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.experiment = build_experiment(
            ExperimentConfig(
                data_dir=RAW,
                window_size=20,
                stride=20,
                channels="xmeas",
                val_fraction=0.5,
                max_runs=2,
                seed=0,
            )
        )

    def test_normal_splits_are_windows_of_fault_free_runs(self) -> None:
        experiment = self.experiment
        self.assertEqual(experiment.train.X.shape[1:], (20, 41))
        self.assertEqual(experiment.train.X.dtype, np.float32)
        self.assertIsNotNone(experiment.val)
        assert experiment.val is not None
        train_runs = set(experiment.train.meta["simulation_run"])
        val_runs = set(experiment.val.meta["simulation_run"])
        self.assertTrue(train_runs.isdisjoint(val_runs))
        self.assertTrue((experiment.train.meta["label"] == 0).all())
        self.assertTrue((experiment.val.meta["label"] == 0).all())
        self.assertTrue((experiment.normal_test.meta["label"] == 0).all())
        self.assertLessEqual(int(experiment.train.meta["sample_end"].max()), 500)
        self.assertLessEqual(int(experiment.normal_test.meta["sample_end"].max()), 960)
        self.assertFalse(bool(experiment.train.meta["post_shutdown"].any()))
        self.assertEqual(len(experiment.scaler.mean_), 41)


if __name__ == "__main__":
    unittest.main()
