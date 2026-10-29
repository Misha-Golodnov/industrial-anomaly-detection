"""Порог, задержка обнаружения и контракт модели."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_novelty import evaluate_model
from industrial_novelty.data.experiment import (
    Experiment,
    ExperimentConfig,
    FaultWindows,
)
from industrial_novelty.data.scaling import ChannelScaler
from industrial_novelty.data.windows import WindowBatch
from industrial_novelty.eval.detection import roc_auc, run_detections, threshold_at_fpr
from industrial_novelty.models.base import NoveltyDetector
from industrial_novelty.models.centroid import CentroidDistance


def batch_of(
    levels: list[float], rows: list[dict[str, object]], split: str
) -> WindowBatch:
    windows = np.asarray(levels, dtype=np.float32).reshape(len(levels), 1, 1)
    meta = pd.DataFrame(rows)
    return WindowBatch(windows, meta, ("xmeas_1",), split)


class ThresholdTests(unittest.TestCase):
    def test_budget_is_an_upper_bound(self) -> None:
        scores = np.arange(100, dtype=np.float64)
        threshold = threshold_at_fpr(scores, 0.1)
        self.assertEqual(threshold, 90.0)
        self.assertEqual(int(np.sum(scores >= threshold)), 10)

    def test_ties_do_not_exceed_the_budget(self) -> None:
        scores = np.array([0.0, 1.0, 1.0, 1.0])
        threshold = threshold_at_fpr(scores, 0.25)
        self.assertEqual(int(np.sum(scores >= threshold)), 0)

    def test_roc_auc_handles_order_and_ties(self) -> None:
        self.assertEqual(roc_auc(np.array([1.0, 1.0]), np.array([0.0, 0.0])), 1.0)
        self.assertEqual(roc_auc(np.array([1.0]), np.array([1.0])), 0.5)
        self.assertTrue(np.isnan(roc_auc(np.array([]), np.array([0.0]))))

    def test_zero_fpr_flags_nothing(self) -> None:
        scores = np.array([0.0, 2.0, 3.5])
        threshold = threshold_at_fpr(scores, 0.0)
        self.assertEqual(int(np.sum(scores >= threshold)), 0)


class DetectionTests(unittest.TestCase):
    def test_shutdown_plateau_and_pre_fault_alarm_do_not_count(self) -> None:
        batch = batch_of(
            [9.0, 3.0, 100.0],
            [
                {
                    "fault_number": 6,
                    "simulation_run": 1,
                    "sample_start": 150,
                    "sample_end": 150,
                    "label": 0,
                    "post_shutdown": False,
                    "split": "faulty_test",
                },
                {
                    "fault_number": 6,
                    "simulation_run": 1,
                    "sample_start": 161,
                    "sample_end": 161,
                    "label": 1,
                    "post_shutdown": False,
                    "split": "faulty_test",
                },
                {
                    "fault_number": 6,
                    "simulation_run": 1,
                    "sample_start": 200,
                    "sample_end": 200,
                    "label": 1,
                    "post_shutdown": True,
                    "split": "faulty_test",
                },
            ],
            "faulty_test",
        )
        table = run_detections(
            batch,
            [9.0, 3.0, 100.0],
            threshold=1.0,
            last_normal_sample=160,
            fault_number=6,
            simulation_runs=(1, 2),
        )
        self.assertEqual(len(table), 2)
        detected = table.loc[table["simulation_run"] == 1].iloc[0]
        missing = table.loc[table["simulation_run"] == 2].iloc[0]
        self.assertTrue(bool(detected["detected"]))
        self.assertEqual(int(detected["first_detection_sample"]), 161)
        self.assertEqual(float(detected["delay_samples"]), 1.0)
        self.assertEqual(float(detected["delay_minutes"]), 3.0)
        self.assertTrue(bool(detected["early"]))
        self.assertEqual(float(detected["alarm_persistence"]), 1.0)
        self.assertEqual(int(detected["shutdown_start_sample"]), 200)
        self.assertEqual(float(detected["shutdown_margin_minutes"]), 117.0)
        self.assertFalse(bool(missing["scoreable"]))
        self.assertFalse(bool(missing["detected"]))
        self.assertFalse(bool(missing["early"]))
        self.assertTrue(np.isnan(missing["alarm_persistence"]))

    def test_only_plateau_is_unscoreable(self) -> None:
        batch = batch_of(
            [100.0],
            [
                {
                    "fault_number": 6,
                    "simulation_run": 1,
                    "sample_start": 200,
                    "sample_end": 200,
                    "label": 1,
                    "post_shutdown": True,
                    "split": "faulty_test",
                }
            ],
            "faulty_test",
        )
        table = run_detections(
            batch,
            [100.0],
            threshold=1.0,
            last_normal_sample=160,
            fault_number=6,
            simulation_runs=(1,),
        )
        self.assertFalse(bool(table.iloc[0]["scoreable"]))
        self.assertFalse(bool(table.iloc[0]["detected"]))
        self.assertFalse(bool(table.iloc[0]["early"]))
        self.assertTrue(np.isnan(table.iloc[0]["shutdown_margin_minutes"]))
        self.assertEqual(int(table.iloc[0]["shutdown_start_sample"]), 200)

    def test_late_alarm_is_not_early_and_persistence_is_partial(self) -> None:
        batch = batch_of(
            [0.0, 5.0],
            [
                {
                    "fault_number": 4,
                    "simulation_run": 1,
                    "sample_start": 161,
                    "sample_end": 180,
                    "label": 1,
                    "post_shutdown": False,
                    "split": "faulty_test",
                },
                {
                    "fault_number": 4,
                    "simulation_run": 1,
                    "sample_start": 181,
                    "sample_end": 200,
                    "label": 1,
                    "post_shutdown": False,
                    "split": "faulty_test",
                },
            ],
            "faulty_test",
        )
        table = run_detections(
            batch,
            [0.0, 5.0],
            threshold=1.0,
            last_normal_sample=160,
            fault_number=4,
            simulation_runs=(1,),
        )
        row = table.iloc[0]
        self.assertTrue(bool(row["detected"]))
        self.assertFalse(bool(row["early"]))
        self.assertEqual(float(row["alarm_persistence"]), 0.5)
        self.assertEqual(int(row["first_detection_sample"]), 200)
        self.assertTrue(np.isnan(row["shutdown_margin_minutes"]))


class ModelContractTests(unittest.TestCase):
    def test_centroid_scores_grow_with_distance(self) -> None:
        normal = np.zeros((4, 2, 1), dtype=np.float32)
        model = CentroidDistance().fit(normal)
        far = np.full((1, 2, 1), 5.0, dtype=np.float32)
        near = np.full((1, 2, 1), 0.5, dtype=np.float32)
        self.assertGreater(float(model.score(far)[0]), float(model.score(near)[0]))

    def test_score_length_is_checked(self) -> None:
        class Broken(NoveltyDetector):
            def anomaly_score(self, X: np.ndarray) -> np.ndarray:
                return np.zeros(len(X) - 1)

        model = Broken()
        with self.assertRaises(ValueError):
            model.score(np.zeros((2, 1, 1)))

    def test_evaluate_model_separates_threshold_test_and_detection(self) -> None:
        class Level(NoveltyDetector):
            def anomaly_score(self, X: np.ndarray) -> np.ndarray:
                return X.reshape(len(X), -1).mean(axis=1)

        def zeros(n: int, split: str) -> WindowBatch:
            rows = [
                {
                    "fault_number": 0,
                    "simulation_run": 1,
                    "sample_start": index + 1,
                    "sample_end": index + 1,
                    "label": 0,
                    "post_shutdown": False,
                    "split": split,
                }
                for index in range(n)
            ]
            return batch_of([0.0] * n, rows, split)

        normal = zeros(10, "normal_test")
        spiked = batch_of(
            [5.0],
            [
                {
                    "fault_number": 0,
                    "simulation_run": 2,
                    "sample_start": 1,
                    "sample_end": 1,
                    "label": 0,
                    "post_shutdown": False,
                    "split": "normal_test",
                }
            ],
            "normal_test",
        )
        normal_test = WindowBatch(
            np.concatenate([normal.X, spiked.X]),
            pd.concat([normal.meta, spiked.meta], ignore_index=True),
            ("xmeas_1",),
            "normal_test",
        )
        fault_batch = batch_of(
            [9.0, 3.0, 100.0],
            [
                {
                    "fault_number": 4,
                    "simulation_run": 1,
                    "sample_start": 100,
                    "sample_end": 100,
                    "label": 0,
                    "post_shutdown": False,
                    "split": "faulty_test",
                },
                {
                    "fault_number": 4,
                    "simulation_run": 1,
                    "sample_start": 170,
                    "sample_end": 180,
                    "label": 1,
                    "post_shutdown": False,
                    "split": "faulty_test",
                },
                {
                    "fault_number": 4,
                    "simulation_run": 1,
                    "sample_start": 220,
                    "sample_end": 230,
                    "label": 1,
                    "post_shutdown": True,
                    "split": "faulty_test",
                },
            ],
            "faulty_test",
        )
        experiment = Experiment(
            config=ExperimentConfig(val_fraction=0.0),
            scaler=ChannelScaler(),
            train=zeros(4, "train"),
            val=zeros(8, "val"),
            normal_test=normal_test,
            faulty_testing_path=Path("unused.RData"),
        )
        experiment.iter_faulty_test = lambda fault_numbers=None: iter(  # type: ignore[method-assign]
            [
                FaultWindows(
                    fault_number=4,
                    batch=fault_batch,
                    simulation_runs=(1,),
                )
            ]
        )
        report = evaluate_model(
            Level().fit(experiment.train.X), experiment, target_fpr=0.0
        )
        self.assertEqual(report.normal_test_fpr, 1 / 11)
        self.assertEqual(report.false_alarm_run_rate, 0.5)
        self.assertEqual(report.n_normal_test_runs, 2)
        row = report.per_run.iloc[0]
        self.assertTrue(bool(row["detected"]))
        self.assertTrue(bool(row["early"]))
        self.assertEqual(int(row["first_detection_sample"]), 180)
        self.assertEqual(float(row["delay_samples"]), 20.0)
        self.assertEqual(float(row["shutdown_margin_minutes"]), 120.0)
        fault = report.per_fault.iloc[0]
        self.assertEqual(float(fault["detection_rate"]), 1.0)
        self.assertEqual(float(fault["early_detection_rate"]), 1.0)
        self.assertEqual(float(fault["median_alarm_persistence"]), 1.0)
        self.assertEqual(float(fault["median_shutdown_margin_minutes"]), 120.0)
        self.assertAlmostEqual(float(fault["auroc"]), 10 / 11)


if __name__ == "__main__":
    unittest.main()
