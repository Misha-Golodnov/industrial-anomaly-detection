"""Проверки причинности, калибровки траекторий и новых представлений."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import pandas as pd

from industrial_novelty import (
    KernelMMDIndex,
    TailMax,
    WindowBatch,
    WindowDependenceIndex,
    WindowMomentIndex,
    WindowSpectrumIndex,
    WithinRunEWMA,
)
from industrial_novelty.eval.calibration import run_maxima, threshold_at_run_risk
from industrial_novelty.models.neural import (
    NeuralTransformationIndex,
    deterministic_contrastive_loss,
)
from industrial_novelty.models.research import history_mean


def normal_meta(runs: int, length: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simulation_run": np.repeat(np.arange(runs), length),
            "sample_end": np.tile(np.arange(24, 24 + length * 8, 8), runs),
            "label": 0,
        }
    )


class RunCalibrationTests(unittest.TestCase):
    def test_finite_sample_rank_and_strict_tie_handling(self):
        meta = normal_meta(19, 2)
        scores = np.repeat(np.arange(19), 2)
        threshold = threshold_at_run_risk(scores, meta, alpha=0.1, horizon=32)
        self.assertGreater(threshold, 17)
        self.assertLess(threshold, 18)
        self.assertEqual(
            int((run_maxima(scores, meta, horizon=32) >= threshold).sum()), 1
        )
        tied = threshold_at_run_risk(np.ones(38), meta, alpha=0.1, horizon=32)
        self.assertGreater(tied, 1)

    def test_too_few_runs_cannot_support_small_risk(self):
        self.assertEqual(
            threshold_at_run_risk(
                np.ones(6), normal_meta(3, 2), alpha=0.01, horizon=32
            ),
            np.inf,
        )

    def test_longer_horizon_and_irregular_grid_are_rejected(self):
        meta = normal_meta(3, 3)
        with self.assertRaises(ValueError):
            run_maxima(np.ones(9), meta, horizon=960)
        with self.assertRaises(ValueError):
            run_maxima(np.ones(8), meta.drop(index=1), horizon=40)

    def test_fault_labels_cannot_calibrate_threshold(self):
        meta = normal_meta(3, 2)
        meta.loc[0, "label"] = 1
        with self.assertRaises(ValueError):
            threshold_at_run_risk(np.ones(6), meta, alpha=0.1, horizon=32)


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.train = self.rng.normal(size=(160, 24, 3))

    def test_dependence_detects_broken_links_with_same_marginals(self):
        train = self.train.copy()
        train[:, :, 1] = train[:, :, 0] + 0.05 * self.rng.normal(
            size=train[:, :, 0].shape
        )
        detector = WindowDependenceIndex(max_pairs=3).fit(train)
        normal = train[:30].copy()
        abnormal = normal.copy()
        abnormal[:, :, 1] *= -1
        self.assertGreater(
            np.median(detector.score(abnormal)), np.median(detector.score(normal)) * 3
        )

    def test_spectrum_changes_with_frequency_but_not_level(self):
        detector = WindowSpectrumIndex().fit(self.train)
        features = detector.features(self.train[:4])
        np.testing.assert_allclose(
            features, detector.features(self.train[:4] + 4), atol=1e-10
        )
        t = np.arange(24)
        low = np.tile(np.sin(2 * np.pi * t / 12)[None, :, None], (3, 1, 3))
        high = np.tile(np.sin(2 * np.pi * t / 3)[None, :, None], (3, 1, 3))
        self.assertGreater(
            np.linalg.norm(detector.features(low) - detector.features(high)), 1
        )

    def test_history_is_causal_order_invariant_and_resets(self):
        values = np.array([[1.0], [3.0], [100.0], [200.0]])
        runs = np.array([1, 1, 2, 2])
        ends = np.array([24, 32, 24, 32])
        expected = np.array([[1.0], [2.0], [100.0], [150.0]])
        np.testing.assert_allclose(history_mean(values, runs, ends, 2), expected)
        order = np.array([3, 0, 2, 1])
        np.testing.assert_allclose(
            history_mean(values[order], runs[order], ends[order], 2), expected[order]
        )
        altered = values.copy()
        altered[-1] = 999
        np.testing.assert_allclose(
            history_mean(altered, runs, ends, 2)[:-1], expected[:-1]
        )

    def test_mmd_score_is_causal_and_sensitive_to_persistent_shift(self):
        model = KernelMMDIndex(history=4, n_features=32).fit(self.train)
        test = self.rng.normal(size=(30, 24, 3))
        scores = model.score(test)
        np.testing.assert_allclose(model.score(test[:15]), scores[:15])
        altered = test.copy()
        altered[15:] += 4
        shifted = model.score(altered)
        np.testing.assert_allclose(shifted[:15], scores[:15])
        self.assertGreater(np.median(shifted[20:]), np.median(scores[20:]))

    def test_empty_input_and_shape_mismatch(self):
        for model in [WindowDependenceIndex(), WindowSpectrumIndex(), KernelMMDIndex()]:
            model.fit(self.train)
            self.assertEqual(model.score(self.train[:0]).shape, (0,))
            with self.assertRaises(ValueError):
                model.score(np.zeros((2, 24, 4)))

    def test_nested_models_preserve_mmd_run_boundaries(self):
        meta = normal_meta(2, 5)
        meta["fault_number"] = 0
        meta["sample_start"] = meta["sample_end"] - 23
        meta["post_shutdown"] = False
        meta["split"] = "normal_test"
        values = self.train[:10].copy()
        values[4] += 6
        names = ("a", "b", "c")
        batch = WindowBatch(values, meta, names, "normal_test")
        first = WindowBatch(values[:5], meta.iloc[:5], names, "normal_test")
        second = WindowBatch(values[5:], meta.iloc[5:], names, "normal_test")
        for model in [
            TailMax((KernelMMDIndex(n_features=16), WindowMomentIndex())),
            WithinRunEWMA(
                TailMax((KernelMMDIndex(n_features=16), WindowMomentIndex()))
            ),
        ]:
            model.fit(self.train)
            expected = np.r_[model.score_batch(first), model.score_batch(second)]
            np.testing.assert_allclose(model.score_batch(batch), expected, atol=1e-9)


@unittest.skipUnless(importlib.util.find_spec("torch"), "нет необязательного PyTorch")
class NeuralTests(unittest.TestCase):
    def test_dcl_matches_explicit_definition_and_has_gradients(self):
        import torch

        torch.set_num_threads(2)
        torch.manual_seed(2)
        z = torch.randn(5, 4, 7, dtype=torch.float64, requires_grad=True)
        actual = deterministic_contrastive_loss(z)
        unit = torch.nn.functional.normalize(z, dim=-1)
        sim = torch.exp(unit @ unit.transpose(1, 2) / 0.1)
        terms = []
        for anchor in range(1, 4):
            denominator = sum(
                sim[:, anchor, other] for other in range(4) if other != anchor
            )
            terms.append(torch.log(denominator / sim[:, anchor, 0]))
        expected = sum(terms) / (3 * np.log(3))
        torch.testing.assert_close(actual, expected)
        actual.mean().backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.norm()), 0)

    def test_training_is_repeatable_and_scoring_is_samplewise(self):
        import torch

        torch.set_num_threads(2)
        x = np.random.default_rng(0).normal(size=(32, 12, 2)).astype(np.float32)
        first = NeuralTransformationIndex(
            epochs=2, hidden=16, latent=8, batch_size=16
        ).fit(x)
        second = NeuralTransformationIndex(
            epochs=2, hidden=16, latent=8, batch_size=16
        ).fit(x)
        np.testing.assert_allclose(first.score(x), second.score(x), rtol=1e-6)
        np.testing.assert_allclose(first.score(x[:5]), first.score(x)[:5], rtol=1e-6)
        self.assertEqual(first.score(x[:0]).shape, (0,))
        self.assertTrue(np.isfinite(first.train_history_).all())


if __name__ == "__main__":
    unittest.main()
