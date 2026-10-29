"""Квадратичный индекс, канонические переменные и свёрточные признаки."""

from __future__ import annotations

import unittest

import numpy as np

from industrial_novelty.models.base import NoveltyDetector
from industrial_novelty.models.canonical import CanonicalVariateIndex
from industrial_novelty.models.fusion import TailSum
from industrial_novelty.models.kernels import ConvolutionalKernelIndex, KernelBank
from industrial_novelty.models.subspace import (
    WindowMomentIndex,
    WindowSubspaceIndex,
    ledoit_wolf,
)


class ConstantScore(NoveltyDetector):
    def __init__(self, value: float) -> None:
        self.value = value

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value, dtype=np.float64)


class QuadraticTests(unittest.TestCase):
    def test_ledoit_wolf_matches_the_published_example(self) -> None:
        covariance = np.array([[0.4, 0.2], [0.2, 0.8]])
        drawn = np.random.RandomState(0).multivariate_normal(
            mean=[0.0, 0.0], cov=covariance, size=50
        )
        shrunk, shrinkage = ledoit_wolf(drawn)
        self.assertAlmostEqual(shrinkage, 0.23, places=2)
        self.assertTrue(
            np.allclose(shrunk, np.array([[0.44, 0.16], [0.16, 0.80]]), atol=0.02)
        )

    def test_minor_axis_outranks_a_longer_step_along_the_correlation(self) -> None:
        rng = np.random.default_rng(0)
        latent = rng.normal(size=800)
        paired = latent + rng.normal(scale=0.01, size=800)
        windows = np.stack([latent, paired], axis=1).astype(np.float32)[:, None, :]
        model = WindowSubspaceIndex(variance=0.99, max_rank=1).fit(windows)
        along = np.array([[[4.0, 4.0]]], dtype=np.float32)
        across = np.array([[[0.3, -0.3]]], dtype=np.float32)
        self.assertLess(float(np.linalg.norm(across)), float(np.linalg.norm(along)))
        self.assertGreater(float(model.score(across)[0]), float(model.score(along)[0]))

    def test_empty_batch_keeps_the_score_length(self) -> None:
        windows = np.zeros((6, 4, 2), dtype=np.float32)
        windows[:, :, 1] = np.linspace(0.0, 1.0, 6)[:, None]
        model = WindowSubspaceIndex(max_rank=2).fit(windows)
        empty = np.zeros((0, 4, 2), dtype=np.float32)
        self.assertEqual(len(model.score(empty)), 0)
        self.assertEqual(model.rank_, 1)


class MomentTests(unittest.TestCase):
    def test_orthogonal_shift_outranks_a_longer_shift_along_the_correlation(
        self,
    ) -> None:
        rng = np.random.default_rng(0)
        latent = rng.normal(size=500)
        paired = np.stack([latent, latent + rng.normal(scale=0.01, size=500)], axis=1)
        windows = np.repeat(paired[:, None, :], 4, axis=1).astype(np.float32)
        model = WindowMomentIndex().fit(windows)
        along = np.full((1, 4, 2), 3.0, dtype=np.float32)
        across = np.zeros((1, 4, 2), dtype=np.float32)
        across[0, :, 0] = 0.35
        across[0, :, 1] = -0.35
        self.assertLess(float(np.linalg.norm(across)), float(np.linalg.norm(along)))
        self.assertGreater(float(model.score(across)[0]), float(model.score(along)[0]))

    def test_inflated_scale_scores_above_the_quiet_class(self) -> None:
        rng = np.random.default_rng(1)
        quiet = rng.normal(scale=0.1, size=(120, 12, 2)).astype(np.float32)
        model = WindowMomentIndex().fit(quiet)
        loud = rng.normal(scale=1.0, size=(40, 12, 2)).astype(np.float32)
        self.assertGreater(
            float(model.score(loud).mean()), float(model.score(quiet).mean())
        )


class CanonicalTests(unittest.TestCase):
    def test_broken_transition_scores_above_the_learned_one(self) -> None:
        rng = np.random.default_rng(1)
        past = rng.normal(size=(300, 4, 2))
        future = past + rng.normal(scale=0.05, size=past.shape)
        windows = np.concatenate([past, future], axis=1).astype(np.float32)
        model = CanonicalVariateIndex().fit(windows)
        probe = rng.normal(size=(60, 4, 2))
        normal = np.concatenate(
            [probe, probe + rng.normal(scale=0.05, size=probe.shape)], axis=1
        )
        broken = np.concatenate([probe, rng.normal(size=probe.shape)], axis=1)
        normal_score = float(model.score(normal.astype(np.float32)).mean())
        broken_score = float(model.score(broken.astype(np.float32)).mean())
        self.assertGreater(broken_score, normal_score)

    def test_compressed_blocks_stay_finite(self) -> None:
        rng = np.random.default_rng(2)
        windows = rng.normal(size=(80, 8, 3)).astype(np.float32)
        model = CanonicalVariateIndex(max_block_dim=2).fit(windows)
        scores = model.score(windows)
        self.assertEqual(len(scores), len(windows))
        self.assertTrue(np.isfinite(scores).all())
        self.assertIsNotNone(model.past_leftover_scale_)


class KernelTests(unittest.TestCase):
    def test_zero_sum_kernels_ignore_a_constant_shift(self) -> None:
        rng = np.random.default_rng(3)
        windows = rng.normal(size=(30, 16, 2)).astype(np.float32)
        bank = KernelBank(n_features=84, seed=0).fit(windows)
        shifted = windows + 4.0
        np.testing.assert_allclose(
            bank.transform(windows), bank.transform(shifted), atol=1e-5
        )

    def test_jagged_windows_score_above_the_smooth_class(self) -> None:
        rng = np.random.default_rng(4)
        time = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        smooth = np.sin(2.0 * np.pi * time).reshape(1, 16, 1)
        windows = smooth + rng.normal(scale=0.05, size=(80, 16, 1)).astype(np.float32)
        self.assertEqual(windows.shape, (80, 16, 1))
        model = ConvolutionalKernelIndex(n_features=84, seed=0).fit(windows)
        jagged = rng.normal(size=(40, 16, 1)).astype(np.float32)
        self.assertGreater(
            float(model.score(jagged).mean()),
            float(model.score(windows[:40]).mean()),
        )
        self.assertGreater(model.n_features_, 84)

    def test_short_window_is_rejected(self) -> None:
        windows = np.zeros((4, 8, 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            ConvolutionalKernelIndex(n_features=84).fit(windows)


class TailSumTests(unittest.TestCase):
    def test_each_member_contributes_one_at_its_own_level(self) -> None:
        windows = np.zeros((5, 4, 1), dtype=np.float32)
        model = TailSum((ConstantScore(2.0), ConstantScore(5.0))).fit(windows)
        np.testing.assert_allclose(model.score(windows), 2.0)

    def test_empty_member_list_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TailSum(())
