"""The sequential statistics may use only observed history within a run."""

import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from industrial_novelty.models.innovation import InnovationScan, lagged, moving_mean


class InnovationTest(unittest.TestCase):
    def test_lags_are_run_local_and_end_before_target(self):
        runs = np.arange(12).reshape(2, 6, 1)
        actual = lagged(runs, order=2)
        np.testing.assert_array_equal(actual[0, 0], [1, 0])
        np.testing.assert_array_equal(actual[1, 0], [7, 6])
        np.testing.assert_array_equal(actual[1, -1], [10, 9])

    def test_moving_mean_is_trailing(self):
        x = np.array([[[2.0], [6.0], [10.0], [4.0]]])
        np.testing.assert_array_equal(moving_mean(x, 2).ravel(), [4, 8, 7])

    def test_future_perturbation_changes_no_past_score(self):
        rng = np.random.default_rng(7)
        fit = rng.normal(size=(12, 70, 4))
        reference = rng.normal(size=(10, 70, 4))
        test = rng.normal(size=(2, 70, 4))
        changed = test.copy()
        changed[0, 30:] += 5
        with threadpool_limits(limits=2):
            scan = InnovationScan(widths=(8, 16)).fit(fit, reference)
            a, b = scan.score(test), scan.score(changed)
            np.testing.assert_allclose(scan.score_mean(test), a["mean"])
            np.testing.assert_allclose(scan.score_mean(changed), b["mean"])
        for name in a:
            np.testing.assert_allclose(a[name][0, :30], b[name][0, :30])
            np.testing.assert_allclose(a[name][1], b[name][1])
        self.assertGreater(b["joint"][0, 30:].max(), a["joint"][0, 30:].max())
