"""Guard the predictor masking and score fusion used by the transfer study."""

import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from industrial_novelty.models.transfer import MAX_FUSIONS, SINGLES, TransferSuite


class TransferTests(unittest.TestCase):
    def test_prediction_never_sees_its_own_target_and_fusion_uses_reference(self):
        rng = np.random.default_rng(16)
        fit = rng.normal(size=(180, 4, 3))
        ref = rng.normal(size=(100, 4, 3))
        fit[:, :, 2] = ref[:, :, 2] = 1
        groups = np.arange(len(ref)) // 20
        ends = np.arange(len(ref))
        with threadpool_limits(limits=2):
            suite = TransferSuite(("x", "y", "constant")).fit(fit, ref, groups, ends)
            changed = ref.copy()
            changed[:, -1, 2] += 2
            errors = suite.virtual.residuals(changed[:, -1])
            matrix, names = suite.score(changed, groups, ends)
        self.assertTrue(
            all(name not in parents for name, parents in suite.virtual.graph().items())
        )
        np.testing.assert_array_equal(errors[:, 2], 2)
        by_name = dict(zip(names, matrix.T, strict=True))
        evidence = {
            name: np.log1p(np.maximum((by_name[name] - center) / scale, 0))
            for name, (center, scale) in suite.fusion_scales.items()
        }
        for name, members in MAX_FUSIONS.items():
            np.testing.assert_allclose(
                by_name[name], np.maximum.reduce([evidence[k] for k in members])
            )
        np.testing.assert_allclose(
            by_name["mean_all"], np.mean([evidence[k] for k in SINGLES], axis=0)
        )
