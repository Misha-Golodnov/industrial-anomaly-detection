"""Physical feature semantics and sequential leakage invariants."""

import unittest

import numpy as np
import pandas as pd

from industrial_novelty.data.ashrae import CHANNELS
from industrial_novelty.models.chiller_diagnosis import FaultSignatureClassifier
from industrial_novelty.models.chiller_directions import (
    PHYSICAL_NAMES,
    ConditionalResidual,
    FeatureGaussian,
    SequentialResidual,
    contiguous_groups,
    logarithmic_mean,
    physical_series,
)
from industrial_novelty.models.chiller_precision import ReferencePrecision
from industrial_novelty.models.chiller_scaling import UnitScaledPhysics


class DirectionsTest(unittest.TestCase):
    def test_physical_metric_is_invariant_to_power_unit_rescaling(self):
        rng = np.random.default_rng(91)
        X = rng.normal(50, 2, (80, 12, len(CHANNELS)))
        original = UnitScaledPhysics().fit(X[:50], X[50:65])
        changed = X.copy()
        changed[..., CHANNELS.index("kW")] *= 1000
        rescaled = UnitScaledPhysics().fit(changed[:50], changed[50:65])
        np.testing.assert_allclose(
            original.score(X[65:]), rescaled.score(changed[65:]), rtol=1e-6, atol=1e-6
        )

    def test_graphical_precision_handles_constant_fit_channels(self):
        rng = np.random.default_rng(31)
        X = rng.normal(size=(60, 12, len(CHANNELS)))
        X[..., -2:] = 0
        model = ReferencePrecision("raw", policy="graphical").fit(X[:40], X[40:50])
        test = X[50:].copy()
        test[..., -1] = 1
        self.assertTrue(np.isfinite(model.score(test)).all())
        self.assertTrue(np.all(model.score(test) >= 0))
        self.assertGreater(np.linalg.eigvalsh(model.precision_).min(), 0)

    def test_severity_ray_profiles_amplitude_and_preserves_fault_direction(self):
        class Identity:
            def residual(self, X):
                return X[:, 0, :]

        model = FaultSignatureClassifier()
        model.normal_ = Identity()
        model.prototypes_ = list(np.eye(7))
        model.threshold_ = 1
        X = np.zeros((3, 1, 7))
        X[0, 0, 2], X[1, 0, 2], X[2, 0, 2] = 2, 4, -4
        np.testing.assert_allclose(model.evidence(X)[:, 2], [4, 16, 0])
        np.testing.assert_array_equal(model.predict(X), [3, 3, 0])

    def test_log_mean_limit_symmetry_and_domain(self):
        result, invalid = logarithmic_mean([4, 4, 8, -1], [4, 4 + 1e-9, 2, 3])
        np.testing.assert_allclose(result[:2], [4, 4], atol=1e-8)
        self.assertAlmostEqual(result[2], 6 / np.log(4))
        self.assertTrue(invalid[3])
        self.assertEqual(result[3], 0)
        reverse, _ = logarithmic_mean([2], [8])
        self.assertAlmostEqual(result[2], reverse[0])

    def test_physical_translation_invariance_and_subcooling_semantics(self):
        rng = np.random.default_rng(0)
        X = rng.normal(50, 2, (8, 12, len(CHANNELS)))
        index = {c: i for i, c in enumerate(CHANNELS)}
        X[..., index["TRE"]] = 35
        X[..., index["TRC"]] = 85
        X[..., index["TRC_sub"]] = 6
        physical = physical_series(X)
        np.testing.assert_allclose(physical[..., PHYSICAL_NAMES.index("subcooling")], 6)
        shifted = X.copy()
        absolute = [
            "TWE_set",
            "TWCI",
            "TEI",
            "TEO",
            "TWEI",
            "TWEO",
            "TCI",
            "TCO",
            "TWCO",
            "TRE",
            "TRC",
            "T_suc",
            "TR_dis",
            "TO_sump",
            "TO_feed",
        ]
        shifted[..., [index[c] for c in absolute]] += 20
        np.testing.assert_allclose(physical_series(shifted), physical, atol=1e-8)

    def test_mewma_resets_at_blocks_and_gaps_and_has_no_future_dependence(self):
        class Residual:
            def residual(self, X):
                return X[:, 0, :]

        model = SequentialResidual(Residual())
        X = np.ones((5, 1, 2))
        meta = pd.DataFrame({"block": [0, 0, 0, 1, 1], "row_end": [12, 18, 36, 42, 48]})
        score = model.score(X, meta)
        self.assertAlmostEqual(score[0], score[2])
        self.assertAlmostEqual(score[0], score[3])
        changed = X.copy()
        changed[1:] *= 100
        self.assertEqual(score[0], model.score(changed, meta)[0])
        self.assertEqual([len(g) for g in contiguous_groups(meta)], [2, 1, 2])

    def test_normal_reference_and_models_return_finite_scores(self):
        rng = np.random.default_rng(8)
        X = rng.normal(50, 2, (80, 12, len(CHANNELS)))
        for model in [
            FeatureGaussian("invariant"),
            ConditionalResidual("raw"),
            ConditionalResidual("raw", mean_model="gp"),
        ]:
            model.fit(X[:50], X[50:65])
            score = model.score(X[65:])
            self.assertEqual(score.shape, (15,))
            self.assertTrue(np.isfinite(score).all())


if __name__ == "__main__":
    unittest.main()
