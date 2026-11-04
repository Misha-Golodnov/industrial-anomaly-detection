"""Leakage boundaries and mathematical invariants of the chiller experiment."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_novelty.data.ashrae import CHANNELS, chiller_windows, normal_split
from industrial_novelty.models.chiller import TorchMeanDetector, tangent_energy


class ChillerProtocolTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional torch dependency")
    def test_flow_log_determinant_matches_autograd(self):
        import torch

        torch.set_num_threads(2)
        X = np.random.default_rng(9).normal(size=(24, 12, 6)).astype(np.float32)
        model = TorchMeanDetector(epochs=2, seed=2).fit(X)
        layer = model.net_[0]
        response = torch.tensor([0.4, -0.3, 0.1, 0.9], requires_grad=True)
        context = torch.tensor([[0.1, 0.2]])
        _, reported = layer(response[None], context)
        jacobian = torch.autograd.functional.jacobian(
            lambda y: layer(y[None], context)[0][0], response
        )
        _, actual = torch.linalg.slogdet(jacobian)
        self.assertAlmostEqual(reported.item(), actual.item(), places=5)

    def test_projection_removes_only_nuisance_direction(self):
        residual = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        tangent = np.broadcast_to(np.array([[[1.0], [0.0], [0.0]]]), (2, 3, 1))
        np.testing.assert_allclose(tangent_energy(residual, tangent), [16.0, 25.0])
        np.testing.assert_allclose(
            tangent_energy(residual, tangent, parallel_weight=1), [25.0, 25.0]
        )
        # Duplicate nuisance columns must not subtract the same energy twice.
        np.testing.assert_allclose(
            tangent_energy(residual, np.repeat(tangent, 2, axis=2)), [16.0, 25.0]
        )

    def test_windows_never_bridge_invalid_rows_or_blocks_and_split_is_disjoint(self):
        n = 5400
        frame = pd.DataFrame(np.full((n, len(CHANNELS)), 50.0), columns=CHANNELS)
        frame["TWE_set"] = np.repeat([40, 45, 50], 1800)
        frame.loc[200, "TWCI"] = -99.4
        frame["Time"] = 0  # Rounded or duplicated time must never reorder rows.
        frame["Active Fault"] = np.arange(n)  # Must not be a model channel.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal.csv"
            frame.to_csv(path, index=False)
            batch = chiller_windows(path)
        self.assertEqual(batch.X.shape[2], len(CHANNELS))
        self.assertFalse(
            ((batch.meta.row_start <= 200) & (batch.meta.row_end >= 200)).any()
        )
        self.assertTrue(
            (batch.meta.row_start // 180 == batch.meta.row_end // 180).all()
        )
        for band in (None, 40):
            parts, _ = normal_split(batch, 4, held_out_band=band)
            roles = list(parts)
            for i, left in enumerate(roles):
                for right in roles[i + 1 :]:
                    self.assertTrue(
                        set(parts[left].meta.block).isdisjoint(parts[right].meta.block)
                    )
            if band:
                for role in ("fit", "reference", "calibration"):
                    self.assertFalse((parts[role].meta.temperature_band == band).any())


if __name__ == "__main__":
    unittest.main()
