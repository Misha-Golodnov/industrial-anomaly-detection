import unittest

import numpy as np
import pandas as pd

from industrial_novelty.data.lbnl import fold_roles, occupied_windows
from industrial_novelty.models.lbnl import ResidualScan


class LbnlProtocolTests(unittest.TestCase):
    def test_each_week_is_held_out_once(self):
        weeks = np.arange(53)
        test_count = np.zeros(len(weeks), dtype=int)
        for fold in range(4):
            roles = fold_roles(weeks, fold)
            np.testing.assert_array_equal(
                sum(mask.astype(int) for mask in roles.values()), 1
            )
            test_count += roles["test"]
        np.testing.assert_array_equal(test_count, 1)

    def test_windows_reset_on_gaps_shutdown_and_midnight(self):
        dates = pd.to_datetime(
            [
                "2018-01-01 23:30",
                "2018-01-01 23:45",
                "2018-01-02 00:00",
                "2018-01-02 00:15",
                "2018-01-02 00:30",
                "2018-01-02 01:00",
                "2018-01-02 01:15",
                "2018-01-02 01:30",
                "2018-01-02 01:45",
                "2018-01-02 02:00",
            ]
        )
        frame = pd.DataFrame(
            {
                "SYS_CTL": 1,
                "SF_SPD_DM": [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            },
            index=dates,
        )
        np.testing.assert_array_equal(occupied_windows(frame, 3), [2, 9])

    def test_scan_falls_back_when_regime_has_no_reference(self):
        rng = np.random.default_rng(13)
        reference = rng.normal(size=(200, 4, 2))
        modes = np.zeros(len(reference), dtype=int)
        scan = ResidualScan([0.2, 0.2]).fit(reference, modes)
        patches = np.stack([np.zeros((4, 2)), np.full((4, 2), 6)])
        scores = scan.score(patches, np.array([0, 2]))["scan"]
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[1], scores[0])
