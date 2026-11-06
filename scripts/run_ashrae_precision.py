"""Adaptive follow-up after the physical/spectral round; separate manifest.

Reuse its fixed splits and scoring, preserving the original runner and results.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import run_ashrae_directions as runner

from industrial_novelty.models.chiller_directions import FeatureGaussian
from industrial_novelty.models.chiller_precision import ReferencePrecision

DIAGNOSTICS = []


class LoggedPrecision(ReferencePrecision):
    def fit(self, X, reference):
        super().fit(X, reference)
        DIAGNOSTICS.append(self.diagnostics_)
        return self


def factories():
    return {
        "raw_lw": lambda: FeatureGaussian("raw"),
        "raw_oas": lambda: FeatureGaussian("raw", covariance="oas"),
        "raw_cov_cv": lambda: LoggedPrecision("raw"),
        "invariant_cov_cv": lambda: LoggedPrecision("invariant"),
        "thermo_cov_cv": lambda: LoggedPrecision("thermo"),
        "raw_graphical_cv": lambda: LoggedPrecision("raw", policy="graphical"),
        "invariant_graphical_cv": lambda: LoggedPrecision(
            "invariant", policy="graphical"
        ),
        "thermo_graphical_cv": lambda: LoggedPrecision("thermo", policy="graphical"),
        "raw_cov_1e-4": lambda: LoggedPrecision("raw", fixed=1e-4),
    }


def main():
    if "--output" not in sys.argv:
        sys.argv += ["--output", "results/ashrae_directions_2026-11-04/precision"]
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    files = [
        Path(__file__),
        Path("src/industrial_novelty/models/chiller_precision.py"),
        Path(runner.__file__),
    ]
    manifest = {
        "stage": "adaptive follow-up after inspecting physical/spectral development results",
        "selection": "covariance regularization selected by normal reference likelihood only",
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "shrink_grid": [1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.3, 0.7, 1],
        "graphical_grid": [0.001, 0.01, 0.05, 0.1, 0.3, 1],
    }
    (output / "extension_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    runner.factories = factories
    runner.main()
    timing = pd.read_csv(output / "timing.csv")
    selected = timing[~timing.method.isin(["raw_lw", "raw_oas"])].reset_index(drop=True)
    if len(selected) != len(DIAGNOSTICS):
        raise ValueError("Diagnostics do not match fits.")
    rows = []
    for row, diagnostics in zip(selected.to_dict("records"), DIAGNOSTICS, strict=True):
        rows.append(
            {
                **row,
                **{k: v for k, v in diagnostics.items() if k != "trials"},
                "trials": json.dumps(diagnostics["trials"]),
            }
        )
    pd.DataFrame(rows).to_csv(output / "selection.csv", index=False)


if __name__ == "__main__":
    main()
