"""Adaptive mixed-unit scaling follow-up, with its own pre-run manifest."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_ashrae_directions as runner

from industrial_novelty.models.chiller_directions import FeatureGaussian
from industrial_novelty.models.chiller_scaling import UnitScaledPhysics


def factories():
    return {
        "raw_lw": lambda: FeatureGaussian("raw"),
        "thermo_lw": lambda: FeatureGaussian("thermo"),
        "thermo_unit_lw": lambda: UnitScaledPhysics(),
        "thermo_unit_oas": lambda: UnitScaledPhysics(covariance="oas"),
        "thermo_unit_cv": lambda: UnitScaledPhysics(covariance="cv"),
        "thermo_eos_unit_lw": lambda: UnitScaledPhysics(eos=True),
    }


def main():
    if "--output" not in sys.argv:
        sys.argv += ["--output", "results/ashrae_directions_2026-11-04/scaling"]
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.mkdir(parents=True, exist_ok=True)
    files = [
        Path(__file__),
        Path("src/industrial_novelty/models/chiller_scaling.py"),
        Path(runner.__file__),
    ]
    manifest = {
        "stage": "adaptive scaling audit after physical and covariance development results",
        "selection": "fixed four new variants plus two repeated controls; no fault-based tuning",
        "scaling": "unit-variance nonconstant physical features; no common .05 floor",
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    (output / "extension_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    runner.factories = factories
    runner.main()


if __name__ == "__main__":
    main()
