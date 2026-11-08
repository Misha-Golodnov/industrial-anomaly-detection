"""Download and audit an author-hosted RP-1043 derivative, pinned to a commit.

Raw files stay in gitignored data/raw. This is not the original ASHRAE archive
and no rights to redistribute the original data are implied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

COMMIT = "467961ef98d2c13dbff6e47490f38be1934ceef9"
BASE = f"https://raw.githubusercontent.com/viettra-xai/S-DAGMM/{COMMIT}/datasets/chiller/complete_dataset"
FILES = [
    f"{fault}{level}.csv"
    for fault in ("normal", "cf", "eo", "fwc", "fwe", "nc", "rl", "ro")
    for level in (10, 20, 30, 40)
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/ashrae_public"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/ashrae_2026-11-04")
    )
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        path = args.data / name
        # Always retrieve the pinned version; verify it before replacing a cache.
        with urllib.request.urlopen(f"{BASE}/{name}", timeout=90) as response:
            content = response.read()
        if not content.decode("utf-8-sig").startswith("Time,"):
            raise ValueError(f"Unexpected CSV header: {name}")
        path.write_bytes(content)
        frame = pd.read_csv(path)
        values = frame.to_numpy(dtype=float)
        dt = frame.Time.diff().dropna()
        return {
            "file": name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "rows": len(frame),
            "columns": len(frame.columns),
            "nonfinite": int((~np.isfinite(values)).sum()),
            "duplicate_time": int(frame.Time.duplicated().sum()),
            "nonincreasing_time_steps": int((dt <= 0).sum()),
            "negative_99_values": int((values == -99.4).sum()),
            "tiny_nonzero_values": int(
                ((np.abs(values) > 0) & (np.abs(values) < 1e-30)).sum()
            ),
            "duplicate_sensor_rows": int(frame.drop(columns="Time").duplicated().sum()),
            "source": f"{BASE}/{name}",
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch, FILES))
    pd.DataFrame(records).to_csv(args.output / "data_audit.csv", index=False)
    normal_hashes = {r["sha256"] for r in records if r["file"].startswith("normal")}
    manifest = {
        "source_repository": "https://github.com/viettra-xai/S-DAGMM",
        "source_commit": COMMIT,
        "access_date": "2026-11-04",
        "artifact_kind": "author-hosted CSV derivative, not original ASHRAE archive",
        "normal_files_byte_identical": len(normal_hashes) == 1,
        "normal_file_used": "normal10.csv",
        "time_warning": "Rounded timestamps have duplicates; preserve row order. No physical detection-delay claims.",
        "independence_warning": "One normal trajectory, not independent replicated runs. Block splits are exploratory.",
        "redistribution_warning": "Do not include raw CSVs in git or assume the code license covers ASHRAE data.",
        "files": records,
    }
    (args.output / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "files": len(records),
                "normal_byte_identical": len(normal_hashes) == 1,
                "rows": sorted({r["rows"] for r in records}),
                "duplicate_time_range": [
                    min(r["duplicate_time"] for r in records),
                    max(r["duplicate_time"] for r in records),
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
