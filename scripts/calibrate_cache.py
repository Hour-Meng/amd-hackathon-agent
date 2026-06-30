#!/usr/bin/env python3
"""Run cache threshold calibration and write sweep CSV + calibrate-cache.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from my_routing_agent.calibration.cache_calibrator import run_cache_calibration
from my_routing_agent.config import CacheConfig


def main() -> int:
    labels = ROOT / "labels.csv"
    if not labels.exists():
        import runpy

        runpy.run_path(str(ROOT / "scripts" / "generate_labels.py"))

    report = run_cache_calibration(
        labels_path=labels,
        cache_store_path=ROOT / "cache_store.json",
        calibration_path=ROOT / "calibration_data.json",
        sweep_csv_path=ROOT / "threshold_sweep.csv",
        report_json_path=ROOT / "calibrate-cache.json",
        config=CacheConfig(),
    )
    print(report["micro_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
