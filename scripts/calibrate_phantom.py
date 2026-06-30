#!/usr/bin/env python3
"""Calibrate PHANTOM ensemble abort rule and write phantom_ensemble.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from my_routing_agent.calibration.phantom_calibrator import run_phantom_calibration
from my_routing_agent.config import load_config


def main() -> int:
    cfg = load_config()
    report = run_phantom_calibration(
        cfg.local.model,
        ROOT / "phantom_ensemble.json",
        n_good=2000,
        n_bad=2000,
    )
    print(report["production_rule"])
    print("ensemble_roc_auc=", report["ensemble_roc_auc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
