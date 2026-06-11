"""Run every config in configs/ sequentially, skipping completed exp_ids.

Usage:
    python scripts/run_all.py [--force]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def completed_exp_ids() -> set[str]:
    results = ROOT / "results.csv"
    if not results.exists():
        return set()
    return set(pd.read_csv(results)["exp_id"].astype(str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run completed experiments")
    args = ap.parse_args()

    done = set() if args.force else completed_exp_ids()
    failures: list[str] = []
    for cfg_path in sorted((ROOT / "configs").glob("*.yaml")):
        exp_id = yaml.safe_load(cfg_path.read_text())["exp_id"]
        if exp_id in done:
            print(f"skip {exp_id} (already in results.csv)")
            continue
        print(f"\n>>> running {exp_id} ({cfg_path.name})")
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "-c", str(cfg_path)]
        ).returncode
        if rc != 0:
            failures.append(exp_id)
            print(f"!!! {exp_id} failed (rc={rc}) — continuing")
    if failures:
        sys.exit(f"failed experiments: {failures}")
    print("\nall experiments done")


if __name__ == "__main__":
    main()
