"""Build a submission file from saved test predictions.

Usage:
    python scripts/make_submission.py -e exp004
    python scripts/make_submission.py -e blend
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import ID_COL, TARGET, load_raw  # noqa: E402
from src.utils import ARTIFACTS, RESULTS_CSV, SUBMISSIONS, clip_preds  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--exp-id", required=True)
    args = ap.parse_args()
    exp = args.exp_id

    test_path = ARTIFACTS / f"test_{exp}.npy"
    if not test_path.exists():
        sys.exit(f"missing {test_path} — run the experiment first")
    preds = clip_preds(np.load(test_path))

    train, _, sub = load_raw()
    assert len(preds) == len(sub), f"pred length {len(preds)} != submission {len(sub)}"
    sub = sub[[ID_COL]].copy()
    sub[TARGET] = preds
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    out = SUBMISSIONS / f"sub_{exp}.csv"
    sub.to_csv(out, index=False)

    y = train[TARGET]
    print(f"wrote {out}")
    print(f"{'':14s}{'pred':>10s}{'train_y':>10s}")
    for name, fn in [("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]:
        print(f"{name:14s}{fn(preds):>10.3f}{fn(y):>10.3f}")
    print(f"{'% at 0':14s}{(preds == 0).mean() * 100:>9.2f}%{(y == 0).mean() * 100:>9.2f}%")
    print(f"{'% at 100':14s}{(preds == 100).mean() * 100:>9.2f}%{(y == 100).mean() * 100:>9.2f}%")

    if RESULTS_CSV.exists():
        res = pd.read_csv(RESULTS_CSV)
        row = res[res["exp_id"] == exp]
        if len(row):
            r = row.iloc[-1]
            print(f"\nCV: mse={r['cv_mse']} rmse={r['cv_rmse']} 2024+={r['rmse_year_2024plus']}")


if __name__ == "__main__":
    main()
