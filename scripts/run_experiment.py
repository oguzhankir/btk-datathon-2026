"""Run one experiment from a YAML config.

Usage:
    python scripts/run_experiment.py -c configs/exp004.yaml [--hpo-trials 200]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv import append_results_row, fold_rmses_json, run_cv  # noqa: E402
from src.data import TARGET, fold_array, load_raw  # noqa: E402
from src.features import build_features  # noqa: E402
from src.models import get_model  # noqa: E402
from src.utils import (  # noqa: E402
    ARTIFACTS,
    EXPERIMENTS_MD,
    detect_device,
    get_logger,
    save_json,
    seed_everything,
)

log = get_logger()


def year_ratio_weights(train_years: np.ndarray, test_years: np.ndarray) -> np.ndarray:
    """Importance weights = test/train year-frequency ratio, normalized to mean 1."""
    tr = np.bincount(train_years, minlength=train_years.max() + 1) / len(train_years)
    te = np.bincount(test_years, minlength=train_years.max() + 1) / len(test_years)
    w = np.where(tr[train_years] > 0, te[train_years] / np.maximum(tr[train_years], 1e-9), 1.0)
    return w / w.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--hpo-trials", type=int, default=None,
                    help="override config hpo.trials (0 disables HPO)")
    ap.add_argument("--hpo-timeout", type=int, default=None,
                    help="override config hpo.timeout (seconds)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    exp_id = cfg["exp_id"]
    seed_everything()
    device = detect_device()
    log.info(f"=== {exp_id}: {cfg['description']} (device={device}) ===")

    train, test, _ = load_raw()
    y = train[TARGET].to_numpy(dtype=float)
    folds = fold_array(train)
    years = train["application_year"].to_numpy()

    if cfg.get("model") == "bert":
        from src.bert_finetune import run_bert_experiment

        result, n_features, notes = run_bert_experiment(train, test, y, folds, years, cfg, device)
    elif cfg.get("model") == "bert_v2":
        from src.bert_finetune_v2 import run_bert_v2_experiment

        result, n_features, notes = run_bert_v2_experiment(train, test, y, folds, years, cfg, device)
    elif cfg.get("model") == "llm":
        from src.llm_finetune import run_llm_experiment

        result, n_features, notes = run_llm_experiment(train, test, y, folds, years, cfg, device)
    else:
        X_tr, X_te, notes = build_features(cfg.get("features", {}), train, test, y, folds)
        n_features = X_tr.shape[1]
        log.info(f"features: {n_features}")

        sample_weight = None
        if cfg.get("sample_weighting") == "year_ratio":
            sample_weight = year_ratio_weights(years, test["application_year"].to_numpy())

        params = cfg.get("model_params") or {}
        if cfg.get("model_params_file"):
            pf = Path(cfg["model_params_file"])
            if pf.exists():
                params = {**json.loads(pf.read_text()), **params}
            else:
                log.warning(f"model_params_file {pf} not found — using default params")
                notes.append(f"params file {pf.name} missing, used defaults")

        hpo_cfg = cfg.get("hpo") or {}
        hpo_trials = args.hpo_trials if args.hpo_trials is not None else int(hpo_cfg.get("trials", 0))
        hpo_timeout = args.hpo_timeout if args.hpo_timeout is not None else hpo_cfg.get("timeout")
        if hpo_trials > 0:
            from src.hpo import GenericOptunaSearch

            best = GenericOptunaSearch(
                cfg["model"], X_tr, y, folds,
                n_trials=hpo_trials, timeout=hpo_timeout,
                device=device, study_name=f"{cfg['model']}_{exp_id}",
            ).run()
            params = {**params, **best}
            notes.append(f"hpo: {hpo_trials} trials")

        if cfg.get("model") == "two_stage":
            from src.two_stage import run_two_stage

            result = run_two_stage(
                X_tr, y, folds, X_te,
                reg_params=params or None, device=device, years=years,
            )
            notes.append(f"best_combo={result['best_combo']}")
        else:
            model_name, n_seeds = cfg["model"], int(cfg.get("n_seeds", 1))
            result = run_cv(
                lambda: get_model(model_name, params or None, device),
                X_tr, y, folds, X_test=X_te,
                sample_weight=sample_weight, years=years, n_seeds=n_seeds,
            )
        save_json(params, ARTIFACTS / f"params_{exp_id}.json")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS / f"oof_{exp_id}.npy", result["oof"])
    np.save(ARTIFACTS / f"test_{exp_id}.npy", result["test_pred"])
    if result.get("importance") is not None:
        result["importance"].rename("importance").to_csv(ARTIFACTS / f"importance_{exp_id}.csv")

    append_results_row(
        {
            "exp_id": exp_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "description": cfg["description"],
            "n_features": n_features,
            "cv_mse": round(result["cv_mse"], 4),
            "cv_rmse": round(result["cv_rmse"], 4),
            "cv_rmse_std": round(result["cv_rmse_std"], 4),
            "fold_rmses": fold_rmses_json(result["fold_rmses"]),
            "rmse_year_2024plus": round(result.get("rmse_year_2024plus", float("nan")), 4),
            "rmse_y_lt_100": round(result["rmse_y_lt_100"], 4),
            "runtime_s": round(result["runtime_s"], 1),
            "device": device,
            "config_path": args.config,
            "notes": "; ".join(notes),
        }
    )
    with EXPERIMENTS_MD.open("a") as f:
        f.write(
            f"\n### {exp_id} — {cfg['description']}\n"
            f"- CV MSE **{result['cv_mse']:.4f}** | RMSE {result['cv_rmse']:.4f} "
            f"(±{result['cv_rmse_std']:.4f}) | 2024+ RMSE {result.get('rmse_year_2024plus', float('nan')):.4f} "
            f"| y<100 RMSE {result['rmse_y_lt_100']:.4f} | {n_features} features\n"
            + (f"- Notes: {'; '.join(notes)}\n" if notes else "")
        )
    log.info(f"{exp_id} complete: cv_mse={result['cv_mse']:.4f}")


if __name__ == "__main__":
    main()
