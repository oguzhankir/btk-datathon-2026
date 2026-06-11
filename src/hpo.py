"""Generic Optuna search over the fixed folds, resumable via sqlite storage."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.data import N_FOLDS
from src.models import get_model
from src.utils import ARTIFACTS, SEED, clip_preds, get_logger, save_json

log = get_logger()


def _lgbm_space(trial) -> dict[str, Any]:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "max_bin": trial.suggest_int("max_bin", 127, 511),
    }


def _xgb_space(trial) -> dict[str, Any]:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _catboost_space(trial) -> dict[str, Any]:
    return {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
    }


def _mlp_space(trial) -> dict[str, Any]:
    width = trial.suggest_categorical("width", [128, 256, 512])
    depth = trial.suggest_int("depth", 2, 3)
    return {
        "hidden": [width] * depth,
        "dropout": trial.suggest_float("dropout", 0.05, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    }


SEARCH_SPACES: dict[str, Callable] = {
    "lgbm": _lgbm_space,
    "xgb": _xgb_space,
    "catboost": _catboost_space,
    "mlp": _mlp_space,
}


class GenericOptunaSearch:
    """Optuna CV-MSE minimization on the fixed folds with median pruning."""

    def __init__(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: np.ndarray,
        folds: np.ndarray,
        n_trials: int = 100,
        timeout: int | None = None,
        device: str = "cpu",
        study_name: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.X, self.y, self.folds = X, np.asarray(y, dtype=float), folds
        self.n_trials, self.timeout, self.device = n_trials, timeout, device
        self.study_name = study_name or model_name

    def _objective(self, trial) -> float:
        params = SEARCH_SPACES[self.model_name](trial)
        fold_mses: list[float] = []
        for fold in range(N_FOLDS):
            tr, va = np.where(self.folds != fold)[0], np.where(self.folds == fold)[0]
            model = get_model(self.model_name, params, self.device)
            model.fit(self.X.iloc[tr], self.y[tr], self.X.iloc[va], self.y[va])
            pred = clip_preds(model.predict(self.X.iloc[va]))
            fold_mses.append(float(np.mean((self.y[va] - pred) ** 2)))
            trial.report(float(np.mean(fold_mses)), fold)
            if fold >= 2 and trial.should_prune():
                import optuna

                raise optuna.TrialPruned()
        return float(np.mean(fold_mses))

    def run(self) -> dict[str, Any]:
        """Run/resume the study; persist best params to artifacts/params_{name}.json."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{ARTIFACTS / f'optuna_{self.study_name}.db'}"
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
            load_if_exists=True,
        )
        study.optimize(self._objective, n_trials=self.n_trials, timeout=self.timeout)
        log.info(f"HPO best mse={study.best_value:.4f} params={study.best_params}")
        save_json(study.best_params, ARTIFACTS / f"params_{self.study_name}.json")
        return study.best_params
