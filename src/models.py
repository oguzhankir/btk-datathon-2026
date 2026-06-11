"""Model registry: device-aware wrappers with a uniform fit/predict interface.

Every wrapper implements:
    fit(X_tr, y_tr, X_val, y_val, sample_weight=None)
    predict(X) -> np.ndarray
    feature_importance() -> pd.Series | None
Categorical columns must be pandas `category` dtype in X (tabular.py guarantees it).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import SEED


class LGBMWrapper:
    """LightGBM regressor (CPU — fast enough at 10k rows) with early stopping."""

    DEFAULTS: dict[str, Any] = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "random_state": SEED,
        "verbosity": -1,
    }

    def __init__(self, params: dict | None = None, device: str = "cpu") -> None:
        self.params = {**self.DEFAULTS, **(params or {})}
        self.model = None

    def fit(self, X_tr, y_tr, X_val, y_val, sample_weight=None) -> "LGBMWrapper":
        import lightgbm as lgb

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(
            X_tr,
            y_tr,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series | None:
        return pd.Series(self.model.feature_importances_, index=self.model.feature_name_)


class LGBMClassifierWrapper(LGBMWrapper):
    """LightGBM binary classifier (for the two-stage ceiling model)."""

    DEFAULTS: dict[str, Any] = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "random_state": SEED,
        "verbosity": -1,
    }

    def fit(self, X_tr, y_tr, X_val, y_val, sample_weight=None) -> "LGBMClassifierWrapper":
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(
            X_tr,
            y_tr,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class XGBWrapper:
    """XGBoost regressor, GPU when available (device='cuda')."""

    DEFAULTS: dict[str, Any] = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "early_stopping_rounds": 100,
        "enable_categorical": True,
        "tree_method": "hist",
        "verbosity": 0,
    }

    def __init__(self, params: dict | None = None, device: str = "cpu") -> None:
        self.params = {**self.DEFAULTS, **(params or {}), "device": device}
        self.model = None

    def fit(self, X_tr, y_tr, X_val, y_val, sample_weight=None) -> "XGBWrapper":
        from xgboost import XGBRegressor

        self.model = XGBRegressor(**self.params)
        self.model.fit(
            X_tr, y_tr, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False
        )
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series | None:
        return pd.Series(self.model.feature_importances_, index=self.model.feature_names_in_)


class CatBoostWrapper:
    """CatBoost regressor with native categorical handling, GPU-capable."""

    DEFAULTS: dict[str, Any] = {
        "iterations": 3000,
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": SEED,
        "early_stopping_rounds": 100,
        "verbose": 0,
        "allow_writing_files": False,
    }

    def __init__(self, params: dict | None = None, device: str = "cpu") -> None:
        self.params = {**self.DEFAULTS, **(params or {})}
        if device == "cuda":
            self.params.setdefault("task_type", "GPU")
        self.model = None

    @staticmethod
    def _cat_features(X: pd.DataFrame) -> list[str]:
        return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]

    @staticmethod
    def _prep(X: pd.DataFrame, cats: list[str]) -> pd.DataFrame:
        # CatBoost wants raw strings for categoricals, not pandas category codes.
        X = X.copy()
        for c in cats:
            X[c] = X[c].astype(str)
        return X

    def fit(self, X_tr, y_tr, X_val, y_val, sample_weight=None) -> "CatBoostWrapper":
        from catboost import CatBoostRegressor, Pool

        cats = self._cat_features(X_tr)
        self._cats = cats
        train_pool = Pool(self._prep(X_tr, cats), y_tr, cat_features=cats, weight=sample_weight)
        val_pool = Pool(self._prep(X_val, cats), y_val, cat_features=cats)
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(train_pool, eval_set=val_pool)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(self._prep(X, self._cats))

    def feature_importance(self) -> pd.Series | None:
        return pd.Series(self.model.get_feature_importance(), index=self.model.feature_names_)


class RidgeWrapper:
    """Ridge on standardized, median-imputed, one-hot-encoded features."""

    def __init__(self, params: dict | None = None, device: str = "cpu") -> None:
        self.params = {"alpha": 10.0, **(params or {})}
        self.pipe = None

    def fit(self, X_tr, y_tr, X_val, y_val, sample_weight=None) -> "RidgeWrapper":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        cats = [c for c in X_tr.columns if isinstance(X_tr[c].dtype, pd.CategoricalDtype)]
        nums = [c for c in X_tr.columns if c not in cats]
        ct = ColumnTransformer(
            [
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
            ]
        )
        self.pipe = Pipeline([("ct", ct), ("ridge", Ridge(**self.params, random_state=SEED))])
        self.pipe.fit(X_tr, y_tr, ridge__sample_weight=sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        return self.pipe.predict(X)

    def feature_importance(self) -> pd.Series | None:
        return None


class MLPWrapper:
    """Torch MLP on standardized/imputed numerics + one-hot cats. Diversity model."""

    DEFAULTS: dict[str, Any] = {
        "hidden": [256, 128],
        "dropout": 0.25,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "max_epochs": 200,
        "patience": 15,
    }

    def __init__(self, params: dict | None = None, device: str = "cpu") -> None:
        params = dict(params or {})
        # accept HPO-style {width, depth} in place of an explicit hidden list
        if "width" in params or "depth" in params:
            params["hidden"] = [params.pop("width", 256)] * params.pop("depth", 2)
        self.params = {**self.DEFAULTS, **params}
        self.device = device

    def _make_xform(self, X: pd.DataFrame):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        nums = [c for c in X.columns if c not in cats]
        return ColumnTransformer(
            [
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), nums),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
            ]
        )

    def fit(self, X_tr, y_tr, X_val, y_val, sample_weight=None) -> "MLPWrapper":
        import torch
        from torch import nn

        torch.manual_seed(SEED)
        p = self.params
        self.xform = self._make_xform(X_tr)
        Xt = torch.tensor(self.xform.fit_transform(X_tr), dtype=torch.float32)
        Xv = torch.tensor(self.xform.transform(X_val), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y_tr, dtype=np.float32)).view(-1, 1)
        yv = torch.tensor(np.asarray(y_val, dtype=np.float32)).view(-1, 1)
        w = torch.tensor(
            np.asarray(sample_weight if sample_weight is not None else np.ones(len(Xt)), dtype=np.float32)
        ).view(-1, 1)

        layers: list[nn.Module] = []
        d = Xt.shape[1]
        for h in p["hidden"]:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p["dropout"])]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers).to(self.device)

        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        ds = torch.utils.data.TensorDataset(Xt, yt, w)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=p["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(SEED)
        )
        best, best_state, bad = np.inf, None, 0
        Xv_d, yv_d = Xv.to(self.device), yv.to(self.device)
        for _ in range(p["max_epochs"]):
            self.net.train()
            for xb, yb, wb in dl:
                xb, yb, wb = xb.to(self.device), yb.to(self.device), wb.to(self.device)
                opt.zero_grad()
                loss = (wb * (self.net(xb) - yb) ** 2).mean()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val_mse = float(((self.net(Xv_d) - yv_d) ** 2).mean())
            if val_mse < best - 1e-4:
                best, bad = val_mse, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= p["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X) -> np.ndarray:
        import torch

        self.net.eval()
        Xt = torch.tensor(self.xform.transform(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.net(Xt).cpu().numpy().ravel()

    def feature_importance(self) -> pd.Series | None:
        return None


REGISTRY: dict[str, type] = {
    "lgbm": LGBMWrapper,
    "lgbm_clf": LGBMClassifierWrapper,
    "xgb": XGBWrapper,
    "catboost": CatBoostWrapper,
    "ridge": RidgeWrapper,
    "mlp": MLPWrapper,
}


def get_model(name: str, params: dict | None = None, device: str = "cpu"):
    """Instantiate a registered model wrapper by name."""
    return REGISTRY[name](params=params, device=device)
