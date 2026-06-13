"""exp025: 1D-CNN regressor on tabular features — a Kaggle TPS diversity booster.

A 1D convolutional net treats the (standardized) feature vector as a length-N signal:
a stem Linear lifts each feature to a small embedding, then Conv1d blocks mix local
groups of features, global-pool, and a linear head regresses the target. It is a
genuinely different model FAMILY from GBMs and the MLP/DAE, so even at GBM-or-worse
accuracy it decorrelates and can add a little to the blend (cf. exp024 DAE: -0.02).

Honest scope: signal is exhausted (residual-GBM ~0), so this is a DIVERSITY play only,
expected ~ -0.02 MSE on the blend. Fold-safe; saves OOF/test artifacts like any experiment.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.cv import rmse
from src.data import N_FOLDS, TEXT_COL  # noqa: F401
from src.features.tabular import build_tabular
from src.utils import SEED, clip_preds, get_logger

log = get_logger()


def run_cnn1d_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    years: np.ndarray,
    cfg: dict,
    device: str,
) -> tuple[dict, int, list[str]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    p = cfg.get("cnn", {})
    emb = int(p.get("embed_dim", 8))
    channels = list(p.get("channels", [32, 64]))
    epochs = int(p.get("epochs", 60))
    lr = float(p.get("lr", 1e-3))
    dropout = float(p.get("dropout", 0.1))
    batch_size = int(p.get("batch_size", 256))
    n_seeds = int(cfg.get("n_seeds", 1))
    seed = int(p.get("seed", SEED))

    # standardized numeric features (NA already flagged as features → fill with median)
    Xtr_df, Xte_df, feats = build_tabular(train, test, fe=True)
    num = [c for c in feats if Xtr_df[c].dtype.kind in "ifu"]
    Xtr = Xtr_df[num].to_numpy(np.float32)
    Xte = Xte_df[num].to_numpy(np.float32)
    allX = np.vstack([Xtr, Xte])
    med = np.nanmedian(allX, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    allX = np.where(np.isnan(allX), med, allX)   # fill BEFORE computing stats (else NaN columns poison mu/sd)
    mu, sd = allX.mean(0), allX.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    n_feat = Xtr.shape[1]
    log.info(f"1D-CNN on {n_feat} numeric features")

    class CNN1D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(1, emb)            # each feature -> emb-dim vector
            convs, c_in = [], emb
            for c_out in channels:
                convs += [nn.Conv1d(c_in, c_out, kernel_size=3, padding=1),
                          nn.BatchNorm1d(c_out), nn.ReLU(), nn.Dropout(dropout)]
                c_in = c_out
            self.convs = nn.Sequential(*convs)
            self.head = nn.Linear(c_in, 1)

        def forward(self, x):                          # x: (B, n_feat)
            h = self.embed(x.unsqueeze(-1))            # (B, n_feat, emb)
            h = h.transpose(1, 2)                       # (B, emb, n_feat)
            h = self.convs(h)                           # (B, C, n_feat)
            h = h.mean(dim=2)                           # global average pool -> (B, C)
            return self.head(h).squeeze(-1)

    yt = y / 100.0
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_rmses = []
    t0 = time.time()
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    for fold in range(N_FOLDS):
        tr = np.where(folds != fold)[0]
        va = np.where(folds == fold)[0]
        va_acc = np.zeros(len(va))
        for s in range(n_seeds):
            torch.manual_seed(seed + fold * 10 + s)
            model = CNN1D().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            dl = DataLoader(
                TensorDataset(torch.tensor(Xtr[tr]), torch.tensor(yt[tr], dtype=torch.float32)),
                batch_size=batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(seed + fold * 10 + s),
            )
            for _ in range(epochs):
                model.train()
                for xb, yb in dl:
                    opt.zero_grad()
                    loss = nn.functional.mse_loss(model(xb.to(device)), yb.to(device))
                    loss.backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                va_acc += model(torch.tensor(Xtr[va]).to(device)).cpu().numpy() * 100 / n_seeds
                test_pred += model(Xte_t.to(device)).cpu().numpy() * 100 / (N_FOLDS * n_seeds)
        oof[va] = va_acc
        fold_rmses.append(rmse(y[va], clip_preds(oof[va])))
        log.info(f"  CNN1D fold {fold}: rmse={fold_rmses[-1]:.4f}")

    oof = clip_preds(oof)
    test_pred = clip_preds(test_pred)
    m2024 = years >= 2024
    result = {
        "oof": oof, "test_pred": test_pred,
        "cv_mse": float(np.mean((y - oof) ** 2)), "cv_rmse": rmse(y, oof),
        "cv_rmse_std": float(np.std(fold_rmses)), "fold_rmses": fold_rmses,
        "runtime_s": time.time() - t0, "importance": None,
        "rmse_year_2024plus": rmse(y[m2024], oof[m2024]),
        "rmse_y_lt_100": rmse(y[y < 100], oof[y < 100]),
    }
    return result, n_feat, [f"cnn1d emb={emb}", f"channels={channels}", f"seeds={n_seeds}"]
