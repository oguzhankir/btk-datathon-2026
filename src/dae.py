"""exp024: swap-noise Denoising AutoEncoder (DAE) features -> MLP regressor.

The canonical Kaggle tabular score-booster (Michael Jahrer, Porto Seguro #1): corrupt
each column with "swap noise" (replace a fraction of values with values resampled from
the same column), train an autoencoder to reconstruct the clean input, then use its
hidden activations as a new, learned representation. A downstream model on that
representation is a genuinely different model FAMILY from our GBMs, so even at similar
accuracy it decorrelates from them and can tighten the blend.

Honest scope: our residual-GBM tests show recoverable signal is ~0, so this will NOT
find new signal — its only value is blend DIVERSITY (expected ~ -0.2..-0.5 MSE), like
exp011 (the raw-embedding MLP, which earns ~0.10 weight). It is the one research-backed
tabular technique we hadn't tried.

The DAE is unsupervised (no target), trained on train+test combined → leakage-free.
The downstream MLP is fit fold-safe. Saves OOF/test artifacts like any experiment.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from src.cv import rmse
from src.data import N_FOLDS, TARGET, fold_array, load_raw
from src.features.tabular import build_tabular
from src.utils import SEED, clip_preds, get_logger

log = get_logger()


def run_dae_experiment(
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

    p = cfg.get("dae", {})
    swap_rate = float(p.get("swap_rate", 0.15))
    hidden = list(p.get("hidden", [512, 512, 512]))   # hidden activations are concatenated -> features
    dae_epochs = int(p.get("dae_epochs", 150))
    dae_lr = float(p.get("dae_lr", 3e-4))
    mlp_hidden = list(p.get("mlp_hidden", [256, 128]))
    mlp_epochs = int(p.get("mlp_epochs", 60))
    mlp_lr = float(p.get("mlp_lr", 1e-3))
    dropout = float(p.get("dropout", 0.1))
    batch_size = int(p.get("batch_size", 256))
    seed = int(p.get("seed", SEED))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # numeric feature matrix (engineered tabular); standardized, NA->0 (NA already flagged as features)
    Xtr_df, Xte_df, feats = build_tabular(train, test, fe=True)
    num = [c for c in feats if Xtr_df[c].dtype.kind in "ifu"]
    Xtr = Xtr_df[num].to_numpy(dtype=np.float32)
    Xte = Xte_df[num].to_numpy(dtype=np.float32)
    allX = np.vstack([Xtr, Xte])
    med = np.nanmedian(allX, axis=0)
    allX = np.where(np.isnan(allX), med, allX)
    mu, sd = allX.mean(0), allX.std(0) + 1e-6
    allX = (allX - mu) / sd
    n_in = allX.shape[1]
    log.info(f"DAE on {n_in} numeric features, {len(allX)} rows (train+test, unsupervised)")

    Xall = torch.tensor(allX, dtype=torch.float32)

    class DAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dims = [n_in] + hidden
            self.enc = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden)))
            self.dec = nn.Linear(sum(hidden), n_in)
            self.act = nn.ReLU()

        def hidden_rep(self, x):
            hs, h = [], x
            for layer in self.enc:
                h = self.act(layer(h))
                hs.append(h)
            return torch.cat(hs, dim=1)

        def forward(self, x):
            return self.dec(self.hidden_rep(x))

    def swap_noise(batch: torch.Tensor) -> torch.Tensor:
        """Replace swap_rate of entries with values resampled from the same column."""
        out = batch.clone()
        mask = torch.rand_like(batch) < swap_rate
        for j in range(batch.shape[1]):
            idx = torch.where(mask[:, j])[0]
            if len(idx):
                src = torch.randint(0, len(batch), (len(idx),))
                out[idx, j] = batch[src, j]
        return out

    # ---- 1) train DAE unsupervised on train+test ----
    dae = DAE().to(device)
    opt = torch.optim.Adam(dae.parameters(), lr=dae_lr)
    dl = DataLoader(TensorDataset(Xall), batch_size=batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    t0 = time.time()
    for ep in range(dae_epochs):
        dae.train()
        tot = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            recon = dae(swap_noise(xb))
            loss = nn.functional.mse_loss(recon, xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if ep % 30 == 0 or ep == dae_epochs - 1:
            log.info(f"  DAE epoch {ep}: recon_mse={tot / len(Xall):.4f}")

    dae.eval()
    with torch.no_grad():
        rep = torch.cat([dae.hidden_rep(Xall[i:i + 1024].to(device)).cpu()
                         for i in range(0, len(Xall), 1024)]).numpy()
    rep = (rep - rep.mean(0)) / (rep.std(0) + 1e-6)
    rep_tr, rep_te = rep[:len(Xtr)], rep[len(Xtr):]
    log.info(f"  DAE representation: {rep.shape[1]} features")

    # ---- 2) fold-safe MLP regressor on the DAE representation ----
    class MLP(nn.Module):
        def __init__(self, d) -> None:
            super().__init__()
            layers, prev = [], d
            for h in mlp_hidden:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    yt = y / 100.0
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_rmses = []
    rep_te_t = torch.tensor(rep_te, dtype=torch.float32)
    for fold in range(N_FOLDS):
        tr = np.where(folds != fold)[0]
        va = np.where(folds == fold)[0]
        torch.manual_seed(seed + fold)
        model = MLP(rep.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=mlp_lr, weight_decay=1e-4)
        Xt = torch.tensor(rep_tr[tr], dtype=torch.float32)
        yt_t = torch.tensor(yt[tr], dtype=torch.float32)
        dl = DataLoader(TensorDataset(Xt, yt_t), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed + fold))
        for ep in range(mlp_epochs):
            model.train()
            for xb, yb in dl:
                opt.zero_grad()
                loss = nn.functional.mse_loss(model(xb.to(device)), yb.to(device))
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            oof[va] = model(torch.tensor(rep_tr[va], dtype=torch.float32).to(device)).cpu().numpy() * 100
            test_pred += model(rep_te_t.to(device)).cpu().numpy() * 100 / N_FOLDS
        fold_rmses.append(rmse(y[va], clip_preds(oof[va])))
        log.info(f"  MLP fold {fold}: rmse={fold_rmses[-1]:.4f}")

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
    return result, rep.shape[1], [f"dae swap={swap_rate}", f"hidden={hidden}", f"rep={rep.shape[1]}"]
