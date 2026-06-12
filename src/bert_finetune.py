"""exp009: fine-tune dbmdz/bert-base-turkish-cased with a regression head.

Per-fold training (MSE loss, 3 epochs, lr 2e-5, max_len 128) → OOF + test
predictions, saved like any experiment so exp010 can use them as a feature.
Skips gracefully on CPU (training would take hours there).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from src.cv import rmse
from src.data import N_FOLDS, TEXT_COL
from src.utils import SEED, clip_preds, get_logger

log = get_logger()

MODEL_NAME = "dbmdz/bert-base-turkish-cased"


def run_bert_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    years: np.ndarray,
    cfg: dict,
    device: str,
) -> tuple[dict, int, list[str]]:
    """Train BERT per fold; returns (cv-result dict, n_features, notes)."""
    if device != "cuda" and not cfg.get("allow_cpu", False):
        log.error(
            "exp009 needs a GPU (BERT fine-tune is ~minutes/fold on GPU, hours on CPU). "
            "Run on a CUDA machine, or set allow_cpu: true in the config to force it."
        )
        sys.exit(1)

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModel, AutoTokenizer

    p = cfg.get("bert", {})
    model_name = p.get("model", MODEL_NAME)
    epochs = int(p.get("epochs", 3))
    lr = float(p.get("lr", 2e-5))
    max_len = int(p.get("max_len", 128))
    batch_size = int(p.get("batch_size", 32))
    max_grad_norm = float(p.get("max_grad_norm", 1.0))
    inner_val_frac = float(p.get("inner_val_frac", 0.1))  # held-out slice for best-epoch selection
    seed = int(p.get("seed", SEED))  # vary per config for multi-seed averaging in the blend

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)

    class _Regressor(torch.nn.Module):
        """Encoder + mean-pool + linear head. Works for any HF encoder model."""
        def __init__(self, name: str) -> None:
            super().__init__()
            self.enc = AutoModel.from_pretrained(name)
            h = self.enc.config.hidden_size
            self.head = torch.nn.Linear(h, 1)
            torch.nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(self.head.bias)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
            m = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return self.head(pooled).squeeze(-1)

    def encode(texts: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        enc = tok(
            texts.fillna("").tolist(),
            truncation=True, padding="max_length", max_length=max_len, return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    ids_tr_all, mask_tr_all = encode(train[TEXT_COL])
    ids_te, mask_te = encode(test[TEXT_COL])
    # train on y/100: a randomly-initialized head can't climb to the 0-100 scale
    # in 3 epochs at lr 2e-5 (verified: raw-scale training collapses to RMSE ~55)
    y_t = torch.tensor(y / 100.0, dtype=torch.float32)

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_rmses: list[float] = []
    t0 = time.time()

    rng = np.random.default_rng(seed)
    for fold in range(N_FOLDS):
        tr_all = np.where(folds != fold)[0]
        va = np.where(folds == fold)[0]
        # inner split of the training fold for best-epoch selection (no val-fold leakage)
        perm = rng.permutation(tr_all)
        n_inner = max(1, int(inner_val_frac * len(perm)))
        iv, tr = perm[:n_inner], perm[n_inner:]

        model = _Regressor(model_name).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        dl = DataLoader(
            TensorDataset(ids_tr_all[tr], mask_tr_all[tr], y_t[tr]),
            batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        total_steps = epochs * len(dl)
        warmup = max(1, int(0.1 * total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: s / warmup if s < warmup else max(0.0, (total_steps - s) / (total_steps - warmup)),
        )

        @torch.no_grad()
        def predict(ids_all: torch.Tensor, mask_all: torch.Tensor) -> np.ndarray:
            model.eval()
            preds = []
            for i in range(0, len(ids_all), 256):
                out = model(
                    input_ids=ids_all[i : i + 256].to(device),
                    attention_mask=mask_all[i : i + 256].to(device),
                )
                preds.append(out.cpu().numpy())
            return np.concatenate(preds) * 100.0  # back to target scale

        best_rmse, best_state = np.inf, None
        for ep in range(epochs):
            model.train()
            for ids, mask, yb in dl:
                ids, mask, yb = ids.to(device), mask.to(device), yb.to(device)
                opt.zero_grad()
                out = model(input_ids=ids, attention_mask=mask)
                loss = torch.nn.functional.mse_loss(out, yb)
                if torch.isnan(loss):
                    opt.zero_grad()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                sched.step()
            iv_rmse = rmse(y[iv], clip_preds(predict(ids_tr_all[iv], mask_tr_all[iv])))
            if iv_rmse < best_rmse - 1e-3:
                best_rmse = iv_rmse
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log.info(f"  bert fold {fold} epoch {ep}: inner_val_rmse={iv_rmse:.4f}")
        if best_state is not None:
            model.load_state_dict(best_state)

        oof[va] = predict(ids_tr_all[va], mask_tr_all[va])
        test_pred += predict(ids_te, mask_te) / N_FOLDS
        fr = rmse(y[va], clip_preds(oof[va]))
        fold_rmses.append(fr)
        log.info(f"  bert fold {fold}: rmse={fr:.4f}")
        del model
        torch.cuda.empty_cache()

    oof = clip_preds(oof)
    test_pred = clip_preds(test_pred)
    m2024 = years >= 2024
    result = {
        "oof": oof,
        "test_pred": test_pred,
        "cv_mse": float(np.mean((y - oof) ** 2)),
        "cv_rmse": rmse(y, oof),
        "cv_rmse_std": float(np.std(fold_rmses)),
        "fold_rmses": fold_rmses,
        "runtime_s": time.time() - t0,
        "importance": None,
        "rmse_year_2024plus": rmse(y[m2024], oof[m2024]),
        "rmse_y_lt_100": rmse(y[y < 100], oof[y < 100]),
    }
    return result, max_len, [f"bert={model_name}"]
