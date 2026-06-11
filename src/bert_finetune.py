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
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    p = cfg.get("bert", {})
    epochs = int(p.get("epochs", 3))
    lr = float(p.get("lr", 2e-5))
    max_len = int(p.get("max_len", 128))
    batch_size = int(p.get("batch_size", 32))

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode(texts: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        enc = tok(
            texts.fillna("").tolist(),
            truncation=True, padding="max_length", max_length=max_len, return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    ids_tr_all, mask_tr_all = encode(train[TEXT_COL])
    ids_te, mask_te = encode(test[TEXT_COL])
    y_t = torch.tensor(y, dtype=torch.float32)

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_rmses: list[float] = []
    t0 = time.time()

    for fold in range(N_FOLDS):
        tr = np.where(folds != fold)[0]
        va = np.where(folds == fold)[0]
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=1, problem_type="regression"
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        dl = DataLoader(
            TensorDataset(ids_tr_all[tr], mask_tr_all[tr], y_t[tr]),
            batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
        )
        model.train()
        for _ in range(epochs):
            for ids, mask, yb in dl:
                ids, mask, yb = ids.to(device), mask.to(device), yb.to(device)
                opt.zero_grad()
                out = model(input_ids=ids, attention_mask=mask).logits.squeeze(-1)
                loss = torch.nn.functional.mse_loss(out, yb)
                loss.backward()
                opt.step()

        @torch.no_grad()
        def predict(ids_all: torch.Tensor, mask_all: torch.Tensor) -> np.ndarray:
            model.eval()
            preds = []
            for i in range(0, len(ids_all), 256):
                out = model(
                    input_ids=ids_all[i : i + 256].to(device),
                    attention_mask=mask_all[i : i + 256].to(device),
                ).logits.squeeze(-1)
                preds.append(out.cpu().numpy())
            return np.concatenate(preds)

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
    return result, max_len, [f"bert={MODEL_NAME}"]
