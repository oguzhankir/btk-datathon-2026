"""Competition-grade transformer fine-tune recipe (exp020+).

Differences vs src/bert_finetune.py (the v1 recipe shared by exp009/012/014/015/018,
all of which plateaued at OOF RMSE ~11.5):
- attention pooling over tokens (learned) instead of plain mean pooling
- multi-sample dropout head (K masks averaged) — strong regularizer at 8k rows/fold
- layer-wise LR decay (LLRD): head learns fast, deep layers stay near pretrained
- cosine LR schedule with warmup instead of linear decay
- more epochs (default 8) with per-epoch best-state selection on an inner val slice

Same fold protocol, same y/100 normalization, same artifact format.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import pandas as pd

from src.cv import rmse
from src.data import N_FOLDS, TEXT_COL
from src.utils import SEED, clip_preds, get_logger

log = get_logger()

MODEL_NAME = "dbmdz/bert-base-turkish-cased"


def run_bert_v2_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    years: np.ndarray,
    cfg: dict,
    device: str,
) -> tuple[dict, int, list[str]]:
    """Train the v2 recipe per fold; returns (cv-result dict, n_features, notes)."""
    if device != "cuda" and not cfg.get("allow_cpu", False):
        log.error("bert_v2 needs a GPU; set allow_cpu: true to force CPU (hours).")
        sys.exit(1)

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModel, AutoTokenizer

    p = cfg.get("bert", {})
    model_name = p.get("model", MODEL_NAME)
    epochs = int(p.get("epochs", 8))
    lr = float(p.get("lr", 2e-5))               # LR of the TOP encoder layer
    head_lr = float(p.get("head_lr", 1e-3))     # pooling + regression head
    llrd = float(p.get("llrd", 0.9))            # per-layer decay going down
    wd = float(p.get("weight_decay", 0.01))
    max_len = int(p.get("max_len", 128))
    batch_size = int(p.get("batch_size", 32))
    max_grad_norm = float(p.get("max_grad_norm", 1.0))
    n_dropout = int(p.get("n_dropout", 5))      # multi-sample dropout masks
    p_dropout = float(p.get("p_dropout", 0.3))
    inner_val_frac = float(p.get("inner_val_frac", 0.1))
    seed = int(p.get("seed", SEED))

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)

    class Regressor(nn.Module):
        """Encoder + learned attention pooling + multi-sample-dropout linear head."""

        def __init__(self, name: str) -> None:
            super().__init__()
            self.enc = AutoModel.from_pretrained(name)
            h = self.enc.config.hidden_size
            self.attn = nn.Linear(h, 1)
            self.norm = nn.LayerNorm(h)
            self.drops = nn.ModuleList(nn.Dropout(p_dropout) for _ in range(n_dropout))
            self.head = nn.Linear(h, 1)
            for lin in (self.attn, self.head):
                nn.init.normal_(lin.weight, mean=0.0, std=0.01)
                nn.init.zeros_(lin.bias)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            hs = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            score = self.attn(hs).squeeze(-1).masked_fill(attention_mask == 0, -1e4)
            attn = torch.softmax(score, dim=-1).unsqueeze(-1)
            pooled = self.norm((hs * attn).sum(1))
            # average head output over K dropout masks (single pass in eval mode)
            return torch.stack([self.head(d(pooled)) for d in self.drops]).mean(0).squeeze(-1)

    def param_groups(model: "Regressor") -> list[dict]:
        """LLRD: top encoder layer at `lr`, each layer below multiplied by `llrd`."""
        layers = [model.enc.embeddings] + list(model.enc.encoder.layer)
        groups = []
        n = len(layers)
        for i, layer in enumerate(layers):
            groups.append({
                "params": list(layer.parameters()),
                "lr": lr * (llrd ** (n - 1 - i)),
                "weight_decay": wd,
            })
        head_params = (
            list(model.attn.parameters()) + list(model.norm.parameters())
            + list(model.head.parameters())
        )
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": 0.0})
        return groups

    def encode(texts: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        enc = tok(
            texts.fillna("").tolist(),
            truncation=True, padding="max_length", max_length=max_len, return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    ids_tr_all, mask_tr_all = encode(train[TEXT_COL])
    ids_te, mask_te = encode(test[TEXT_COL])
    y_t = torch.tensor(y / 100.0, dtype=torch.float32)  # y/100: see v1 note

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_rmses: list[float] = []
    t0 = time.time()

    rng = np.random.default_rng(seed)
    for fold in range(N_FOLDS):
        tr_all = np.where(folds != fold)[0]
        va = np.where(folds == fold)[0]
        perm = rng.permutation(tr_all)
        n_inner = max(1, int(inner_val_frac * len(perm)))
        iv, tr = perm[:n_inner], perm[n_inner:]

        model = Regressor(model_name).to(device)
        opt = torch.optim.AdamW(param_groups(model))
        dl = DataLoader(
            TensorDataset(ids_tr_all[tr], mask_tr_all[tr], y_t[tr]),
            batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        total_steps = epochs * len(dl)
        warmup = max(1, int(0.1 * total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: s / warmup if s < warmup
            else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
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
            return np.concatenate(preds) * 100.0

        best_rmse, best_state = np.inf, None
        for ep in range(epochs):
            model.train()
            for ids, mask, yb in dl:
                ids, mask, yb = ids.to(device), mask.to(device), yb.to(device)
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(
                    model(input_ids=ids, attention_mask=mask), yb
                )
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
            log.info(f"  bert_v2 fold {fold} epoch {ep}: inner_val_rmse={iv_rmse:.4f}")
        if best_state is not None:
            model.load_state_dict(best_state)

        oof[va] = predict(ids_tr_all[va], mask_tr_all[va])
        test_pred += predict(ids_te, mask_te) / N_FOLDS
        fr = rmse(y[va], clip_preds(oof[va]))
        fold_rmses.append(fr)
        log.info(f"  bert_v2 fold {fold}: rmse={fr:.4f}")
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
    return result, max_len, [f"bert_v2={model_name}", f"llrd={llrd}", f"msd={n_dropout}x{p_dropout}"]
