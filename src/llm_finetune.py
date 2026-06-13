"""exp023+: LoRA fine-tune of a LARGE LLM as a regression head over the text.

Why this experiment exists. Six transformer fine-tunes (BERT, XLM-R-base/large, both
recipes) all plateaued at OOF RMSE ~11.5, and an aspect-based extraction added nothing
the existing text features didn't already hold. That strongly suggests the text signal
is saturated — BUT every model we tried is <=550M params. This is the one untested
mechanism: a multi-billion-parameter LLM has far more capacity to disentangle subtle,
compositional Turkish phrasing. If the 11.5 floor is a *capacity* limit, a 7-8B model
with LoRA breaks it; if it is a *signal* limit (the text genuinely doesn't carry more
than the profile), this confirms it definitively. Either way it is decisive.

Mechanism: mean-pool the last hidden state of a (optionally 4-bit) causal/encoder LLM,
a small regression head on top, LoRA adapters on the attention/MLP projections, y/100
target, same fold protocol / best-epoch selection as src/bert_finetune_v2.py.

Requires: `pip install peft bitsandbytes accelerate`. GPU only (skips on CPU).
Heavy: budget ~10-30 min/fold depending on model size and 4-bit. Saves OOF/test
artifacts like any experiment, so it can be added to the blend and used as a meta-feature.
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


def run_llm_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    folds: np.ndarray,
    years: np.ndarray,
    cfg: dict,
    device: str,
) -> tuple[dict, int, list[str]]:
    """LoRA fine-tune a large LLM per fold; returns (cv-result dict, n_features, notes)."""
    if device != "cuda" and not cfg.get("allow_cpu", False):
        log.error("llm needs a GPU; a multi-B LoRA fine-tune is infeasible on CPU.")
        sys.exit(1)

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModel, AutoTokenizer

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        log.error("peft not installed — run: pip install peft bitsandbytes accelerate")
        sys.exit(1)

    p = cfg.get("llm", {})
    model_name = p["model"]
    epochs = int(p.get("epochs", 3))
    lr = float(p.get("lr", 1e-4))
    head_lr = float(p.get("head_lr", 1e-3))
    max_len = int(p.get("max_len", 192))
    batch_size = int(p.get("batch_size", 8))
    grad_accum = int(p.get("grad_accum", 2))
    max_grad_norm = float(p.get("max_grad_norm", 1.0))
    lora_r = int(p.get("lora_r", 16))
    lora_alpha = int(p.get("lora_alpha", 32))
    lora_dropout = float(p.get("lora_dropout", 0.05))
    load_4bit = bool(p.get("load_in_4bit", True))
    target_modules = p.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    inner_val_frac = float(p.get("inner_val_frac", 0.1))
    seed = int(p.get("seed", SEED))
    torch.manual_seed(seed)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def base_model():
        kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = {"": 0}
        m = AutoModel.from_pretrained(model_name, **kwargs)
        return get_peft_model(m, LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules, bias="none", task_type="FEATURE_EXTRACTION",
        ))

    class Regressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = base_model()
            h = self.enc.config.hidden_size
            self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1)).to(torch.bfloat16)
            for mod in self.head.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.normal_(mod.weight, std=0.01)
                    nn.init.zeros_(mod.bias)

        def forward(self, input_ids, attention_mask):
            hs = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).to(hs.dtype)
            pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return self.head(pooled).squeeze(-1).float()

    def encode(texts: pd.Series):
        enc = tok(texts.fillna("").tolist(), truncation=True, padding="max_length",
                  max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    ids_all, mask_all = encode(train[TEXT_COL])
    ids_te, mask_te = encode(test[TEXT_COL])
    y_t = torch.tensor(y / 100.0, dtype=torch.float32)

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

        model = Regressor().to(device)
        opt = torch.optim.AdamW([
            {"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ])
        dl = DataLoader(TensorDataset(ids_all[tr], mask_all[tr], y_t[tr]),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
        total = epochs * math.ceil(len(dl) / grad_accum)
        warm = max(1, int(0.1 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / warm if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

        @torch.no_grad()
        def predict(ids, mask):
            model.eval()
            out = []
            for i in range(0, len(ids), 64):
                p_ = model(ids[i:i + 64].to(device), mask[i:i + 64].to(device))
                out.append(p_.float().cpu().numpy())
            return np.concatenate(out) * 100.0

        best, best_state = np.inf, None
        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            for i, (bi, bm, by) in enumerate(dl):
                pr = model(bi.to(device), bm.to(device))
                loss = torch.nn.functional.mse_loss(pr, by.to(device)) / grad_accum
                if not torch.isnan(loss):
                    loss.backward()
                if (i + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step(); sched.step(); opt.zero_grad()
            iv_rmse = rmse(y[iv], clip_preds(predict(ids_all[iv], mask_all[iv])))
            if iv_rmse < best - 1e-3:
                best = iv_rmse
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items() if v.requires_grad}
            log.info(f"  llm fold {fold} epoch {ep}: inner_val_rmse={iv_rmse:.4f}")
        if best_state is not None:
            model.load_state_dict(best_state, strict=False)

        oof[va] = predict(ids_all[va], mask_all[va])
        test_pred += predict(ids_te, mask_te) / N_FOLDS
        fold_rmses.append(rmse(y[va], clip_preds(oof[va])))
        log.info(f"  llm fold {fold}: rmse={fold_rmses[-1]:.4f}")
        del model
        torch.cuda.empty_cache()

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
    return result, max_len, [f"llm={model_name}", f"lora_r={lora_r}", f"4bit={load_4bit}"]
