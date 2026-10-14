"""Q2 — How much of the utility-prediction gap closes once the probe sees the search results?

Representations (final layer, last token, as in the paper's Appendix B.6):
  pre  = hidden state of the NO-TOOL prompt   (what the paper's LUE_x sees)
  post = hidden state of the ALWAYS-TOOL final prompt, which contains the
         search results (information the pre-call probe cannot have)

Classifier = the paper's setup: StandardScaler + sklearn MLPClassifier,
grid over hidden sizes {(), 128, 256, (128,64), (1024,64)} x lr {1e-3, 1e-4},
early stopping, seed 42, inner 5-fold model selection, outer 5-fold
stratified out-of-fold predictions. Both probes share identical folds.

NOTE: a post-search probe is NOT a deployable pre-call controller (the search
cost is already paid). It measures the information bottleneck.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

PAPER_GRID = {
    "mlp__hidden_layer_sizes": [(), (128,), (256,), (128, 64), (1024, 64)],
    "mlp__learning_rate_init": [1e-3, 1e-4],
}


# ---------------------------------------------------- hidden states ---
def extract_last_token_states(prompts: list[str], model_name: str, batch_size: int = 4,
                              device: str | None = None) -> np.ndarray:
    """Final-layer, final-token hidden state for each fully-formatted prompt.
    Prompts saved by the harness are already chat-templated, so no special
    tokens are added again."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=4096, add_special_tokens=False).to(device)
            hs = model(**enc, output_hidden_states=True).hidden_states[-1]
            out.append(hs[:, -1, :].float().cpu().numpy())   # left padding -> last pos is real
    return np.concatenate(out)


# ------------------------------------------------------------ probes ---
def oof_probe_scores(X: np.ndarray, y: np.ndarray, seed: int = 42, n_splits: int = 5,
                     grid: dict | None = None) -> np.ndarray:
    """Out-of-fold P(y=1) with the paper's probe and nested model selection."""
    grid = grid or PAPER_GRID
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.zeros(len(y), dtype=float)
    for tr, te in outer.split(X, y):
        pipe = Pipeline([("scale", StandardScaler()),
                         ("mlp", MLPClassifier(max_iter=100, early_stopping=True,
                                               n_iter_no_change=5, random_state=seed))])
        inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=-1)
        gs.fit(X[tr], y[tr])
        scores[te] = gs.predict_proba(X[te])[:, 1]
    return scores


def paired_bootstrap_auc_diff(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
                              n_boot: int = 2000, seed: int = 0) -> dict:
    """AUROC(b) - AUROC(a) on the same questions, with a paired bootstrap CI."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, s_b) - roc_auc_score(y, s_a)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], s_b[idx]) - roc_auc_score(y[idx], s_a[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"auc_a": roc_auc_score(y, s_a), "auc_b": roc_auc_score(y, s_b),
            "diff": float(base), "ci_low": float(lo), "ci_high": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def gap_closure(auc_pre: float, auc_post: float, ceiling: float) -> float:
    """Share of the pre-probe's distance to the noise ceiling that seeing the
    tool output closes. 1 = all of it, 0 = none, <0 = worse."""
    room = ceiling - auc_pre
    return float((auc_post - auc_pre) / room) if room > 1e-9 else float("nan")


# ----------------------------------------------------- Q3: decisions ---
def budgeted_gain(scores: np.ndarray, delta: np.ndarray, budget_frac: float) -> float:
    """Mean utility gain over no-tool when the top-K questions by score get the tool
    (paper's affordability protocol). delta = s_AT - s_NT per question."""
    k = int(round(budget_frac * len(scores)))
    top = np.argsort(-scores)[:k]
    return float(delta[top].sum() / len(scores))


def oracle_gain(delta: np.ndarray, budget_frac: float) -> float:
    k = int(round(budget_frac * len(delta)))
    best = np.sort(delta)[::-1][:k]
    return float(np.clip(best, 0, None).sum() / len(delta))
