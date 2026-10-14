"""Q1 — How reliable are the need and utility labels?

Inputs: a long table of judgments with columns
    condition (no_tool | force_<engine>), question, rep, score (0/1)

Label definitions follow the paper (BFCL, binary scoring):
    need    = 1 if no-tool answer is judged incorrect
    utility = 1 if always-tool score > no-tool score  (positive utility)

All metrics are computed per question across repeated measurements, then
summarised with question-level bootstrap confidence intervals.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score


# ---------------------------------------------------------------- tables ---
def score_matrix(judg: pd.DataFrame, condition: str) -> pd.DataFrame:
    """questions x reps matrix of 0/1 scores for one condition."""
    sub = judg[judg["condition"] == condition]
    return sub.pivot_table(index="question", columns="rep", values="score", aggfunc="first")


def need_labels(nt: pd.DataFrame) -> pd.DataFrame:
    return (nt == 0).astype(int)


def utility_labels(nt: pd.DataFrame, at: pd.DataFrame) -> pd.DataFrame:
    """Rep j of utility pairs no-tool rep j with always-tool rep j (independent judge calls)."""
    nt, at = nt.align(at, join="inner", axis=0)
    cols = sorted(set(nt.columns) & set(at.columns))
    return pd.DataFrame((at[cols].values - nt[cols].values > 0).astype(int),
                        index=nt.index, columns=cols)


# -------------------------------------------------------------- metrics ---
def flip_rate(labels: pd.DataFrame) -> float:
    """Share of questions whose label is not identical across all repeats."""
    v = labels.values
    return float((v.max(1) != v.min(1)).mean())


def mean_pairwise_kappa(labels: pd.DataFrame) -> float:
    v = labels.values
    ks = []
    for i, j in itertools.combinations(range(v.shape[1]), 2):
        if len(set(v[:, i]) | set(v[:, j])) > 1:
            ks.append(cohen_kappa_score(v[:, i], v[:, j]))
    return float(np.mean(ks)) if ks else float("nan")


def fleiss_kappa(labels: pd.DataFrame) -> float:
    """Fleiss' kappa for binary labels, treating each repeat as a rater."""
    v = labels.values
    n, k = v.shape
    ones = v.sum(1)
    counts = np.stack([k - ones, ones], 1)
    p_j = counts.sum(0) / (n * k)
    P_i = (np.square(counts).sum(1) - k) / (k * (k - 1))
    P_bar, P_e = P_i.mean(), np.square(p_j).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else float("nan")


def empirical_ceiling_auroc(labels: pd.DataFrame) -> float:
    """Leave-one-repeat-out ceiling.

    For each repeat j, score every question by the mean of its *other* repeats
    and compute AUROC against repeat j. A predictor that knew each question's
    true label propensity could do no better than this on a single-run label,
    so it estimates how much AUROC measurement noise alone takes away.
    Needs >= 2 repeats; with few repeats it is an approximate (conservative) ceiling.
    """
    v = labels.values.astype(float)
    k = v.shape[1]
    aucs = []
    for j in range(k):
        y = v[:, j]
        s = (v.sum(1) - y) / (k - 1)
        if len(set(y)) > 1:
            aucs.append(roc_auc_score(y, s))
    return float(np.mean(aucs)) if aucs else float("nan")


# --------------------------------------------------- noise decomposition ---
def utility_variance_components(judg: pd.DataFrame, engines: list[str]) -> dict:
    """Split within-question variance of Delta = s_AT - s_NT into
    judge vs search-engine parts.

    Delta[q, e, j] uses no-tool rep j and engine e rep j.
      judge part  = mean over engines of variance over judge reps
      search part = variance over engines of the judge-averaged Delta
    """
    nt = score_matrix(judg, "no_tool")
    deltas = []
    for e in engines:
        at = score_matrix(judg, f"force_{e}")
        a, b = nt.align(at, join="inner", axis=0)
        cols = sorted(set(a.columns) & set(b.columns))
        deltas.append(b[cols].values - a[cols].values)
    D = np.stack(deltas, 1)                      # q x engine x rep
    k = D.shape[2]
    # Method-of-moments (random-effects ANOVA): the spread of judge-averaged
    # Delta across engines still contains judge noise / k, so subtract it.
    judge = D.var(axis=2, ddof=1).mean(axis=1)                       # per question
    search = np.clip(D.mean(axis=2).var(axis=1, ddof=1) - judge / k, 0, None)
    total = judge + search
    return {
        "judge_var_mean": float(judge.mean()),
        "search_var_mean": float(search.mean()),
        "judge_share": float(judge.sum() / total.sum()) if total.sum() else float("nan"),
        "search_share": float(search.sum() / total.sum()) if total.sum() else float("nan"),
        "questions_with_any_variance": float((total > 0).mean()),
    }


# ------------------------------------------------------------ bootstrap ---
def bootstrap_ci(labels: pd.DataFrame, fn, n_boot: int = 2000, seed: int = 0):
    """Question-level percentile bootstrap CI for any metric fn(labels)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(labels.iloc[idx]))
        except ValueError:
            continue
    vals = np.array([x for x in vals if not np.isnan(x)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def reliability_report(judg: pd.DataFrame, engine: str = "google") -> pd.DataFrame:
    nt = score_matrix(judg, "no_tool")
    at = score_matrix(judg, f"force_{engine}")
    rows = []
    for name, lab in [("need", need_labels(nt)), ("utility", utility_labels(nt, at))]:
        r = {"label": name, "n_questions": len(lab), "positive_rate": float(lab.values.mean())}
        for mname, fn in [("flip_rate", flip_rate), ("fleiss_kappa", fleiss_kappa),
                          ("ceiling_auroc", empirical_ceiling_auroc)]:
            r[mname] = fn(lab)
            lo, hi = bootstrap_ci(lab, fn)
            r[f"{mname}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(r)
    return pd.DataFrame(rows)
