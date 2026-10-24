"""Remediation analyses requested by the 24 Oct 2026 audit (see REMEDIATION_PLAN.md).

All four run on CPU from files the main pipeline already produces; none writes a
number that was not computed from real inputs.

  stratify   Correctness-stratified decomposition. On BFCL with binary scores,
             utility = (tool answer correct) AND (no-tool answer wrong), i.e.
             utility = correct x need, exactly. So:
               * inside need = 0, utility is always 0 (nothing to predict);
               * inside need = 1, utility IS tool-answer correctness.
             We report PRE/POST AUROC for need, correctness and utility, the
             PRE-vs-POST utility gain inside need = 1 (paired bootstrap), and a
             "product" baseline P(need | PRE) x P(correct | POST). If the product
             matches POST->utility, utility prediction reduces to need probing +
             correctness probing.

  textonly   Text-only baselines on the SAME retrieved snippets, no hidden states:
             prompt length, number of results, question-snippet word overlap, a
             nested-CV TF-IDF logistic regression, and (diagnostic reference only,
             it uses the gold answer) gold-string-in-snippets restricted to the
             snippet span.

  seeds      Multi-seed evaluation: re-runs the PRE and POST utility probes with
             different outer/inner fold splits and MLP seeds, reports the spread
             of the gain, plus a label-permutation negative control.

  foldmean   Paper-style aggregation: per-fold AUROC mean +/- SD (Wu et al. 2026,
             Table 4) next to the pooled out-of-fold AUROC used in our report,
             with label positive rates, to explain PRE 0.60 vs paper LUE_x 0.72.

Usage (from the repo root):
  python src/remediation.py stratify --hidden data/clean/hidden.npz --judgments data/judgments.jsonl --out results/remediation/stratify_clean.json
  python src/remediation.py textonly --runs runs --gt ../ToCall_or_NotToCall/data/BFCL.csv --judgments data/judgments.jsonl --hidden data/clean/hidden.npz --out results/remediation/textonly_clean.json
  python src/remediation.py seeds    --hidden data/clean/hidden.npz --judgments data/judgments.jsonl --n-seeds 10 --out results/remediation/seeds_clean.json
  python src/remediation.py foldmean --hidden data/clean/hidden.npz --judgments data/judgments.jsonl --out results/remediation/foldmean_clean.json
Add --fast-grid for a quick smoke run; reported numbers must use the default (paper) grid.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from io_utils import load_jsonl, load_ground_truth, question_of
from reliability import score_matrix
from probes import PAPER_GRID, paired_bootstrap_auc_diff

FAST_GRID = {"mlp__hidden_layer_sizes": [(), (128,)], "mlp__learning_rate_init": [1e-3]}
PAPER_TABLE4_LLAMA_BFCL = {"LNE": (0.812, 0.039), "LUE_x": (0.723, 0.024), "LUE_x_dF": (0.680, 0.066)}


# ---------------------------------------------------------------- labels ---
def labels_rep0(judgments: pd.DataFrame, questions, engine: str = "google") -> pd.DataFrame:
    """need / correct / utility per question from judge repeat 0 (the single-run
    label used in the main experiment), aligned to `questions` (NaN if missing)."""
    nt = score_matrix(judgments, "no_tool")
    at = score_matrix(judgments, f"force_{engine}")
    nt, at = nt.align(at, join="inner", axis=0)
    rep = sorted(set(nt.columns) & set(at.columns))[0]
    df = pd.DataFrame({"need": (nt[rep] == 0).astype(int),
                       "correct": at[rep].astype(int),
                       "utility": ((at[rep] - nt[rep]) > 0).astype(int)})
    return df.reindex(pd.Index(questions))


def load_judgments(path) -> pd.DataFrame:
    j = pd.DataFrame(load_jsonl(path))
    j["score"] = j["score"].astype(int)
    return j


# ------------------------------------------------------------ probe core ---
def oof_scores_and_folds(X, y, seed=42, n_splits=5, grid=None):
    """Same estimator, grid and nested CV as probes.oof_probe_scores (verified
    equal in tests), but also returns the outer-fold index of every item so a
    per-fold AUROC can be reported."""
    grid = grid or PAPER_GRID
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.zeros(len(y), dtype=float)
    fold = np.full(len(y), -1)
    for k, (tr, te) in enumerate(outer.split(X, y)):
        pipe = Pipeline([("scale", StandardScaler()),
                         ("mlp", MLPClassifier(max_iter=100, early_stopping=True,
                                               n_iter_no_change=5, random_state=seed))])
        inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        gs = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=-1)
        gs.fit(X[tr], y[tr])
        scores[te] = gs.predict_proba(X[te])[:, 1]
        fold[te] = k
    return scores, fold


def fold_auc_stats(y, s, fold):
    aucs = [roc_auc_score(y[fold == k], s[fold == k]) for k in np.unique(fold)
            if len(set(y[fold == k])) > 1]
    return {"pooled_oof_auroc": round(float(roc_auc_score(y, s)), 4),
            "fold_mean_auroc": round(float(np.mean(aucs)), 4),
            "fold_sd_auroc": round(float(np.std(aucs, ddof=1)), 4) if len(aucs) > 1 else None,
            "per_fold": [round(float(a), 4) for a in aucs]}


def auc_ci(y, s, n_boot=2000, seed=0):
    """Question-level percentile bootstrap CI for one AUROC (fixed OOF scores)."""
    rng = np.random.default_rng(seed); n = len(y); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def _hidden(path):
    H = np.load(path, allow_pickle=True)
    return list(H["questions"]), H["pre"], H["post"]


def _write(res, out):
    txt = json.dumps(res, indent=2, default=float)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(txt); print("wrote", out)
    print(txt)


# ------------------------------------------------------------ stratify -----
def run_stratify(hidden, judgments, grid=None, seed=42, out=None):
    qs, pre, post = _hidden(hidden)
    lab = labels_rep0(load_judgments(judgments), qs)
    keep = lab.notna().all(1).values
    lab = lab[keep].astype(int); pre, post = pre[keep], post[keep]
    need, cor, util = (lab[c].values for c in ("need", "correct", "utility"))
    res = {"analysis": "correctness_stratified_decomposition", "hidden": str(hidden),
           "n": int(len(util)), "n_dropped_missing_labels": int((~keep).sum()),
           "positive_rate": {c: round(float(lab[c].mean()), 4) for c in lab},
           # identity check: utility must equal correct * need for binary scores
           "utility_equals_correct_times_need": bool(np.array_equal(util, cor * need)),
           "utility_equals_correctness_frac": round(float((util == cor).mean()), 4)}

    scores = {}
    for rep_name, X in (("PRE", pre), ("POST", post)):
        for lab_name, y in (("need", need), ("correct", cor), ("utility", util)):
            if len(set(y)) < 2:
                continue
            s, _ = oof_scores_and_folds(X, y, seed=seed, grid=grid)
            scores[(rep_name, lab_name)] = s
            res[f"auroc_{rep_name}_to_{lab_name}"] = round(float(roc_auc_score(y, s)), 4)

    d = paired_bootstrap_auc_diff(util, scores[("PRE", "utility")], scores[("POST", "utility")])
    res["utility_gain_all"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

    # inside need = 1, utility == correctness: train and evaluate only there
    m = need == 1
    res["need1"] = {"n": int(m.sum()), "utility_positive_rate": round(float(util[m].mean()), 4)}
    if m.sum() >= 20 and len(set(util[m])) == 2:
        s_pre, _ = oof_scores_and_folds(pre[m], util[m], seed=seed, grid=grid)
        s_post, _ = oof_scores_and_folds(post[m], util[m], seed=seed, grid=grid)
        d1 = paired_bootstrap_auc_diff(util[m], s_pre, s_post)
        res["need1"]["PRE_vs_POST_utility"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d1.items()}
    res["need0"] = {"n": int((~m).sum()), "utility_positives": int(util[~m].sum()),
                    "note": "utility is 0 by definition when the no-tool answer is already correct"}

    # product baseline: need from PRE x correctness from POST
    if ("PRE", "need") in scores and ("POST", "correct") in scores:
        prod = scores[("PRE", "need")] * scores[("POST", "correct")]
        res["auroc_product_PREneed_x_POSTcorrect_to_utility"] = round(float(roc_auc_score(util, prod)), 4)
        dp = paired_bootstrap_auc_diff(util, prod, scores[("POST", "utility")])
        res["POSTutility_minus_product"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in dp.items()}
    res["how_to_read"] = (
        "If POST's gain over PRE survives inside need=1 and POST->utility clearly beats the "
        "product baseline, the post-search state carries utility signal beyond need+correctness. "
        "If the product matches POST->utility, the +0.15 is need probing plus correctness probing.")
    _write(res, out); return res


# ------------------------------------------------------------ textonly -----
SNIP_RE = re.compile(r"Search Results:(.*?)(?:Question:|$)", re.S)


def snippet_span(final_prompt: str) -> str:
    """Retrieved-results text inside a base-loop final prompt ('' if none)."""
    m = SNIP_RE.search(final_prompt or "")
    return m.group(1) if m else ""


def _toks(s):
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split() if len(t) > 2]


STOP = set("the and for what who which when where was were with from that this are has have how did does name".split())


def overlap(q, snip):
    qt = set(_toks(q)) - STOP
    return len(qt & set(_toks(snip))) / len(qt) if qt else 0.0


def gold_in_span(gold, snip):
    """(present, usable): usable=False when the gold has no token > 2 chars."""
    toks = _toks(gold)
    if not toks:
        g = re.sub(r"[^a-z0-9]", "", str(gold).lower())
        return (bool(g) and g in re.sub(r"[^a-z0-9]", "", snip.lower())), False
    st = set(_toks(snip))
    return all(t in st for t in toks), True


def tfidf_oof(texts, y, seed=42, n_splits=5):
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    s = np.zeros(len(y))
    for tr, te in outer.split(np.zeros(len(y)), y):
        pipe = Pipeline([("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
                         ("lr", LogisticRegression(max_iter=2000))])
        inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        gs = GridSearchCV(pipe, {"lr__C": [0.01, 0.1, 1.0, 10.0]}, cv=inner, scoring="roc_auc")
        gs.fit([texts[i] for i in tr], y[tr])
        s[te] = gs.predict_proba([texts[i] for i in te])[:, 1]
    return s


def run_textonly(runs, gt_csv, judgments, hidden=None, engine="google", seed=42, out=None):
    gt = load_ground_truth(gt_csv)
    rows = {question_of(r): r for r in load_jsonl(Path(runs) / f"force_{engine}.jsonl")}
    qs = _hidden(hidden)[0] if hidden else list(rows)
    missing = [q for q in qs if q not in rows]
    qs = [q for q in qs if q in rows]
    lab = labels_rep0(load_judgments(judgments), qs)
    ok = lab.notna().all(1).values
    qs = [q for q, k in zip(qs, ok) if k]; lab = lab[ok].astype(int)
    fp = [str(rows[q].get("final_prompt", "")) for q in qs]
    spans = [snippet_span(p) for p in fp]
    gold = [gold_in_span(gt.get(q, ""), sp) for q, sp in zip(qs, spans)]
    feats = {"prompt_length_chars": np.array([len(p) for p in fp], float),
             "n_results": np.array([len(re.findall(r"^\s*\d+\.\s", sp, re.M)) for sp in spans], float),
             "question_snippet_overlap": np.array([overlap(q, sp) for q, sp in zip(qs, spans)])}
    res = {"analysis": "text_only_baselines", "runs": str(runs), "restricted_to_hidden": str(hidden),
           "n": int(len(qs)), "n_missing_in_runs": len(missing), "n_missing_labels": int((~ok).sum()),
           "n_empty_snippet_span": int(sum(not sp.strip() for sp in spans)),
           "gold_unusable_short_tokens": int(sum(not u for _, u in gold)), "auroc": {}}
    texts = [q + " \n " + sp for q, sp in zip(qs, spans)]
    for target in ("utility", "correct"):
        y = lab[target].values
        if len(set(y)) < 2:
            continue
        r = {name: round(float(roc_auc_score(y, f)), 4) for name, f in feats.items()}
        s_tfidf = tfidf_oof(texts, y, seed=seed)
        r["tfidf_logreg_nestedcv"] = round(float(roc_auc_score(y, s_tfidf)), 4)
        r["tfidf_logreg_ci"] = auc_ci(y, s_tfidf)
        r["REFERENCE_ONLY_gold_in_snippets"] = round(float(roc_auc_score(y, [float(p) for p, _ in gold])), 4)
        res["auroc"][target] = r
    res["gold_in_snippets_rate"] = round(float(np.mean([p for p, _ in gold])), 4)
    res["note"] = ("Single features are scored directly (no fitting). The TF-IDF model uses the same "
                   "outer folds (seed 42) as the probes. gold_in_snippets uses the gold answer and is a "
                   "diagnostic reference, not a legitimate baseline. Compare against the POST probe AUROC "
                   "on the same question set.")
    _write(res, out); return res


# --------------------------------------------------------------- seeds -----
def run_seeds(hidden, judgments, n_seeds=10, n_perm=20, grid=None, out=None):
    qs, pre, post = _hidden(hidden)
    lab = labels_rep0(load_judgments(judgments), qs)
    keep = lab["utility"].notna().values
    y = lab["utility"][keep].astype(int).values; pre, post = pre[keep], post[keep]
    rows = []
    for s in range(n_seeds):
        sp, _ = oof_scores_and_folds(pre, y, seed=s, grid=grid)
        so, _ = oof_scores_and_folds(post, y, seed=s, grid=grid)
        a, b = roc_auc_score(y, sp), roc_auc_score(y, so)
        rows.append({"seed": s, "pre": round(float(a), 4), "post": round(float(b), 4), "diff": round(float(b - a), 4)})
        print(rows[-1])
    D = np.array([r["diff"] for r in rows])
    rng = np.random.default_rng(12345); perm = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        sp, _ = oof_scores_and_folds(post, yp, seed=42, grid=FAST_GRID)
        perm.append(round(float(roc_auc_score(yp, sp)), 4))
    res = {"analysis": "multi_seed_and_permutation", "hidden": str(hidden), "n": int(len(y)),
           "grid": "paper" if grid is None or grid == PAPER_GRID else "fast", "per_seed": rows,
           "diff_mean": round(float(D.mean()), 4), "diff_sd": round(float(D.std(ddof=1)), 4) if len(D) > 1 else None,
           "diff_min": round(float(D.min()), 4), "diff_max": round(float(D.max()), 4),
           "frac_seeds_diff_gt_0": round(float((D > 0).mean()), 4),
           "permutation_POST_auroc": {"n_perm": n_perm, "mean": round(float(np.mean(perm)), 4),
                                      "p95": round(float(np.percentile(perm, 95)), 4), "values": perm,
                                      "note": "labels shuffled; fast grid for cost; should centre on 0.5"}}
    _write(res, out); return res


# ------------------------------------------------------------ foldmean -----
def run_foldmean(hidden, judgments, grid=None, seed=42, out=None):
    qs, pre, post = _hidden(hidden)
    lab = labels_rep0(load_judgments(judgments), qs)
    keep = lab.notna().all(1).values
    lab = lab[keep].astype(int); pre = pre[keep]
    res = {"analysis": "paper_style_fold_aggregation", "hidden": str(hidden), "n": int(len(lab)),
           "positive_rate": {c: round(float(lab[c].mean()), 4) for c in lab},
           "paper_table4_llama32_3b_bfcl_mean_sd": PAPER_TABLE4_LLAMA_BFCL}
    for name, y in (("LNE_need_from_PRE", lab["need"].values), ("LUE_x_utility_from_PRE", lab["utility"].values)):
        s, f = oof_scores_and_folds(pre, y, seed=seed, grid=grid)
        res[name] = fold_auc_stats(y, s, f)
        res[name]["pooled_ci"] = auc_ci(y, s)
    res["note"] = ("The paper reports mean +/- SD of per-fold AUROC over 5 stratified folds on all 314 "
                   "questions, GPT-4o judge, chat-template harness. Our report used pooled out-of-fold "
                   "AUROC. Differences that remain after matching aggregation come from labels, judge, "
                   "harness, n, or randomness.")
    _write(res, out); return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stratify", "seeds", "foldmean"):
        p = sub.add_parser(name)
        p.add_argument("--hidden", required=True); p.add_argument("--judgments", required=True)
        p.add_argument("--out"); p.add_argument("--fast-grid", action="store_true")
        if name == "seeds":
            p.add_argument("--n-seeds", type=int, default=10); p.add_argument("--n-perm", type=int, default=20)
    t = sub.add_parser("textonly")
    t.add_argument("--runs", required=True); t.add_argument("--gt", required=True)
    t.add_argument("--judgments", required=True); t.add_argument("--hidden"); t.add_argument("--out")
    a = ap.parse_args()
    grid = FAST_GRID if getattr(a, "fast_grid", False) else None
    if a.cmd == "stratify":
        run_stratify(a.hidden, a.judgments, grid=grid, out=a.out)
    elif a.cmd == "seeds":
        run_seeds(a.hidden, a.judgments, n_seeds=a.n_seeds, n_perm=a.n_perm, grid=grid, out=a.out)
    elif a.cmd == "foldmean":
        run_foldmean(a.hidden, a.judgments, grid=grid, out=a.out)
    else:
        run_textonly(a.runs, a.gt, a.judgments, hidden=a.hidden, out=a.out)


if __name__ == "__main__":
    main()
