"""End-to-end analysis: Q1 (reliability), Q2 (pre vs post probes), Q3 (budgets).

  python src/analyze.py --runs runs --judgments data/judgments.jsonl \
        [--hidden data/hidden.npz] --out results

--hidden is optional (Q2/Q3 need it; produce it with src/extract_hidden.py).
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from io_utils import load_jsonl, question_of
from reliability import (score_matrix, need_labels, utility_labels, reliability_report,
                         utility_variance_components, empirical_ceiling_auroc)

warnings.filterwarnings("ignore")


def determinism(runs: Path) -> dict:
    a, b = runs / "det_a.jsonl", runs / "det_b.jsonl"
    if not (a.exists() and b.exists()):
        return {}
    ra, rb = load_jsonl(a), load_jsonl(b)
    same = sum(x["response"].strip() == y["response"].strip() for x, y in zip(ra, rb))
    return {"identical": same, "total": len(ra)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--judgments", default="data/judgments.jsonl")
    ap.add_argument("--hidden", default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {}

    # ---- A. determinism -------------------------------------------------
    summary["determinism_temp0"] = determinism(Path(a.runs))

    # ---- B/Q1. reliability ---------------------------------------------
    j = pd.DataFrame(load_jsonl(a.judgments))
    j["score"] = j["score"].astype(int)
    engines = [c.replace("force_", "") for c in j["condition"].unique() if c.startswith("force_")]
    nt = score_matrix(j, "no_tool")
    acc = {"no_tool": float(nt[0].mean())}
    for e in engines:
        acc[f"force_{e}"] = float(score_matrix(j, f"force_{e}")[0].mean())
    summary["accuracy_judge_rep0"] = acc
    summary["paper_reference_bfcl_llama"] = {"no_tool": 0.17, "always_tool": 0.65,
                                             "LNE_auroc": 0.812, "LUE_x_auroc": 0.723}

    rel = reliability_report(j, engine="google")
    rel.to_csv(out / "q1_reliability.csv", index=False)
    summary["q1_reliability"] = rel.to_dict(orient="records")

    if len(engines) > 1:
        summary["q1_noise_components"] = utility_variance_components(j, engines)
        ug = utility_labels(nt, score_matrix(j, "force_google"))[0]
        ub = utility_labels(nt, score_matrix(j, "force_brave"))[0]
        g, b = ug.align(ub, join="inner")
        from sklearn.metrics import cohen_kappa_score
        summary["q1_engine_agreement"] = {"agreement": float((g == b).mean()),
                                          "kappa": float(cohen_kappa_score(g, b))}

    # ---- Q2/Q3. probes ---------------------------------------------------
    if a.hidden:
        from probes import (oof_probe_scores, paired_bootstrap_auc_diff, gap_closure,
                            budgeted_gain, oracle_gain)
        H = np.load(a.hidden, allow_pickle=True)
        qs = list(H["questions"])
        at = score_matrix(j, "force_google")
        y_util = utility_labels(nt, at).reindex(qs)
        y_need = need_labels(nt).reindex(qs)
        # main label = paper-style single run (rep 0); denoised = majority over reps
        y1 = y_util[0].values.astype(int)
        y_major = (y_util.mean(1) > 0.5).astype(int).values
        delta = (at.reindex(qs).mean(1) - nt.reindex(qs).mean(1)).values

        res = {}
        s_need = oof_probe_scores(H["pre"], y_need[0].values.astype(int))
        from sklearn.metrics import roc_auc_score
        res["LNE_auroc"] = float(roc_auc_score(y_need[0].values.astype(int), s_need))
        ceiling = empirical_ceiling_auroc(y_util)
        for lab_name, y in [("single_run", y1), ("majority_of_reps", y_major)]:
            s_pre = oof_probe_scores(H["pre"], y)
            s_post = oof_probe_scores(H["post"], y)
            d = paired_bootstrap_auc_diff(y, s_pre, s_post)
            d["gap_closure"] = gap_closure(d["auc_a"], d["auc_b"], ceiling)
            res[lab_name] = d
            if lab_name == "single_run":
                res["budgets"] = {
                    str(bf): {"pre_LUE": budgeted_gain(s_pre, delta, bf),
                              "post_LUE": budgeted_gain(s_post, delta, bf),
                              "LNE": budgeted_gain(s_need, delta, bf),
                              "oracle": oracle_gain(delta, bf)}
                    for bf in (0.2, 0.4, 0.8)}
        res["utility_noise_ceiling"] = ceiling
        summary["q2_q3_probes"] = res

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
