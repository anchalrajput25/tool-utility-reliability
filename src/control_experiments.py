"""
Control experiments for the tool-utility study.

Three analyses, in increasing cost. Each tests a specific threat to the
headline result (POST-search probe beats PRE-search probe on utility, +0.15):

  A. lexical      CPU only. AUROC of "the gold answer appears in the retrieved
                  snippets" for predicting positive utility. If this trivial
                  one-feature check already predicts utility about as well as
                  the probe, the probe is not adding much over grep.

  B. decompose    CPU only, reuses the EXISTING data/hidden.npz. Trains the POST
                  probe to predict (i) positive utility and (ii) answer
                  correctness on the SAME hidden states, and reports how much
                  utility and correctness coincide. If AUROC(POST->correctness)
                  ~= AUROC(POST->utility) and utility ~= correctness on this
                  data, then the +0.15 is a restatement of the known result
                  "hidden states predict answer correctness" (Kadavath 2022,
                  Orgad 2025), not a new insight about tool utility.

  C. mismatch     Needs the model (GPU). Re-extracts POST states on prompts
                  whose retrieved snippets are swapped with ANOTHER question's,
                  holding prompt format and evidence position fixed. Separates
                  evidence CONTENT from format/position:
                    AUROC(mismatch) ~= AUROC(PRE)  -> content carries the signal
                                                      (headline defensible)
                    AUROC(mismatch) ~= AUROC(POST) -> format/position, not
                                                      content (headline collapses)
                    in between                     -> partial.

Reuses the existing pipeline (probes.py, io_utils.py, reliability.py,
diagnostics.py). Writes to results/control/. Nothing is written unless it was
computed from real inputs; there are no default or placeholder numbers.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from io_utils import load_jsonl, load_ground_truth, question_of
from reliability import score_matrix
from diagnostics import gold_in_context
from probes import oof_probe_scores, paired_bootstrap_auc_diff

# A small grid keeps the control analyses fast; pass --paper-grid to match the
# main experiment exactly. (The paper's grid is in probes.PAPER_GRID.)
FAST_GRID = {"mlp__hidden_layer_sizes": [(), (128,)], "mlp__learning_rate_init": [1e-3]}


# --------------------------------------------------------------- labels ----
def aligned_labels(judgments: pd.DataFrame, questions, engine: str = "google"):
    """Per-question utility (rep 0), answer-correctness (force rep 0) and need
    labels, aligned to `questions`. Utility = force score > no-tool score;
    correctness = force answer judged correct; need = no-tool answer wrong.
    Rep 0 is used so the label matches the main experiment's single-run label.
    """
    nt = score_matrix(judgments, "no_tool")
    at = score_matrix(judgments, f"force_{engine}")
    nt, at = nt.align(at, join="inner", axis=0)
    rep = sorted(set(nt.columns) & set(at.columns))[0]
    util = ((at[rep] - nt[rep]) > 0).astype(int)
    correct = at[rep].astype(int)
    need = (nt[rep] == 0).astype(int)
    idx = pd.Index(questions)
    return (util.reindex(idx), correct.reindex(idx), need.reindex(idx))


def _drop_na(y, *Xs):
    m = ~y.isna().values
    y2 = y[m].astype(int).values
    return (y2, m) if not Xs else (y2, [X[m] for X in Xs])


# ------------------------------------------------------- A. lexical --------
def run_lexical(runs_dir, gt_csv, judgments_path, engine="google", out=None):
    gt = load_ground_truth(gt_csv)
    rows = load_jsonl(Path(runs_dir) / f"force_{engine}.jsonl")
    j = pd.DataFrame(load_jsonl(judgments_path)); j["score"] = j["score"].astype(int)
    gold = {question_of(r): bool(gold_in_context(r, gt.get(question_of(r)))) for r in rows}
    qs = [q for q in gold]
    util, _, _ = aligned_labels(j, qs, engine)
    df = pd.DataFrame({"q": qs, "gold": [gold[q] for q in qs], "util": util.values})
    df = df.dropna(subset=["util"]); df["util"] = df["util"].astype(int)
    auc = roc_auc_score(df["util"], df["gold"].astype(float)) if df["util"].nunique() > 1 else None
    res = {"analysis": "lexical_gold_in_snippets_vs_utility", "n": int(len(df)),
           "gold_in_snippets_rate": float(df["gold"].mean()),
           "utility_positive_rate": float(df["util"].mean()),
           "auroc_gold_in_snippets_for_utility": None if auc is None else round(float(auc), 4),
           "note": "Single-feature AUROC: compare against the POST probe's utility AUROC. "
                   "gold_in_context matches against the whole final prompt, which includes the "
                   "question, so this rate is an upper bound on true snippet coverage."}
    _write(res, out); return res


# --------------------------------------------------- B. decompose ----------
def run_decompose(hidden_npz, judgments_path, engine="google", grid=None, seed=42, out=None):
    H = np.load(hidden_npz, allow_pickle=True)
    qs = list(H["questions"]); pre, post = H["pre"], H["post"]
    j = pd.DataFrame(load_jsonl(judgments_path)); j["score"] = j["score"].astype(int)
    util, correct, _ = aligned_labels(j, qs, engine)

    res = {"analysis": "correctness_decomposition", "hidden_npz": str(hidden_npz)}
    # utility <-> correctness overlap (does utility just equal correctness here?)
    both = pd.DataFrame({"u": util.values, "c": correct.values}).dropna()
    res["n_labeled"] = int(len(both))
    res["utility_positive_rate"] = round(float(both["u"].mean()), 4)
    res["correct_positive_rate"] = round(float(both["c"].mean()), 4)
    res["utility_equals_correctness_frac"] = round(float((both["u"] == both["c"]).mean()), 4)

    def auc_for(y_series, X):
        y, [Xk] = _drop_na(y_series, X)
        if len(set(y)) < 2:
            return None, 0
        s = oof_probe_scores(Xk, y, seed=seed, grid=grid or FAST_GRID)
        return round(float(roc_auc_score(y, s)), 4), int(len(y))

    res["auroc_POST_to_utility"], res["n_post_util"] = auc_for(util, post)
    res["auroc_POST_to_correctness"], res["n_post_correct"] = auc_for(correct, post)
    res["auroc_PRE_to_utility"], res["n_pre_util"] = auc_for(util, pre)
    res["interpretation"] = (
        "If auroc_POST_to_correctness is close to auroc_POST_to_utility AND "
        "utility_equals_correctness_frac is high, the POST->utility gain is "
        "explained by known correctness probing rather than a new utility signal.")
    _write(res, out); return res


# ----------------------------------------------- C. mismatch (build) -------
def build_mismatch(runs_dir, engine="google", seed=0, out_prompts=None,
                   results_field="tool_result", dry_run=False):
    """Create prompts with each question's retrieved snippets replaced by another
    question's, holding the template/format fixed.

    Preferred: records carry the raw retrieved text in `results_field`
    (e.g. 'tool_result' / 'search_context'); we swap that field across a
    derangement and (for base-loop records) splice it into final_prompt by
    replacing the original snippet text. If neither the field nor the original
    snippet text can be located in final_prompt, the record is skipped and
    reported, so no malformed prompt is ever produced.

    Returns the list of built records and writes them for the extract step.
    ALWAYS run with --dry-run first and read the printed examples: the splice
    depends on your run format, and a bad splice would silently corrupt the
    control.
    """
    rows = load_jsonl(Path(runs_dir) / f"force_{engine}.jsonl")
    recs = [{"q": question_of(r), "final_prompt": r.get("final_prompt", ""),
             "snip": r.get(results_field) or r.get("search_context") or r.get("tool_result")}
            for r in rows]
    recs = [r for r in recs if r["final_prompt"] and r["snip"]]
    n = len(recs)
    if n < 2:
        raise SystemExit(f"Need >=2 usable records with final_prompt and {results_field}; got {n}. "
                         f"Check the field name with --results-field.")
    rng = np.random.default_rng(seed)
    perm = _derange(n, rng)  # derangement: every question gets a different question's snippets
    built, skipped = [], []
    for i, r in enumerate(recs):
        donor = recs[perm[i]]
        snip_self, snip_other = str(r["snip"]), str(donor["snip"])
        if snip_self and snip_self in r["final_prompt"]:
            mp = r["final_prompt"].replace(snip_self, snip_other)
            built.append({"q": r["q"], "donor_q": donor["q"], "final_prompt": mp})
        else:
            skipped.append(r["q"])
    report = {"built": len(built), "skipped_no_splice": len(skipped),
              "total_usable": n, "seed": seed, "results_field": results_field}
    if dry_run:
        print(json.dumps(report, indent=2))
        for ex in built[:2]:
            print("\n===== SWAPPED EXAMPLE (q = %s, donor = %s) =====" % (ex["q"][:60], ex["donor_q"][:60]))
            print(ex["final_prompt"][-1200:])
        return built, report
    if out_prompts:
        with open(out_prompts, "w") as f:
            for b in built:
                f.write(json.dumps(b) + "\n")
    report["out_prompts"] = out_prompts
    _write(report, None if not out_prompts else str(Path(out_prompts).with_suffix(".report.json")))
    return built, report


def run_mismatch_probe(mismatched_prompts, model_name, judgments_path, hidden_npz,
                       engine="google", grid=None, seed=42, out=None):
    """Extract POST states on the mismatched prompts and compare the utility
    AUROC to the matched POST (from hidden.npz) and to PRE. Needs the model.
    """
    from probes import extract_last_token_states  # torch imported lazily here
    built = load_jsonl(mismatched_prompts)
    qs = [b["q"] for b in built]
    post_mis = extract_last_token_states([b["final_prompt"] for b in built], model_name)

    H = np.load(hidden_npz, allow_pickle=True)
    hq = list(H["questions"]); pre_by_q = dict(zip(hq, H["pre"])); post_by_q = dict(zip(hq, H["post"]))
    keep = [i for i, q in enumerate(qs) if q in post_by_q]
    qs = [qs[i] for i in keep]; post_mis = post_mis[keep]
    pre = np.stack([pre_by_q[q] for q in qs]); post = np.stack([post_by_q[q] for q in qs])

    j = pd.DataFrame(load_jsonl(judgments_path)); j["score"] = j["score"].astype(int)
    util, _, _ = aligned_labels(j, qs, engine)
    y, [pre, post, post_mis] = _drop_na(util, pre, post, post_mis)
    g = grid or FAST_GRID
    s_pre = oof_probe_scores(pre, y, seed=seed, grid=g)
    s_post = oof_probe_scores(post, y, seed=seed, grid=g)
    s_mis = oof_probe_scores(post_mis, y, seed=seed, grid=g)
    res = {"analysis": "mismatched_evidence_control", "model": model_name, "n": int(len(y)),
           "auroc_PRE": round(float(roc_auc_score(y, s_pre)), 4),
           "auroc_POST_matched": round(float(roc_auc_score(y, s_post)), 4),
           "auroc_POST_mismatched": round(float(roc_auc_score(y, s_mis)), 4),
           "matched_vs_mismatched": paired_bootstrap_auc_diff(y, s_mis, s_post),
           "interpretation": "mismatched ~ PRE: evidence content carries the signal. "
                             "mismatched ~ matched POST: format/position, not content."}
    _write(res, out); return res


# ------------------------------------------------------------- utils -------
def _derange(n, rng):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def _write(res, out):
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=2, default=float))
        print("wrote", out)
    else:
        print(json.dumps(res, indent=2, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("lexical"); a.add_argument("--runs", default="runs"); a.add_argument("--gt", required=True)
    a.add_argument("--judgments", default="data/judgments.jsonl"); a.add_argument("--out", default="results/control/lexical.json")
    b = sub.add_parser("decompose"); b.add_argument("--hidden", default="data/hidden.npz")
    b.add_argument("--judgments", default="data/judgments.jsonl"); b.add_argument("--paper-grid", action="store_true")
    b.add_argument("--out", default="results/control/decompose.json")
    c = sub.add_parser("build-mismatch"); c.add_argument("--runs", default="runs")
    c.add_argument("--results-field", default="tool_result"); c.add_argument("--seed", type=int, default=0)
    c.add_argument("--dry-run", action="store_true"); c.add_argument("--out", default="data/mismatched_prompts.jsonl")
    d = sub.add_parser("run-mismatch"); d.add_argument("--prompts", default="data/mismatched_prompts.jsonl")
    d.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct"); d.add_argument("--hidden", default="data/hidden.npz")
    d.add_argument("--judgments", default="data/judgments.jsonl"); d.add_argument("--out", default="results/control/mismatch.json")
    args = ap.parse_args()
    from probes import PAPER_GRID
    if args.cmd == "lexical":
        run_lexical(args.runs, args.gt, args.judgments, out=args.out)
    elif args.cmd == "decompose":
        run_decompose(args.hidden, args.judgments, grid=PAPER_GRID if args.paper_grid else None, out=args.out)
    elif args.cmd == "build-mismatch":
        build_mismatch(args.runs, seed=args.seed, out_prompts=args.out,
                       results_field=args.results_field, dry_run=args.dry_run)
    elif args.cmd == "run-mismatch":
        run_mismatch_probe(args.prompts, args.model, args.judgments if False else args.judgments,
                           args.hidden, out=args.out)


if __name__ == "__main__":
    main()
