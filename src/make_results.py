"""Regenerate every reported results file from raw data, then compare.

The JSONs in results/ were written by hand from interactive Colab work. This
script rebuilds them from the raw files with the repo's own functions, writes the
regenerated versions to a NEW folder (results/ is never overwritten), and checks
every reported number against its regenerated value.

  python src/make_results.py --runs runs --judgments data/judgments.jsonl \
      --gt ../ToCall_or_NotToCall/data/BFCL.csv --hidden data/hidden.npz \
      --crosscheck-csv human_check_SUGGESTED.csv --out results/generated

Outputs in --out:
  diagnostics/            diagnostics.py output (legacy flag = reported clean set) incl. exclusions.csv
  clean229.json           same schema as results/clean229.json
  full314_final.json      same schema as results/full314_final.json
  judge_crosscheck.json   recomputed from the label CSV + the disagreement list
  oof_scores_*.csv        per-question out-of-fold probe scores (so any AUROC can be re-checked)
  comparison.json         reported vs regenerated, per number: match / MISMATCH / not_regenerable
Exit code 1 if any reported number does not match within its rounding.
"""
from __future__ import annotations
import argparse, json, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, cohen_kappa_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from io_utils import load_jsonl                                     # noqa: E402
from reliability import (score_matrix, need_labels, utility_labels, fleiss_kappa, flip_rate,  # noqa: E402
                         empirical_ceiling_auroc, bootstrap_ci)
from probes import (oof_probe_scores, paired_bootstrap_auc_diff, gap_closure,  # noqa: E402
                    budgeted_gain, oracle_gain)
from remediation import auc_ci                                      # noqa: E402

PAPER = {"no_tool": 0.17, "always_tool": 0.65, "LNE": 0.812, "LUE_x": 0.723}


def r(x, k=3):
    return None if x is None else round(float(x), k)


def determinism(runs: Path):
    a, b = runs / "det_a.jsonl", runs / "det_b.jsonl"
    if not (a.exists() and b.exists()):
        return None
    ra, rb = load_jsonl(a), load_jsonl(b)
    return f"{sum(x['response'].strip() == y['response'].strip() for x, y in zip(ra, rb))}/{len(ra)}"


def label_block(lab: pd.DataFrame) -> dict:
    lo, hi = bootstrap_ci(lab, flip_rate); klo, khi = bootstrap_ci(lab, fleiss_kappa)
    clo, chi = bootstrap_ci(lab, empirical_ceiling_auroc)
    return {"n": int(len(lab)), "pos_rate": r(lab.values.mean()), "flip": r(flip_rate(lab)),
            "flip_ci": [r(lo), r(hi)], "kappa": r(fleiss_kappa(lab)), "kappa_ci": [r(klo), r(khi)],
            "ceiling": r(empirical_ceiling_auroc(lab)), "ceiling_ci": [r(clo), r(chi)]}


def probe_block(j: pd.DataFrame, hidden: Path, tag: str, out: Path) -> dict:
    H = np.load(hidden, allow_pickle=True)
    qs = list(H["questions"])
    nt, at = score_matrix(j, "no_tool"), score_matrix(j, "force_google")
    yu = utility_labels(nt, at).reindex(qs); yn = need_labels(nt).reindex(qs)
    ok = yu.notna().all(1).values & yn.notna().all(1).values
    qs = [q for q, k in zip(qs, ok) if k]; pre, post = H["pre"][ok], H["post"][ok]
    yu, yn = yu[ok], yn[ok]
    y1 = yu[0].values.astype(int); ymaj = (yu.mean(1) > 0.5).astype(int).values
    ceiling = empirical_ceiling_auroc(yu)
    s_need = oof_probe_scores(pre, yn[0].values.astype(int))
    res = {"n_with_hidden": len(qs), "n_dropped_no_label": int((~ok).sum()),
           "LNE": r(roc_auc_score(yn[0].values.astype(int), s_need)), "agreement_reference": r(ceiling)}
    scores = {"question": qs, "need": yn[0].values.astype(int), "utility_rep0": y1, "s_need_pre": s_need}
    for name, y in (("single_run", y1), ("majority", ymaj)):
        sp, so = oof_probe_scores(pre, y), oof_probe_scores(post, y)
        d = paired_bootstrap_auc_diff(y, sp, so)
        res[name] = {"pre": r(d["auc_a"]), "pre_ci": auc_ci(y, sp), "post": r(d["auc_b"]),
                     "post_ci": auc_ci(y, so), "diff": r(d["diff"]), "ci": [r(d["ci_low"]), r(d["ci_high"])],
                     "significant": d["significant"], "gap_closure": r(gap_closure(d["auc_a"], d["auc_b"], ceiling))}
        scores[f"s_pre_{name}"], scores[f"s_post_{name}"] = sp, so
        if name == "single_run":
            delta = (at.reindex(qs).mean(1) - nt.reindex(qs).mean(1)).values
            res["budgets"] = {str(b): {"pre": r(budgeted_gain(sp, delta, b)), "post": r(budgeted_gain(so, delta, b)),
                                       "LNE": r(budgeted_gain(s_need, delta, b)), "oracle": r(oracle_gain(delta, b))}
                              for b in (0.2, 0.4, 0.8)}
    pd.DataFrame(scores).to_csv(out / f"oof_scores_{tag}.csv", index=False)
    return res


def crosscheck(csv_path: Path, reported: Path) -> dict:
    """Second-model labels from the CSV; judge labels = CSV label flipped on the
    ids listed as disagreements (the only judge information saved)."""
    df = pd.read_csv(csv_path)
    col = [c for c in df.columns if c.startswith("YOUR_FINAL")][0]
    sug = [c for c in df.columns if c.startswith("claude_suggestion")][0]
    lab = df[pd.to_numeric(df[col], errors="coerce").notna()]
    lab = lab[pd.to_numeric(lab[col]) >= 0]
    dis = {d["id"]: d for d in json.loads(reported.read_text())["disagreements"]}
    second = lab[col].astype(int).values
    judge = np.array([dis[i]["qwen_judge"] if i in dis else s for i, s in zip(lab["id"], second)])
    return {"source": f"{csv_path.name} + disagreement ids in {reported.name}",
            "n_sampled": int(len(df)), "n_labeled": int(len(lab)),
            "unresolved_ids": sorted(set(df["id"]) - set(lab["id"])),
            "final_equals_model_suggestion_all": bool((df[col].astype(str) == df[sug].astype(str)).all() or
                                                      (lab[col].astype(int) == lab[sug].astype(int)).all()),
            "agreement": r(float((second == judge).mean())), "cohen_kappa": r(cohen_kappa_score(second, judge)),
            "caveat": "judge labels are reconstructed from the disagreement list; the per-id judge verdicts "
                      "were not saved separately"}


# ---------------------------------------------------------- comparison ----
def _flat(d, p=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{p}.{k}" if p else k))
    elif isinstance(d, list) and all(isinstance(x, (int, float)) for x in d):
        for i, v in enumerate(d):
            out[f"{p}[{i}]"] = v
    else:
        out[p] = d
    return out


def _tol(v):
    s = repr(float(v))
    dec = len(s.split(".")[1].rstrip("0")) if "." in s else 0
    return 0.5 * 10 ** (-dec) + 1e-9


def compare(reported: dict, regenerated: dict, mapping: dict) -> list[dict]:
    rep, gen = _flat(reported), _flat(regenerated)
    rows = []
    for k, v in rep.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        gk = mapping.get(k, k)
        if gk is None:
            rows.append({"key": k, "reported": v, "status": "not_regenerable"}); continue
        g = gen.get(gk)
        if g is None:
            rows.append({"key": k, "reported": v, "regenerated_key": gk, "status": "not_regenerable"}); continue
        rows.append({"key": k, "reported": v, "regenerated": g,
                     "status": "match" if abs(float(g) - float(v)) <= _tol(v) else "MISMATCH"})
    return rows


CLEAN_MAP = {  # reported key -> regenerated key (None = no code in the repo produces it)
    "accuracy_by_kind.tool_ok": "accuracy_by_kind.tool_ok", "accuracy_by_kind.tool_arg_error": "accuracy_by_kind.tool_arg_error",
    "accuracy_by_kind.pronoun_questions": "accuracy_by_kind.pronoun_questions",
    "accuracy_by_kind.clean_standalone": "accuracy_by_kind.clean_standalone",
    "time_sensitivity_test.current_latest": None, "time_sensitivity_test.fixed_year": None,
    "time_sensitivity_test.stable": None,
    "clean229.q2.pre": "clean229.q2.pre", "clean229.q2.post": "clean229.q2.post",
    "clean229.q2.diff": "clean229.q2.diff", "clean229.q2.ci[0]": "clean229.q2.ci[0]",
    "clean229.q2.ci[1]": "clean229.q2.ci[1]",
    "clean229.q2.gap_closure_vs_reference": "clean229.q2.gap_closure",
}
FULL_MAP = {"q2_single_run.ci[0]": "q2_single_run.ci[0]"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True); ap.add_argument("--judgments", required=True)
    ap.add_argument("--gt", required=True); ap.add_argument("--hidden", required=True)
    ap.add_argument("--crosscheck-csv"); ap.add_argument("--reported", default=str(HERE.parent / "results"))
    ap.add_argument("--out", default="results/generated")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rep_dir = Path(a.reported)
    if out.resolve() == rep_dir.resolve():
        sys.exit("refusing to overwrite the reported results folder")

    # 1. diagnostics + clean split, with the code path that defined the reported clean set
    dg = out / "diagnostics"
    subprocess.run([sys.executable, str(HERE / "diagnostics.py"), "--runs", a.runs, "--judgments", a.judgments,
                    "--gt", a.gt, "--hidden", a.hidden, "--out", str(dg), "--flagger", "legacy"],
                   check=True, stdout=subprocess.DEVNULL)
    diag = json.loads((dg / "diagnostics.json").read_text())

    j = pd.DataFrame(load_jsonl(a.judgments)); j["score"] = j["score"].astype(int)
    jc = pd.DataFrame(load_jsonl(dg / "judgments.jsonl")); jc["score"] = jc["score"].astype(int)

    # 2. full set
    nt, at = score_matrix(j, "no_tool"), score_matrix(j, "force_google")
    full = {"determinism": determinism(Path(a.runs)),
            "n_rows_force_google": len(load_jsonl(Path(a.runs) / "force_google.jsonl")),
            "n_unique_questions_judged": int(len(at)),
            "accuracy": {"no_tool": r(nt[0].mean()), "force_google": r(at[0].mean())}, "paper": PAPER,
            "need": label_block(need_labels(nt)), "utility": label_block(utility_labels(nt, at))}
    pb = probe_block(j, Path(a.hidden), "full", out)
    full.update({"LNE_auroc": pb["LNE"], "q2_single_run": pb["single_run"], "q2_majority": pb["majority"],
                 "budgets": pb["budgets"], "n_with_hidden": pb["n_with_hidden"]})

    # 3. clean set
    ntc, atc = score_matrix(jc, "no_tool"), score_matrix(jc, "force_google")
    uc = utility_labels(ntc, atc); klo, khi = bootstrap_ci(uc, fleiss_kappa)
    pc = probe_block(jc, dg / "hidden.npz", "clean", out)
    kinds = diag["accuracy_by_flag"]
    clean = {"accuracy_by_kind": {"tool_ok": r(kinds.get("ok", {}).get("mean")),
                                  "tool_arg_error": r(kinds.get("tool_arg_error", {}).get("mean")),
                                  "pronoun_questions": r(diag["context_dependent"].get("True", {}).get("mean")),
                                  "clean_standalone": diag.get("clean_accuracy")},
             "gold_in_search_context": {"rate": diag["gold_in_context_rate"], "computed_on": "all force_google rows",
                                        "acc_if_present": r(diag["accuracy_by_gold_in_context"].get("True", {}).get("mean")),
                                        "acc_if_absent": r(diag["accuracy_by_gold_in_context"].get("False", {}).get("mean"))},
             "clean229": {"n_clean": diag["clean_n"], "n_with_hidden": pc["n_with_hidden"],
                          "accuracy": {"no_tool": r(ntc[0].mean()), "force_google": r(atc[0].mean())},
                          "utility_kappa": r(fleiss_kappa(uc)), "kappa_ci": [r(klo), r(khi)],
                          "agreement_reference": pc["agreement_reference"], "LNE": pc["LNE"], "q2": pc["single_run"],
                          "q2_majority": pc["majority"]}}
    (out / "full314_final.json").write_text(json.dumps(full, indent=1))
    (out / "clean229.json").write_text(json.dumps(clean, indent=1))

    # 4. compare against the hand-written files
    comp = {"clean229.json": compare(json.loads((rep_dir / "clean229.json").read_text()), clean, CLEAN_MAP),
            "full314_final.json": compare(json.loads((rep_dir / "full314_final.json").read_text()), full, FULL_MAP),
            "not_regenerable_files": {
                "full314_valid194.json": "earlier run whose 114 quota-failed searches were later re-run; raw files superseded",
                "pilot50_base_harness.json": "50-question pilot; regenerate only if its run folder was kept"}}
    if a.crosscheck_csv:
        cc = crosscheck(Path(a.crosscheck_csv), rep_dir / "judge_crosscheck.json")
        (out / "judge_crosscheck.json").write_text(json.dumps(cc, indent=1))
        comp["judge_crosscheck.json"] = compare(json.loads((rep_dir / "judge_crosscheck.json").read_text()), cc, {})
    flat = [x for v in comp.values() if isinstance(v, list) for x in v]
    comp["summary"] = {s: sum(x["status"] == s for x in flat) for s in ("match", "MISMATCH", "not_regenerable")}
    (out / "comparison.json").write_text(json.dumps(comp, indent=1))
    print(json.dumps(comp["summary"]))
    for x in flat:
        if x["status"] == "MISMATCH":
            print("MISMATCH", x)
    sys.exit(1 if comp["summary"]["MISMATCH"] else 0)


if __name__ == "__main__":
    main()
