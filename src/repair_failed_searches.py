"""Repair the always-search runs that never received search results.

In the base local loop, a model-written JSON tool input is passed to the tool
unchanged (harness src/tool_manager.py, ToolManager.arguments_from_input). When
Llama-3.2-3B writes {"search_query": ...} instead of {"query": ...}, the search
function raises a TypeError and the final prompt holds "Search error: ..." instead
of results. This script re-runs ONLY those questions with one change: argument
names for web_search are mapped to `query`. Everything else (model, temperature 0,
prompts, provider, top-5) is the same.

This is a DEVIATION from the authors' harness and must be reported as such. The
searches also run on a later date than the originals, so their snippets differ.
Original files are never modified; every output goes to new paths.

Steps (Colab, GPU + SERPAPI_API_KEY):
  1. python src/repair_failed_searches.py prepare --runs runs --gt ../ToCall_or_NotToCall/data/BFCL.csv --out data/repair
       -> data/repair/failed_subset.csv  (questions whose recorded tool result is an argument error)
       -> data/repair/failed_ids.json
  2. python src/repair_failed_searches.py run --repo ../ToCall_or_NotToCall --subset data/repair/failed_subset.csv \
         --model unsloth/Llama-3.2-3B-Instruct --out runs_repair_raw
  3. python src/repair_failed_searches.py merge --runs runs --repaired runs_repair_raw/force_google.jsonl \
         --judgments data/judgments.jsonl --out runs_repaired
       -> runs_repaired/force_google.jsonl (originals + repaired rows, each repaired row tagged)
       -> runs_repaired/no_tool.jsonl (copied unchanged)
       -> data/judgments_repaired_base.jsonl (original judgments minus the repaired force_google rows)
  4. python src/local_judge.py --runs runs_repaired --gt ... --k 3 --out data/judgments_repaired_base.jsonl
       (cached keys mean only the repaired rows are judged)
  5. python src/extract_hidden.py --runs runs_repaired --model unsloth/Llama-3.2-3B-Instruct --out data/hidden_repaired.npz
  6. python src/make_results.py --runs runs_repaired --judgments data/judgments_repaired_base.jsonl \
         --hidden data/hidden_repaired.npz --gt ... --out results/generated_repaired
Validation: after step 3, re-run diagnostics with --flagger structured on runs_repaired;
tool_arg_error must be 0 among repaired rows, and every repaired row must have a
non-error tool result. Report how many were repaired, how many still failed, and why.
"""
from __future__ import annotations
import argparse, json, runpy, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from io_utils import load_jsonl, question_of
from diagnostics import flag_structured

ARG_ALIASES = ("query", "search_query", "q", "query_string", "search", "search_term", "keywords", "text")


def remap_web_search_args(args: dict, fallback_query: str = "") -> dict:
    """Map any recognised alias to `query`; keep `count` if present; drop the rest."""
    out = {}
    for k in ARG_ALIASES:
        if k in args and str(args[k]).strip():
            out["query"] = args[k]
            break
    if "query" not in out:
        out["query"] = fallback_query
    if "count" in args:
        try:
            out["count"] = int(args["count"])
        except (TypeError, ValueError):
            pass
    return out


def cmd_prepare(a):
    rows = load_jsonl(Path(a.runs) / "force_google.jsonl")
    failed = [question_of(r) for r in rows if flag_structured(r) == "tool_arg_error"]
    gt = pd.read_csv(a.gt)
    sub = gt[gt["question"].isin(set(failed))]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out / "failed_subset.csv", index=False)
    (out / "failed_ids.json").write_text(json.dumps({
        "n_failed_rows": len(failed), "n_unique_questions": len(set(failed)),
        "n_rows_in_subset_csv": int(len(sub)), "questions": sorted(set(failed)),
        "detected_with": "diagnostics.flag_structured == tool_arg_error"}, indent=1))
    print(f"{len(failed)} failed rows, {len(set(failed))} unique questions -> {out/'failed_subset.csv'}")


def cmd_run(a):
    """Run the authors' main.py with ToolManager.arguments_from_input patched."""
    src = str(Path(a.repo).resolve() / "src")
    sys.path.insert(0, src)
    import tool_manager  # the harness module, imported before main.py so the patch is shared
    orig = tool_manager.ToolManager.arguments_from_input

    def patched(self, tool_name, tool_input, fallback_query=""):
        args = orig(self, tool_name, tool_input, fallback_query)
        return remap_web_search_args(args, fallback_query) if tool_name == "web_search" else args

    tool_manager.ToolManager.arguments_from_input = patched
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    sys.argv = ["main.py", "--data", str(Path(a.subset).resolve()), "--task", "bfcl",
                "--model-name", a.model, "--temperature", "0", "--skip-scorer",
                "--web-search", "--force-search", "--search-provider", "google",
                "--output-dir", str(out / "force_google")]
    import os; os.chdir(src)
    runpy.run_path(str(Path(src) / "main.py"), run_name="__main__")
    newest = max((out / "force_google").rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    (out / "force_google.jsonl").write_text(newest.read_text())
    print("repaired runs ->", out / "force_google.jsonl")


def cmd_merge(a):
    orig = load_jsonl(Path(a.runs) / "force_google.jsonl")
    rep = {question_of(r): r for r in load_jsonl(a.repaired)}
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    merged, log = [], []
    for r in orig:
        q = question_of(r)
        if flag_structured(r) == "tool_arg_error" and q in rep:
            new = dict(rep[q])
            status = flag_structured(new)
            new["repair"] = {"original_flag": "tool_arg_error", "new_flag": status,
                             "patched_argument_mapping": True, "merged_at": stamp}
            merged.append(new); log.append({"question": q, "new_flag": status})
        else:
            merged.append(r)
    with open(out / "force_google.jsonl", "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")
    (out / "no_tool.jsonl").write_text((Path(a.runs) / "no_tool.jsonl").read_text())
    repaired_qs = {x["question"] for x in log}
    base = [j for j in load_jsonl(a.judgments)
            if not (j["condition"] == "force_google" and j["question"] in repaired_qs)]
    jb = Path(a.judgments).with_name("judgments_repaired_base.jsonl")
    with open(jb, "w") as f:
        for j in base:
            f.write(json.dumps(j) + "\n")
    summary = {"n_original_rows": len(orig), "n_replaced": len(log),
               "replaced_now_ok": sum(x["new_flag"] == "ok" for x in log),
               "replaced_still_failing": [x for x in log if x["new_flag"] != "ok"],
               "judgments_base": str(jb), "judgments_dropped_for_rejudging": len(load_jsonl(a.judgments)) - len(base)}
    (out / "repair_log.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--runs", default="runs"); p.add_argument("--gt", required=True)
    p.add_argument("--out", default="data/repair")
    r = sub.add_parser("run"); r.add_argument("--repo", required=True); r.add_argument("--subset", required=True)
    r.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct"); r.add_argument("--out", default="runs_repair_raw")
    m = sub.add_parser("merge"); m.add_argument("--runs", default="runs"); m.add_argument("--repaired", required=True)
    m.add_argument("--judgments", default="data/judgments.jsonl"); m.add_argument("--out", default="runs_repaired")
    a = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
