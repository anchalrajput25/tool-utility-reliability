"""Data-quality diagnostics and the clean-set filter used in the report.

Flags, per always-search run:
  tool_arg_error  - the model called web_search with a wrong argument name and
                    the harness returned an error string instead of results
  quota_429       - the search API rejected the call (rate/quota limit)
  context_dependent - short questions with an unresolved pronoun/reference
                    ("Where is his hometown?") that cannot be answered standalone
Also reports whether the gold answer appears in the retrieved context.

  python src/diagnostics.py --runs runs --judgments data/judgments.jsonl \
      --gt ../ToCall_or_NotToCall/data/BFCL.csv --hidden data/hidden.npz --out data/clean
Writes clean judgments + hidden states for:  python src/analyze.py --judgments data/clean/judgments.jsonl --hidden data/clean/hidden.npz
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from io_utils import load_jsonl, load_ground_truth, question_of

PRONOUN = re.compile(r"\b(?:his|her|he|she|they|their|its|the team|the album|the club|"
                     r"the company|the school|the city|this|that)\b", re.I)


def flag(row: dict) -> str:
    text = str(row.get("final_prompt", "")) + str(row.get("response", ""))
    if "unexpected keyword" in text or "not expecting" in text or "_execute_web_search" in text:
        return "tool_arg_error"
    if "429" in text:
        return "quota_429"
    return "ok"


def flag_structured(row: dict) -> str:
    """Same categories as flag(), but read from the recorded tool execution
    (row["tool_calls"][i]["result"]) instead of substring-matching the whole
    prompt and answer. flag() is kept unchanged because the reported clean set
    was built with it; diagnostics.json always lists where the two disagree
    (e.g. a snippet that merely contains the digits "429")."""
    results = [tc.get("result") for tc in row.get("tool_calls", []) or []
               if tc.get("type") == "tool_execution"]
    if not results:
        return "no_tool_execution"
    for res in results:
        txt = res if isinstance(res, str) else json.dumps(res, default=str)
        try:
            obj = json.loads(txt) if isinstance(txt, str) else txt
        except json.JSONDecodeError:
            obj = None
        err = obj.get("error") if isinstance(obj, dict) else None
        if err is None:
            continue
        err = str(err)
        if "unexpected keyword" in err or "_execute_web_search" in err:
            return "tool_arg_error"
        if "429" in err or "quota" in err.lower() or "rate limit" in err.lower():
            return "quota_429"
        return "other_tool_error"
    if "No search results found." in str(row.get("final_prompt", "")):
        return "empty_results"
    return "ok"


def is_context_dependent(q: str) -> bool:
    return bool(PRONOUN.search(q)) and len(q.split()) < 9


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower())


def gold_in_context(row: dict, gold) -> bool:
    toks = [t for t in _norm(gold).split() if len(t) > 2]
    ctx = _norm(row.get("final_prompt", ""))
    return bool(toks) and all(t in ctx for t in toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--judgments", default="data/judgments.jsonl")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--hidden", default="data/hidden.npz")
    ap.add_argument("--engine", default="google")
    ap.add_argument("--out", default="data/clean")
    ap.add_argument("--flagger", choices=["legacy", "structured"], default="legacy",
                    help="legacy reproduces the reported clean set; structured reads tool results")
    a = ap.parse_args()

    gt = load_ground_truth(a.gt)
    rows = load_jsonl(Path(a.runs) / f"force_{a.engine}.jsonl")
    j = pd.DataFrame(load_jsonl(a.judgments))
    acc = j[(j.condition == f"force_{a.engine}") & (j.rep == 0)].groupby("question").score.mean()

    flagger = flag if a.flagger == "legacy" else flag_structured
    d = pd.DataFrame([{"question": question_of(r), "flag": flagger(r),
                       "flag_legacy": flag(r), "flag_structured": flag_structured(r),
                       "context_dependent": is_context_dependent(question_of(r)),
                       "gold_in_context": gold_in_context(r, gt.get(question_of(r))),
                       "correct": acc.get(question_of(r))} for r in rows])
    report = {
        "flagger": a.flagger,
        "flag_counts_legacy": d.flag_legacy.value_counts().to_dict(),
        "flag_counts_structured": d.flag_structured.value_counts().to_dict(),
        "flag_disagreements": d.loc[d.flag_legacy != d.flag_structured,
                                    ["question", "flag_legacy", "flag_structured"]].to_dict("records"),
        "accuracy_by_flag": d.groupby("flag").correct.agg(["mean", "count"]).round(3).to_dict("index"),
        "context_dependent": d.groupby("context_dependent").correct.agg(["mean", "count"]).round(3).to_dict("index"),
        "gold_in_context_rate": round(float(d.gold_in_context.mean()), 3),
        "accuracy_by_gold_in_context": d.groupby("gold_in_context").correct.agg(["mean", "count"]).round(3).to_dict("index"),
    }
    drop = set(d.loc[(d.flag != "ok") | d.context_dependent, "question"])
    clean = d[~d.question.isin(drop)]
    report["clean_n"] = int(len(clean))
    report["clean_accuracy"] = round(float(clean.correct.mean()), 3)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "judgments.jsonl", "w") as f:
        for r in load_jsonl(a.judgments):
            if r["question"] not in drop:
                f.write(json.dumps(r) + "\n")
    if Path(a.hidden).exists():
        H = dict(np.load(a.hidden, allow_pickle=True))
        keep = np.array([q not in drop for q in H["questions"]])
        np.savez(out / "hidden.npz", **{k: (v[keep] if len(v) == len(keep) else v) for k, v in H.items()})
        report["clean_with_hidden"] = int(keep.sum())
    (out / "excluded_questions.json").write_text(json.dumps(sorted(drop), indent=1))
    # every question with its reason(s), so exclusions are auditable, not silent
    ex = d.assign(excluded=d.question.isin(drop),
                  reason=[";".join(x for x in ([f if f != "ok" else ""] + (["context_dependent"] if c else [])) if x)
                          for f, c in zip(d.flag, d.context_dependent)])
    ex[["question", "excluded", "reason", "flag_legacy", "flag_structured", "context_dependent",
        "gold_in_context", "correct"]].to_csv(out / "exclusions.csv", index=False)
    (out / "diagnostics.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
