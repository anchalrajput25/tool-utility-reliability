"""Check what the 14 Oct SerpAPI retry actually fixed.

The 114 retried questions (data/failed_search_questions.json) were chosen with a
broad filter: final_prompt contains "429" OR the substring "error" (PROVENANCE C2).
That filter also catches tool-argument-error runs, which fail again at
temperature 0. The retry's "still failing" check looked only for "429", so
"112 of 114 fixed" may count unchanged argument errors as fixed.

This script classifies every retried question before and after the retry with
both flaggers from diagnostics.py and reports the transition table.

  python src/retry_audit.py --before runs/force_google_with429.jsonl \
      --after runs/force_google.jsonl --retried data/failed_search_questions.json \
      --out results/remediation/retry_audit.json

Reads only; never changes the run files.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from io_utils import load_jsonl, question_of
from diagnostics import flag, flag_structured


def broad_filter(row: dict) -> bool:
    fp = str(row.get("final_prompt", ""))
    return "429" in fp or "error" in fp.lower()


def audit(before: list[dict], after: list[dict], retried: set[str]) -> dict:
    b = {question_of(r): r for r in before}
    a = {question_of(r): r for r in after}
    broad = {q for q, r in b.items() if broad_filter(r)}
    res = {
        "n_retried_list": len(retried),
        "broad_filter_on_before_reproduces_list": broad == retried,
        "broad_filter_count_on_before": len(broad),
        "retried_missing_from_before": sorted(retried - set(b)),
        "retried_missing_from_after": sorted(retried - set(a)),
    }
    rows = []
    for q in sorted(retried & set(b) & set(a)):
        rows.append({"question": q,
                     "before_structured": flag_structured(b[q]), "after_structured": flag_structured(a[q]),
                     "before_legacy": flag(b[q]), "after_legacy": flag(a[q]),
                     "after_has_429": "429" in str(a[q].get("final_prompt", "")),
                     "row_replaced": b[q] is not a[q] and b[q] != a[q]})
    res["before_structured"] = dict(Counter(r["before_structured"] for r in rows))
    res["after_structured"] = dict(Counter(r["after_structured"] for r in rows))
    res["transitions_structured"] = dict(Counter(f'{r["before_structured"]} -> {r["after_structured"]}' for r in rows))
    res["counted_fixed_by_original_check"] = sum(not r["after_has_429"] for r in rows)
    res["actually_ok_after"] = sum(r["after_structured"] in ("ok", "empty_results") for r in rows)
    res["true_quota_before"] = sum(r["before_structured"] == "quota_429" for r in rows)
    res["true_quota_fixed"] = sum(r["before_structured"] == "quota_429" and r["after_structured"] in ("ok", "empty_results")
                                  for r in rows)
    res["arg_error_before_and_after"] = sum(r["before_structured"] == "tool_arg_error" == r["after_structured"]
                                            for r in rows)
    res["per_question"] = rows
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--retried", required=True)
    ap.add_argument("--out", default="results/remediation/retry_audit.json")
    a = ap.parse_args()
    res = audit(load_jsonl(a.before), load_jsonl(a.after), set(json.load(open(a.retried))))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_question"}, indent=1))


if __name__ == "__main__":
    main()
