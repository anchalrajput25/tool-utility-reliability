"""Count silent judge-parse failures.

free_judge._parse() returns 0.0 ("wrong") whenever the judge output has no valid
JSON with a boolean is_correct, e.g. when the 80-token cap cuts the JSON off. That
failure is silent in the score. This script counts it from the raw judge text.

  python src/judge_audit.py --judgments data/judgments.jsonl --out results/remediation/judge_parse_audit.json

The judgments behind the reported results were written before raw text was
logged (they hold only condition/question/rep/judge_model/score). For those the
script reports "not countable" and exits with code 2 instead of guessing. Re-judge
into a NEW file with the updated src/local_judge.py (it logs `raw` and `parse_ok`)
to count; keep the original file unchanged.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
from io_utils import load_jsonl
from free_judge import parse_ok


def audit(rows: list[dict]) -> dict:
    with_raw = [r for r in rows if "raw" in r]
    res = {"n_judgments": len(rows), "n_with_raw_text": len(with_raw)}
    if not with_raw:
        res["status"] = "not_countable"
        res["reason"] = ("no raw judge text in this file; parse failures were scored 0 silently "
                         "and cannot be recovered from scores")
        return res
    fails = [r for r in with_raw if not parse_ok(r["raw"])]
    truncated = [r for r in fails if "{" in (r["raw"] or "") and "}" not in r["raw"]]
    by = Counter((r["condition"], r["rep"]) for r in fails)
    tot = Counter((r["condition"], r["rep"]) for r in with_raw)
    res.update({
        "status": "counted",
        "parse_failures": len(fails),
        "parse_failure_rate": round(len(fails) / len(with_raw), 4),
        "of_which_truncated_json": len(truncated),
        "by_condition_rep": {f"{c}|rep{k}": {"fail": by[(c, k)], "total": n} for (c, k), n in sorted(tot.items())},
        "questions_affected": sorted({r["question"] for r in fails}),
        "max_new_tokens_seen": sorted({r.get("max_new_tokens") for r in with_raw if r.get("max_new_tokens")}),
        "decision_rule": "if parse_failure_rate > 0.01, re-judge with --max-new-tokens 256 into a new file "
                         "and report labels both ways",
    })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = audit(load_jsonl(a.judgments))
    txt = json.dumps(res, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(txt)
    print(txt)
    sys.exit(2 if res["status"] == "not_countable" else 0)


if __name__ == "__main__":
    main()
