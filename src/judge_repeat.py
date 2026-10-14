"""Re-judge saved responses K times with the paper's own BFCL judge (LLMJudgeScorer).

Measures judge (evaluation) noise on *identical* outputs. Results are cached so
re-running never re-bills a call that already succeeded.

Usage:
  python src/judge_repeat.py --run runs/no_tool.jsonl --name no_tool \
      --gt ../ToCall_or_NotToCall/data/BFCL.csv --repo ../ToCall_or_NotToCall --k 3
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from io_utils import load_jsonl, load_ground_truth, question_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="JSONL produced by the harness")
    ap.add_argument("--name", required=True, help="condition name, e.g. no_tool / force_google")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--repo", required=True, help="path to ToCall_or_NotToCall")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--backend", choices=["free", "openai"], default="free",
                    help="free = paper's prompt on an OpenAI-compatible free API (Gemini); "
                         "openai = paper's own LLMJudgeScorer with GPT-4o")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--out", default="data/judgments.jsonl")
    a = ap.parse_args()

    if a.backend == "openai":
        sys.path.insert(0, str(Path(a.repo) / "src"))
        from scorer import LLMJudgeScorer  # the paper's scorer, unmodified
        a.judge_model = a.judge_model or "gpt-4o"
        judge = LLMJudgeScorer(judge_model=a.judge_model)
    else:
        from free_judge import FreeJudge
        judge = FreeJudge(model=a.judge_model)
        a.judge_model = judge.model
    gt = load_ground_truth(a.gt)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        for r in load_jsonl(out):
            done.add((r["condition"], r["question"], r["rep"], r["judge_model"]))

    with open(out, "a") as f:
        for row in load_jsonl(a.run):
            q = question_of(row)
            if q not in gt:
                print(f"[skip] no ground truth for: {q[:60]}"); continue
            for rep in range(a.k):
                key = (a.name, q, rep, a.judge_model)
                if key in done:
                    continue
                s = judge.score(response=row["response"], ground_truth=gt[q], question=q)
                f.write(json.dumps({"condition": a.name, "question": q, "rep": rep,
                                    "judge_model": a.judge_model, "score": s}) + "\n")
                f.flush()
    print(f"judgments cached in {out}")


if __name__ == "__main__":
    main()
