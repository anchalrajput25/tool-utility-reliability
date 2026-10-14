"""Extract pre-search and post-search hidden states for the probes (GPU recommended).

  python src/extract_hidden.py --runs runs --out data/hidden.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from io_utils import load_jsonl, question_of
from probes import extract_last_token_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--engine", default="google")
    ap.add_argument("--out", default="data/hidden.npz")
    a = ap.parse_args()

    nt = {question_of(r): r for r in load_jsonl(Path(a.runs) / "no_tool.jsonl")}
    at = {question_of(r): r for r in load_jsonl(Path(a.runs) / f"force_{a.engine}.jsonl")}
    qs = [q for q in nt if q in at and nt[q].get("final_prompt") and at[q].get("final_prompt")]
    print(f"{len(qs)} questions with both prompts")

    pre = extract_last_token_states([nt[q]["final_prompt"] for q in qs], a.model)
    post = extract_last_token_states([at[q]["final_prompt"] for q in qs], a.model)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, questions=np.array(qs, dtype=object), pre=pre, post=post)
    print(f"saved {a.out}: pre {pre.shape}, post {post.shape}")


if __name__ == "__main__":
    main()
