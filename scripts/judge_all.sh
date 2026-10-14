#!/usr/bin/env bash
# Judge every condition K times (cached; safe to re-run).
# JUDGE=local  (default, free): open model on the Colab GPU
# JUDGE=free   : paper prompt via a free OpenAI-compatible API (needs JUDGE_API_KEY)
# JUDGE=openai : paper's own GPT-4o scorer (needs OPENAI_API_KEY)
set -euo pipefail
REPO=${REPO:-../ToCall_or_NotToCall}; K=${K:-3}; RUNS=${RUNS:-runs}; JUDGE=${JUDGE:-local}
if [ "$JUDGE" = "local" ]; then
  python src/local_judge.py --runs "$RUNS" --gt "$REPO/data/BFCL.csv" --k "$K"
else
  for c in no_tool force_google force_brave; do
    [ -f "$RUNS/$c.jsonl" ] || continue
    python src/judge_repeat.py --run "$RUNS/$c.jsonl" --name "$c" --gt "$REPO/data/BFCL.csv" --repo "$REPO" --k "$K" --backend "$JUDGE"
  done
fi
