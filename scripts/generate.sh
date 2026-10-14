#!/usr/bin/env bash
# Generate model answers with the paper's harness (unmodified), BFCL task.
# Usage: REPO=../ToCall_or_NotToCall LIMIT=50 bash scripts/generate.sh
set -euo pipefail
REPO=${REPO:-../ToCall_or_NotToCall}
MODEL=${MODEL:-unsloth/Llama-3.2-3B-Instruct}   # ungated copy of meta-llama weights
LIMIT=${LIMIT:-314}
OUT=$(realpath -m ${OUT:-runs})
export AGENT_HARNESS=${AGENT_HARNESS:-}   # base local loop (repo README: use for open vLLM models)
mkdir -p "$OUT"

run () {  # $1=name  $2=extra flags  $3=limit
  echo ">> $1"
  ( cd "$REPO/src" && python main.py --data ../data/BFCL.csv --task bfcl \
      --model-name "$MODEL" --temperature 0 --limit "$3" --skip-scorer \
      --output-dir "$OUT/$1" $2 )
  # copy newest JSONL to a stable name
  cp "$(ls -t $(find "$OUT/$1" -name '*.jsonl') | head -1)" "$OUT/$1.jsonl"
}

# NOTE: repo default temperature is 1.0 (src/config.py); the paper reports 0, so we pass 0.
run det_a   ""   20
run det_b   ""   20
run no_tool ""   "$LIMIT"
run force_google "--web-search --force-search --search-provider google" "$LIMIT"
if [ -n "${BRAVE_API_KEY:-}" ]; then
  run force_brave "--web-search --force-search --search-provider brave" "$LIMIT"
fi
echo "done -> $OUT"
