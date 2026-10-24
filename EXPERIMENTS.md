# Control experiments — how to run, and what each outcome means

These address the two threats to the headline result (POST-search probe beats
PRE-search probe on utility, +0.15 AUROC):

1. Is the gain just **known correctness probing** (hidden states predict whether
   the answer will be right)? — Analyses **A (lexical)** and **B (decompose)**.
2. Does evidence **content** cause the gain, or prompt **format/position**? —
   Analysis **C (mismatch)**.
3. Is the **19% tool-argument failure** real, or an artifact of the base loop? —
   **Experiment 2 (chat harness)**.

All code is in `src/control_experiments.py` and reuses the existing pipeline
(`probes.py`, `io_utils.py`, `reliability.py`, `diagnostics.py`). A synthetic
correctness test is in `tests/test_control_synth.py` (`PYTHONPATH=src python
tests/test_control_synth.py`, CPU, seconds). **No analysis writes a number
unless it was computed from real inputs — there are no placeholder results.**

Prerequisites (the files are gitignored, so run where they live — your Colab):
`data/hidden.npz` (from `src/extract_hidden.py`), `data/judgments.jsonl`,
`runs/force_google.jsonl`, and the BFCL ground-truth CSV.

## A. Lexical baseline — CPU, no model, ~1 min
```bash
python src/control_experiments.py lexical \
  --runs runs --gt ../ToCall_or_NotToCall/data/BFCL.csv \
  --judgments data/judgments.jsonl --out results/control/lexical.json
```
Reports AUROC of "gold answer appears in the snippets" for predicting utility.
**If this ~= the POST probe's utility AUROC (0.75), the probe barely beats grep.**

## B. Correctness decomposition — CPU, reuses hidden.npz, ~2–5 min
```bash
python src/control_experiments.py decompose \
  --hidden data/hidden.npz --judgments data/judgments.jsonl \
  --paper-grid --out results/control/decompose.json
```
Trains POST→utility and POST→correctness on the **same** states, and reports how
often utility == correctness. **If AUROC(POST→correctness) ~= AUROC(POST→utility)
and utility ~= correctness, the +0.15 is correctness probing, not a utility
insight.** This is the single most decisive and cheapest test.

## C. Mismatched-evidence control — needs the model (GPU)
```bash
# 1) build swapped-snippet prompts; ALWAYS dry-run first and read the examples
python src/control_experiments.py build-mismatch --runs runs --dry-run \
  --results-field tool_result          # try search_context if tool_result is absent
# 2) when the swap looks right:
python src/control_experiments.py build-mismatch --runs runs \
  --results-field tool_result --out data/mismatched_prompts.jsonl
# 3) extract POST states on the swapped prompts and compare (same probe/CV/seed)
python src/control_experiments.py run-mismatch \
  --prompts data/mismatched_prompts.jsonl --model unsloth/Llama-3.2-3B-Instruct \
  --hidden data/hidden.npz --judgments data/judgments.jsonl \
  --out results/control/mismatch.json
```
Each question keeps its own prompt format and evidence **position** but gets
**another** question's snippets (a derangement, so no question keeps its own).
Interpretation:

| Result | Meaning |
|---|---|
| AUROC(mismatch) ~= AUROC(PRE), ~0.60 | evidence **content** carries the signal — headline defensible |
| AUROC(mismatch) ~= AUROC(POST), ~0.75 | **format/position**, not content — headline collapses |
| in between | partial; report the split |

The control isolates content from format/position; it does **not** change the
label, the splits, the probe grid, or the CV folds, so the only moving part is
the snippet content.

## Experiment 2 — chat-template harness, does the 19% survive? (GPU)
The paper runs Llama-3.2-3B on its **chat** harness; your runs used the **base
loop**. Re-run forced search on a subset with the chat harness, then recount:
```bash
# in the paper's repo (ToCall_or_NotToCall); confirm the flag in its README
AGENT_HARNESS=chat TOOL=web_search SEARCH_PROVIDER=google TASK=bfcl LIMIT=60 \
  bash src/_script_chat_harness.sh
# then reuse your own diagnostics to recount the tool-argument-error rate:
python src/diagnostics.py --runs <chat_runs_dir> \
  --judgments <chat_judgments.jsonl> --gt ../ToCall_or_NotToCall/data/BFCL.csv
```
`diagnostics.flag()` already classifies `tool_arg_error`. Compare the new rate to
19%. **A subset of 60 is enough to tell "~0%" from "~19%"**: if the base loop's
19% were the true rate, seeing 0/60 errors would be extremely unlikely (one-sided
95% upper bound on 0/60 is ~5%). Report the exact count and n; do not assume.

## Integrity notes
- Reuse the same labels (rep 0), splits (nested 5-fold, seed 42), and probe grid
  (`--paper-grid`) as the main experiment; the controls change one factor each.
- No leakage: out-of-fold predictions only; the mismatch swap never touches labels.
- Report n and exclusions every time (the helpers print them).
- A truncated/`-1` cross-check answer is excluded, never assigned a verdict.

## Remediation analyses (added 24 Oct 2026)
`src/remediation.py` adds four CPU analyses on the same files: `stratify`
(utility = correctness x need, so PRE vs POST is compared inside need = 1, plus a
need x correctness product baseline), `textonly` (length, overlap, TF-IDF
logistic regression on the same snippets), `seeds` (10 seeds + label
permutation) and `foldmean` (the paper's fold mean +/- SD). Commands, inputs and
pass/fail criteria are in `docs/REMEDIATION_PLAN.md`. `src/make_results.py`
regenerates every `results/*.json` from raw data and reports mismatches.
