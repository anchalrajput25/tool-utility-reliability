# Remediation plan (24 Oct 2026)

Source audit: "Max Planck project: research-level review" (24 Oct 2026). This plan maps every blocker and must-fix item to files, inputs, outputs and a pass/fail check. Status words: **done** (done and checked), **code ready** (written and tested on synthetic data only), **blocked** (needs a file, key or GPU we do not have here).

**The project is not ready to share as a finished result.** It can be shared as a preliminary prototype once every A item is done or explicitly disclosed.

## What was accessible for this work

| Needed | Accessible? | Where |
| --- | --- | --- |
| Repo code, README, report, results JSONs | yes | local repo `Desktop/github-upload/tool-utility-reliability` at `c554337` (public GitHub is still `fd39285`) |
| `control_experiments.py`, `EXPERIMENTS.md` | yes | local commit `c554337` |
| Cross-check labels | yes | `Desktop/tool-utility-reliability/human_check_SUGGESTED.csv` (copied to `results/crosscheck_labels.csv`) |
| Authors' harness and `BFCL.csv` | yes | github.com/QinyuanWu0710/ToCall_or_NotToCall at `0d2c356` |
| `data/hidden.npz` | **no** | Colab / Google Drive `MyDrive/tool-utility-reliability/` (notebook saves `runs data results` there) |
| `data/judgments.jsonl` | **no** | same |
| `runs/*.jsonl` (no_tool, force_google, det_a, det_b) | **no** | same; contain SerpAPI URLs, never commit |
| GPU, `SERPAPI_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY` | **no** | Colab Secrets |
| Push access to GitHub | **no** | push from GitHub Desktop |

Consequence: no new number on real data was produced here. Every analysis below is ready to run, and none has been run on real data.

## A. Critical before sharing with the professor

| # | Task | Files | Inputs | Expected output | Validation | Status |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | Correct public cross-check stats and claims | `results/judge_crosscheck.json` (fixed in `c554337`), `results/crosscheck_labels.csv` (new), `README.md`, `report/REPORT.txt`, `report/Report_ToolUtility.pdf` | local repo | public repo shows both versions: 45/50 = 0.90, κ 0.80 (original, 23/37 resolved) and 44/48 = 0.917, κ 0.83 (23/37 dropped later) | Both recomputed from the label CSV and the Qwen rep-0 vector in `docs/PROVENANCE.md` C8: 0.900 / 0.799 and 0.917 / 0.832. Public README no longer says "significantly improves", "unmodified harness" or 90% / n = 50 | **done locally; push blocked** (needs you: delete `.git/HEAD.lock`, commit, push) |
| A2 | Explain PRE 0.60 vs paper LUE_x 0.72 | `docs/PAPER_COMPARISON.md`, README "Deviations" | paper Table 4; for numbers: `hidden.npz`, `judgments.jsonl` | factor table + paper-style fold mean ± SD | `remediation.py foldmean` output reported next to the paper's 0.723 ± 0.024, full and clean | **text done; numbers blocked** (CPU, ~10 min, needs data) |
| A3 | Regenerate every results file from raw data | `src/make_results.py` (new) | `runs/`, `data/judgments.jsonl`, `data/hidden.npz`, `BFCL.csv`, `results/crosscheck_labels.csv` | `results/generated/*.json`, per-question OOF scores, `comparison.json` | `comparison.json` summary has MISMATCH = 0, or every mismatch is reported in the README. Exits 1 on any mismatch | **code ready** (synthetic test: 97/97 match, a changed number is caught) |
| A4 | Count silent judge-parse failures | `src/judge_audit.py` (new), `src/local_judge.py` and `src/free_judge.py` (now log `raw`, `parse_ok`) | judgments with raw text | parse-failure count and rate by condition and repeat | Existing `judgments.jsonl` has no raw text, so the script returns `not_countable` (exit 2) rather than guessing. Re-judge into a NEW file to count; if rate > 1%, re-judge with `--max-new-tokens 256` and report labels both ways | **code ready; count blocked** (GPU) |
| A5 | Disclose exclusions, not silently drop | `src/diagnostics.py` (now writes `exclusions.csv` with reasons, and compares the legacy and structured failure flags) | `runs/force_google.jsonl` | every one of the 82 excluded questions listed with its reason | `flag_disagreements` in `diagnostics.json` is empty or explained (the legacy flag marks any text containing "429" as a quota error) | **code ready** |
| A7 | Verify the "114 quota failures, 112 fixed" claim | `src/retry_audit.py` (new) | `runs/force_google_with429.jsonl`, `runs/force_google.jsonl`, `data/failed_search_questions.json` | `results/remediation/retry_audit.json`: before/after flag per retried question | report `true_quota_fixed` and `arg_error_before_and_after`; replace "114 returned HTTP 429" with the real numbers | **code ready** (CPU, seconds) |
| A6 | Separate verified / unverified / limitations / deviations | `README.md`, `report/REPORT.txt` | this plan | sections present | each reported number tagged with its evidence status | **done locally** (rebuild the PDF from `REPORT.txt` before sharing) |

## B. Scientific controls before any strong research claim

Run in this order; B1 to B4 are CPU-only on files already in Colab.

| # | Control | Command | Supports the claim if | Weakens it if |
| --- | --- | --- | --- | --- |
| B1 | Correctness-stratified decomposition | `python src/remediation.py stratify --hidden data/clean/hidden.npz --judgments data/judgments.jsonl --out results/remediation/stratify_clean.json` (also on full) | POST beats PRE **inside need = 1** (where utility equals correctness) and POST->utility clearly beats the product baseline P(need given PRE) x P(correct given POST) | the product baseline matches POST->utility: the +0.15 is need probing plus correctness probing |
| B2 | Text-only baselines on the same snippets | `python src/remediation.py textonly --runs runs --gt ../ToCall_or_NotToCall/data/BFCL.csv --judgments data/judgments.jsonl --hidden data/clean/hidden.npz` | POST probe (0.75) above TF-IDF, length and overlap baselines | a text model reaches about 0.75 |
| B3 | Multi-seed + permutation control | `python src/remediation.py seeds --hidden data/clean/hidden.npz --judgments data/judgments.jsonl --n-seeds 10` | gain > 0 in all or nearly all seeds; permuted labels centre on 0.5 | gain crosses 0 in several seeds |
| B4 | Paper-style aggregation | `python src/remediation.py foldmean ...` | n/a (explains A2) | n/a |
| B5 | Repair the 60 argument-error searches | `src/repair_failed_searches.py` prepare / run / merge, then judge, extract, `make_results.py` (steps in its docstring) | full-set gain with no exclusions stays near the clean-set gain | gain shrinks when every question has real results |
| B6 | Mismatched-evidence control | `src/control_experiments.py build-mismatch` / `run-mismatch` (GPU) | POST AUROC falls toward PRE with another question's snippets | POST AUROC stays high: format/length drives it |
| B7 | Human labels | fill the `YOUR_FINAL_verdict` column of `results/crosscheck_labels.csv` yourself after reading each answer; today it equals the model's suggestion for all 48 rows, so it is **not** a human check | human vs judge agreement high | systematic judge error |
| B8 | Chat-harness rerun | Experiment 2 in `EXPERIMENTS.md` (GPU) | argument-error rate near 0 and results similar | results depend on the harness |

Note on B1: the earlier audit suggested comparing PRE and POST "within correct and within incorrect tool answers". That split is degenerate: utility = correct x need, so among incorrect tool answers utility is always 0. The valid stratum is need = 1, where utility equals correctness; `stratify` checks this identity and reports it.

## C. Safe limitations to disclose rather than solve now

- One model (Llama-3.2-3B), one task (BFCL), one engine (Google), one seed for the headline.
- Qwen2.5-7B judge instead of GPT-4o; correctness checked by a second LLM on 48 answers, not by humans.
- Base loop instead of the paper's chat-template harness.
- Live web search: snippets from October 2026 cannot be reproduced exactly; raw runs are not public because they contain API-key URLs.
- `BFCL.csv` contains 6 repeated question strings (314 rows, 308 distinct); duplicates collapse in every per-question table.
- Time-sensitivity split (0.41 vs 0.41), `full314_valid194.json` and `pilot50_base_harness.json` have no regenerating code or superseded raw data; report them as exploratory.
- The POST probe is a diagnostic, not a deployable pre-call controller.

## What you need to do

1. Run the cells in `scripts/colab_export_raw.py` in Colab (cell C must print `files with possible keys: 0`), then unzip the download into the local repo. Or, by hand: from Colab or Drive, download `data/hidden.npz`, `data/judgments.jsonl`, `data/clean/` (if kept) and `runs/*.jsonl` into the local repo's `data/` and `runs/` (both are gitignored). Do not commit them.
2. Run B1 to B4 and `make_results.py` (CPU). Or put the files there and ask me to run them.
3. Then GPU items in Colab: A4 re-judge, B5, B6, B8.
4. Push the repo once README numbers match `results/generated/comparison.json`.
