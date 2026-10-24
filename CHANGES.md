# Change log

## Uncommitted, 24 Oct 2026 (on top of local commit c554337; not pushed)

Nothing in `results/*.json`, `runs/` or `data/` was modified or deleted. No new number on real data was produced; every new script was tested on synthetic data only.

New files
- `docs/REMEDIATION_PLAN.md`: A/B/C plan with files, inputs, outputs and checks; what was and was not accessible.
- `docs/PAPER_COMPARISON.md`: documented differences behind PRE 0.60 vs paper LUE_x 0.72, each with its test.
- `src/remediation.py`: `stratify`, `textonly`, `seeds`, `foldmean`.
- `src/make_results.py`: rebuilds `clean229.json`, `full314_final.json`, `judge_crosscheck.json` from raw data into `results/generated/`, saves per-question out-of-fold scores, writes `comparison.json`; exits 1 on any mismatch; refuses to write into `results/`.
- `src/judge_audit.py`: counts judge outputs that failed to parse; reports `not_countable` for files without raw text.
- `src/repair_failed_searches.py`: prepare / run / merge for the 60 argument-error searches; maps web_search argument names to `query` (disclosed deviation); writes only to new paths and tags each repaired row.
- `results/crosscheck_labels.csv`: copy of `human_check_SUGGESTED.csv` (second-model labels for the 50-answer cross-check).
- `tests/test_remediation_synth.py`: synthetic tests for all of the above.

Modified files
- `src/local_judge.py`: also writes `raw`, `parse_ok`, `max_new_tokens` per judgment; new `--max-new-tokens` (default 80 = reported run). Scores are computed exactly as before.
- `src/free_judge.py`: new `parse_ok()` helper; `_parse()` unchanged.
- `src/diagnostics.py`: new `flag_structured()` (reads the recorded tool result) beside the unchanged `flag()`; `--flagger` (default `legacy`, which reproduces the reported clean set); `diagnostics.json` lists flag counts and disagreements; new `exclusions.csv` with the reason for every excluded question.
- `README.md`: paper Table 4 column; softened Q2 claim; "Evidence status" (verified / not verified / deviations); 308 distinct questions; full-set inflation caveat; make_results instructions.
- `report/REPORT.txt`: same corrections; fixed "~97% agreement" (it was Fleiss kappa 0.97, not percent agreement); new Sec. 7 Evidence status. `report/Report_ToolUtility.pdf` was NOT rebuilt and is now out of date.
- `EXPERIMENTS.md`: pointer to the new analyses. `.gitignore`: `runs_*/` (repaired runs contain API URLs).

Test results (synthetic only): `tests/test_analysis.py` 6/6, `tests/test_control_synth.py` PASS, `tests/test_remediation_synth.py` 6/6.

## 24 Oct 2026 (later): provenance record from the Colab chat

New files
- `docs/PROVENANCE.md`: the Colab chat's own dated log and verbatim Colab cells, copied unchanged (no key strings found in it).
- `scripts/colab_export_raw.py`: Colab cells that copy the raw files from Drive, redact API keys, check that none remain, rebuild `data/clean/` and zip it for download.
- `src/retry_audit.py` (+ test): checks what the 14 Oct retry fixed, before vs after, with both flaggers.

Findings from the provenance record, recomputed here where possible
- Cross-check: with the Qwen rep-0 vector given in PROVENANCE C8 and `results/crosscheck_labels.csv`, n = 50 (23 → 1, 37 → 0) gives 45/50 = 0.900, κ 0.799 (matches the original 14 Oct record); n = 48 (23, 37 dropped) gives 44/48 = 0.917, κ 0.832. Disagreements: ids 1, 15, 26, 50, plus 23 in the n = 50 version. The 48-answer version was chosen after the first result; README and REPORT now show both.
- "114 searches returned HTTP 429" came from a filter on "429" or "error" in the final prompt (PROVENANCE C2), so it includes the tool-argument-error runs. The retry's "still failing" check looked only for "429" (C1), so "112 of 114 fixed" is not verified. README, REPORT and PAPER_COMPARISON now say so. The notes inside `results/full314_valid194.json` and `results/full314_final.json` are left unchanged as original records.
- Pilot 1 (13 Oct, chat-template harness) returned raw tool-call JSON as answers, accuracy 0.24; this is the recorded reason for the base loop.
- `src/diagnostics.py` was never run end to end; the 60 / 2 / 22 counts come from Colab cell C5.

## 24 Oct 2026 (evening): results on the real data, new report

- Raw run files pulled from the Colab Drive folder with the Drive connector into the gitignored `data/`, `runs/`, `runs_before_retry/`; API keys redacted (92 occurrences, all in `force_google_with429.jsonl`). Originals on Drive untouched.
- `make_results.py`: 74 of 74 regenerable numbers match; exclusion list identical to Colab's; out-of-fold scores match Colab's to 1e-4.
- Controls run (`results/remediation/`): fold mean, 10 seeds + permutation, text-only baselines, correctness-stratified decomposition, retry audit.
- Main claim changed: the post-search gain (+0.15) is robust but is explained by need × answer correctness.
- Corrected: 46 quota errors (40 recovered), not 114/112; legacy "2 quota" runs are false positives; 9 empty-result runs.
- New `scripts/make_figures.py` and figures `fig1_auroc`, `fig2_seeds`, `fig3_decomposition`, `fig4_outcomes` (PDF + PNG), all drawn from result JSONs. `figures/fig2_accuracy.png` is the old hand-drawn figure and is no longer used.
- Report rewritten as LaTeX (`report/report.tex` → `report/Report_ToolUtility.pdf`); `REPORT.txt` now points to it. README rewritten around the new results.
