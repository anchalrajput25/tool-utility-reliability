# Why our pre-search utility AUROC (0.60) is below the paper's LUE_x (0.72)

Status, 24 Oct 2026: **the gap is not yet explained.** Below are the differences we can document, the direction each could push the number, and the command that tests it. Nothing in this file is a measured effect unless it says so.

Paper numbers: *To Call or Not to Call* (arXiv:2605.00737), Table 4, Llama-3.2-3B, BFCL, out-of-fold AUROC, mean ± SD over five stratified folds: **LNE 0.812 ± 0.039, LUE_x 0.723 ± 0.024, LUE_x,dF 0.680 ± 0.066.**

Ours (`results/clean229.json`, `results/full314_final.json`, pooled out-of-fold AUROC, seed 42): **LNE 0.904 / 0.914, PRE utility 0.598 (clean, n = 229) / 0.681 (full, n = 308).**

| Factor | Paper | This project | Could move LUE_x because | Status / test |
| --- | --- | --- | --- | --- |
| Aggregation | mean of 5 per-fold AUROCs (± SD) | one AUROC over pooled out-of-fold scores | pooled AUROC mixes scores from 5 differently calibrated models and is usually a bit lower than the fold mean | **testable now on CPU:** `python src/remediation.py foldmean --hidden data/hidden.npz --judgments data/judgments.jsonl` (full) and on `data/clean/hidden.npz` |
| Sample | 314 BFCL rows (whether duplicates were merged is not stated) | 308 unique questions (full) and 229 (clean). `BFCL.csv` has 314 rows but only 308 distinct question strings (verified); all our tables key on the question string, so duplicates collapse | fewer, and different, items; the clean set removes the 60 argument-error runs and 22 context-dependent questions | full vs clean `foldmean` separates this |
| Judge | GPT-4o, paper's BFCL prompt | same prompt on Qwen2.5-7B-Instruct, 4-bit, temperature 1.0, 80-token output cap | a stricter judge (3 of 4 second-model disagreements were judge false negatives) changes which items are utility-positive | needs `OPENAI_API_KEY`: `python src/judge_repeat.py --backend openai ...` on both conditions, then rerun `foldmean` |
| Harness | trained harness on the native chat template (paper Sec. 4, App. B.4) | base local loop (`AGENT_HARNESS` empty). Reason recorded in our first `generate.sh` and in the 13 Oct pilot (docs/PROVENANCE.md): with the chat harness Llama-3.2-3B emitted raw tool-call JSON at answer time instead of an answer, and pilot accuracy was 0.24 | changes the always-search answers (accuracy 0.40 full / 0.50 clean vs paper 0.65) and therefore the utility labels; the PRE (no-tool) state should be similar | GPU: Experiment 2 in `EXPERIMENTS.md` |
| Label base rates | not reported in Table 4 | need 0.881, utility 0.317 (full) | the same probe can score differently on a different positive rate | reported by `foldmean` (`positive_rate`) |
| Search date | authors' runs | October 2026; 114 searches flagged by a broad filter ("429" or "error" in the prompt) re-run about 2 h later with a new key. The filter also catches argument-error runs, so how many true quota failures were fixed is unverified (`src/retry_audit.py`) | different snippets change always-search answers, hence labels | not reversible; disclose |
| Randomness | seed 42 | seed 42, one fold split | paper's fold SD is 0.024; one split can move by a few hundredths | `python src/remediation.py seeds ...` |

What we will write once the CPU tests have run: "Matching the paper's aggregation gives X (paper 0.72); the remaining difference is consistent with our different judge, harness and item set, which we have not isolated." Until then the honest sentence is: "Our PRE utility probe (0.60 clean, 0.68 full) is below the paper's LUE_x (0.72 ± 0.02); we use a different judge, harness, item set and aggregation, and have not yet isolated which causes the gap. The +0.15 is measured against our own PRE probe, not the paper's."
