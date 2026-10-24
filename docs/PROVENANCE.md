<!-- Copied unchanged on 24 Oct 2026 from the Colab chat that ran the experiments. It is that chat's own record, not re-verified; numbers it marks "not sure" are still unverified. Recomputations against it: CHANGES.md. -->

# Provenance log: tool-utility-reliability

This file records how every reported number was produced. It was rebuilt on 24 Oct 2026 from the chat history (which includes the exact code that was typed into Colab) and from the files in this repo.

- Times are IST.
- "Not sure" marks anything the record does not settle.
- Colab notebook: `https://colab.research.google.com/drive/1vO67kE1HWL4KtLxz7RlT7b8LH8IFwSvw`.
- Drive folder: `MyDrive/tool-utility-reliability/`.

**Important:** one scratch cell in the Colab notebook was overwritten again and again (DOM cell 17, Monaco model 19). The notebook therefore keeps only its last content. The scratch code exists only in the chat, and it is copied verbatim in section C below.

---

## A. Dated log

### Before 13 Oct: setup decisions

- **Model:** `meta-llama/Llama-3.2-3B-Instruct` returned 403 (gated), so the ungated copy `unsloth/Llama-3.2-3B-Instruct` was used. It has the same weights.
- **Judge:** there was no OpenAI key, so the paper's GPT-4o judge could not be used. The Gemini free tier was tried and was too limited (about 20 requests/day). Final choice: the paper's exact BFCL judge prompt run on `Qwen/Qwen2.5-7B-Instruct`, 4-bit (bitsandbytes), temperature 1.0, K = 3 repeats, `max_new_tokens=80`, seed `1000+rep` (`src/local_judge.py`).
- Exact dates of these decisions: not sure.

### 13 Oct: pilots (50 questions)

- **Pilot 1:** run with `AGENT_HARNESS=chat` (chat-template harness).
  - Answers came out as raw tool-call JSON, and force_google accuracy was 0.24.
  - Discarded. The repo README says open vLLM models should use the base local loop.
  - Time: not sure.
- **Pilot 2 (corrected):** finished by about 22:20.
  - `AGENT_HARNESS=` (empty = base loop), `MODEL=unsloth/Llama-3.2-3B-Instruct`, `LIMIT=50`, `--temperature 0`, `--skip-scorer`.
  - Search: `--web-search --force-search --search-provider google` (SerpAPI).
  - Judge as above, K = 3.
  - Results: determinism 20/20; no-tool 0.18; Google 0.56; utility κ 0.973.
  - Saved as `results/pilot50_base_harness.json`.
- Afterwards the Colab runtime was lost when the laptop slept, and the run files were gone.

### 14 Oct, about 11:20–11:40: setup and an accidental pilot re-run

- The setup cell was rewritten to embed the code (a base64 tarball of `src/` and `scripts/`). No upload was needed.
- By mistake, the old 50-question pilot cell was run again (about 10 minutes lost). Its outputs were deleted by the next step.

### 14 Oct, about 11:55–13:40: full run (Colab "FULL RUN" cell)

```
%cd /content/tool-utility-reliability
!rm -rf runs data/judgments.jsonl data/hidden.npz results
%env LIMIT=314
%env MODEL=unsloth/Llama-3.2-3B-Instruct
%env AGENT_HARNESS=
!bash scripts/generate.sh 2>&1 | tail -15
!bash scripts/judge_all.sh 2>&1 | tail -8
!python src/extract_hidden.py --runs runs --model unsloth/Llama-3.2-3B-Instruct --out data/hidden.npz 2>&1 | tail -5
!python src/analyze.py --runs runs --judgments data/judgments.jsonl --hidden data/hidden.npz --out results
```

- `generate.sh` produced four runs:
  - `det_a` and `det_b`: no tool, 20 questions each (determinism check).
  - `no_tool`: 314 questions.
  - `force_google`: 314 questions.
  - All runs: temperature 0, `--skip-scorer`.
- `judge_all.sh` used `JUDGE=local` and `K=3`.
- Generation finished at about 12:56 (314/314). Judging, hidden states and analysis followed.

### 14 Oct, about 13:40–14:00: SerpAPI quota problem found

- The free plan (250 searches/month) ran out during the full run.
- The SerpAPI key appeared once in a Colab output. That output was cleared, and rotating the key was recommended.
- **Interim analysis on 194 questions** (`results/full314_valid194.json`): questions were dropped when their `final_prompt` contained `'429'` **or** the substring `'error'`.
  - **Caveat (not sure):** this filter also catches tool-argument-error runs, and any snippet that contains the word "error".
  - So "114 searches returned HTTP 429", as stated in the summary and chat, is the count from this broad filter. It is not a verified count of true 429 responses.
  - Results: pre 0.597 → post 0.735, +0.138 [0.045, 0.232]; utility κ 0.986.
- About 14:13: `runs data results results_valid src scripts` were copied to Drive.

### 14 Oct, about 14:30–15:05: retry with a new SerpAPI key, then merge

- The SerpAPI key was replaced in Colab Secrets (from a second account). The key was checked without being printed: 250 searches were left.
- The cell is in section C1. In short:
  - The 114 questions from `data/failed_search_questions.json` were written to `BFCL_retry.csv`.
  - The original run was copied to `runs/force_google_with429.jsonl`.
  - `main.py` was re-run on those questions with the same flags (temperature 0, `--skip-scorer`, `--web-search --force-search --search-provider google`), with output in `runs/fg_retry/`.
    - Harness: the same kernel was used, so `AGENT_HARNESS=` was most likely still set. Not sure, because it was not re-set inside this cell.
  - Rows were merged by `query` into `runs/force_google.jsonl`.
  - The force_google judgments for retried questions were deleted and re-judged (cached local judge, K = 3).
  - Then: hidden states re-extracted, `analyze.py` re-run, and `runs data results` copied to Drive.
- Retry outcome: 112 of 114 fixed; 2 still contained `429`.
- Full-set result: `results/full314_final.json`.
  - 308 questions had both prompts and went into the probes.
  - Why exactly 6 of the 314 lack both prompts: not sure. Two are the still-failing searches.
- The original searches ran around 12:00–13:00 and the retries around 14:30, all on 14 Oct.

### 14 Oct, about 18:40–19:10: diagnostics and clean set, after the GPT review

- **Time-sensitivity test** (C3): rejected as the cause.
  - Accuracy: 0.41 on current/latest questions, 0.37 on fixed-year questions, 0.41 on stable questions.
  - Group sizes were printed in Colab; I do not have them. Not sure.
- **Gold-in-context check** (C4): run on all 314 force_google rows. Gold present in 148 (accuracy 0.74), absent in 166 (accuracy 0.09).
  - The absent group includes the 60 tool-argument-error runs, which received no results.
- **Failure types** (C5):
  - `tool_arg_error`: 60 runs (accuracy 0.08). The model called `web_search` with `search_query`, and the harness returned an error string.
  - `quota_429`: 2 runs.
  - Context-dependent pronoun questions (regex, fewer than 9 words): 22 (accuracy 0.09).
- **Clean-set rule:** decided at about 18:50–19:00 on 14 Oct, after the accuracy by failure type had been seen.
  - The rule is **post hoc**.
  - Rule: drop a question if it has a tool-argument error, a 429 error, or is a context-dependent pronoun question.
  - Result: 232 questions; 229 with hidden states.
  - The excluded list was saved to Drive as `data/excluded_questions_clean.json`.
  - The filtered judgments and hidden states were written to `/content/clean/` (**temporary, now lost**).
  - `analyze.py` was run on them with `--out results_clean` on Drive.
- **Per-AUROC CIs** (C7): about 19:15, saved to Drive as `results_clean/oof_scores.npz`.
- **Human-check sample** (C6): 50 rows, written to Drive.

### 14 Oct, about 19:50: second-model cross-check (NOT human)

- Claude proposed verdicts for the 50 rows. Rows 23 and 37 were decided after reading the full answers in Colab.
- These were compared with the Qwen judge's rep-0 verdicts (C8).
- Result: agreement 0.90, Cohen κ 0.799; 4 judge false negatives and 1 false positive.
- Saved as `results/judge_crosscheck.json`.
- **No 92% / κ 0.83 figure was produced in this chat.** If that number appears anywhere, it does not come from this record.

### 14 Oct, after the cross-check: no new experiments

- `src/diagnostics.py` was written at about 19:20 to package C5 and the clean filter. **It was never run on real data.**
- Figures were drawn from hard-coded numbers.
- The report PDF and repo were built from the numbers above.
- I found no Colab runs after 14 Oct in the chat. Not fully sure for 15–19 Oct, but the record shows only writing and application work after that.

---

## B. How each reported number was computed

Label definitions:
- **Need** = no-tool answer wrong.
- **Utility** = force_google score > no-tool score, on the rep-*j* pair.
- **Headline labels** = rep 0 ("single_run" in `analyze.py`).

| Number | Computed by | Inputs | In repo? |
|---|---|---|---|
| AUROC 0.60 / 0.75 (clean) | `analyze.py` → `probes.oof_probe_scores` (nested 5×5 CV, MLP grid, seed 42), run from Colab on the filtered data | `/content/clean/j.jsonl`, `/content/clean/h.npz` (lost; rebuild with export cell D) | Code: yes. Filtering: Colab only (C5). |
| +0.15 [0.06, 0.24] | `probes.paired_bootstrap_auc_diff` (2000 resamples, seed 0, over OOF scores; probe not refit) | same | Yes (code); run from Colab |
| Per-AUROC CIs [0.52, 0.67] / [0.68, 0.81] | Colab cell C7 (2000 resamples, seed 0, percentile) | same; scores saved in `results_clean/oof_scores.npz` | Colab only |
| Full set 0.68 → 0.82, +0.14 [0.07, 0.21] | `analyze.py` | full `data/judgments.jsonl`, `data/hidden.npz` (n = 308) | Yes |
| Fleiss κ 0.97 [0.94, 0.99] (clean) | `reliability.fleiss_kappa` + bootstrap, via `analyze.py` | clean judgments | Code: yes |
| 0.99 judge-agreement reference | `reliability.empirical_ceiling_auroc`: leave-one-repeat-out AUROC between judge repeats. Measures consistency only. | clean utility labels (0.992) | Code: yes |
| 90% / κ 0.80 cross-check | inline Python in my workspace (C8) vs judge rep 0 | `human_check_with_judge.csv` + Claude labels | Result JSON: yes. Code: no. |
| 47% / 0.74 / 0.09 | Colab cell C4: gold-answer tokens (length > 2) all present in normalized `final_prompt` | all 314 force_google rows, rep 0 | Colab only (same logic later in `diagnostics.py`, never run) |
| Time sensitivity 0.41 / 0.37 / 0.41 | Colab cell C3 (regex groups on question text) | rep-0 accuracy, all 314 | Colab only |
| Accuracy 0.12 / 0.40 (full), 0.12 / 0.50 (clean) | `analyze.py` (rep 0 means); clean value from C5 | judgments | Partly |
| Determinism 20/20 | `analyze.determinism` (det_a vs det_b) | `runs/det_a.jsonl`, `runs/det_b.jsonl` | Yes |

---

## C. Colab scratch code, verbatim from the chat

### C1. Retry 114 Google searches, merge, re-judge, re-extract, re-analyse (14 Oct, about 14:30)

```python
import os, json, glob, shutil, pandas as pd
os.chdir('/content/tool-utility-reliability')
bad=set(json.load(open('data/failed_search_questions.json')))
df=pd.read_csv('/content/ToCall_or_NotToCall/data/BFCL.csv')
df[df.question.isin(bad)].to_csv('/content/ToCall_or_NotToCall/data/BFCL_retry.csv', index=False)
print('retry questions:', df.question.isin(bad).sum())
shutil.copy('runs/force_google.jsonl','runs/force_google_with429.jsonl')
!cd /content/ToCall_or_NotToCall/src && python main.py --data ../data/BFCL_retry.csv --task bfcl --model-name unsloth/Llama-3.2-3B-Instruct --temperature 0 --skip-scorer --web-search --force-search --search-provider google --output-dir /content/tool-utility-reliability/runs/fg_retry 2>&1 | tail -4
new=[json.loads(l) for l in open(max(glob.glob('runs/fg_retry/**/*.jsonl', recursive=True), key=os.path.getmtime))]
still=[r for r in new if '429' in str(r.get('final_prompt',''))]
print('retried:', len(new), 'still failing:', len(still))
by={r['query']:r for r in new}
old=[json.loads(l) for l in open('runs/force_google_with429.jsonl')]
with open('runs/force_google.jsonl','w') as f:
    for r in old: f.write(json.dumps(by.get(r['query'], r))+'\n')
J=[json.loads(l) for l in open('data/judgments.jsonl')]
with open('data/judgments.jsonl','w') as f:
    for r in J:
        if not (r['condition']=='force_google' and r['question'] in by): f.write(json.dumps(r)+'\n')
!python src/local_judge.py --runs runs --gt /content/ToCall_or_NotToCall/data/BFCL.csv --k 3 2>&1 | grep -v Warn | tail -4
!python src/extract_hidden.py --runs runs --model unsloth/Llama-3.2-3B-Instruct --out data/hidden.npz 2>&1 | tail -2
!python src/analyze.py --runs runs --judgments data/judgments.jsonl --hidden data/hidden.npz --out results 2>&1 | grep -v AVX
!cp -r runs data results /content/drive/MyDrive/tool-utility-reliability/ && echo SAVED_TO_DRIVE
```

### C2. Interim 194-question filter (14 Oct, about 13:50; this defined `failed_search_questions.json`)

```python
import json, numpy as np
rows=[json.loads(l) for l in open('runs/force_google.jsonl')]
bad={r['query'] for r in rows if '429' in str(r.get('final_prompt','')) or 'error' in str(r.get('final_prompt','')).lower()}
J=[json.loads(l) for l in open('data/judgments.jsonl')]
with open('data/judgments_valid.jsonl','w') as f:
    for r in J:
        if r['question'] not in bad: f.write(json.dumps(r)+'\n')
H=dict(np.load('data/hidden.npz', allow_pickle=True)); keep=np.array([q not in bad for q in H['questions']])
np.savez('data/hidden_valid.npz', **{k:(v[keep] if len(v)==len(keep) else v) for k,v in H.items()})
json.dump(sorted(bad), open('data/failed_search_questions.json','w'))
print('valid questions:', int(keep.sum()), 'dropped:', len(bad))
!python src/analyze.py --runs runs --judgments data/judgments_valid.jsonl --hidden data/hidden_valid.npz --out results_valid 2>&1 | grep -v AVX
```

### C3. Time-sensitivity test (14 Oct, about 18:45)

```python
import json, pandas as pd, os
D='/content/drive/MyDrive/tool-utility-reliability'
j=pd.DataFrame([json.loads(l) for l in open(D+'/data/judgments.jsonl')])
a=j[j.rep==0].pivot_table(index='question',columns='condition',values='score',aggfunc='mean')
cur=a.index.str.contains(r'current|currently|latest|\bnow\b|recent|today|newest|as of|\bnew\b', case=False, regex=True)
yr=a.index.str.contains(r'20(2[3-6])', regex=True) & ~cur
st=~cur & ~yr
for name,mask in [('ALL',[True]*len(a)),('current/latest',cur),('fixed year 2023-26',yr),('stable',st)]:
    s=a[mask]; print(f"{name:20s} n={len(s):3d}  no_tool={s.no_tool.mean():.2f}  google={s.force_google.mean():.2f}")
```

### C4. Gold-in-context check (14 Oct, about 18:50)

```python
import json, pandas as pd, re
D='/content/drive/MyDrive/tool-utility-reliability'
gt=dict(pd.read_csv('https://raw.githubusercontent.com/QinyuanWu0710/ToCall_or_NotToCall/main/data/BFCL.csv').values)
rows=[json.loads(l) for l in open(D+'/runs/force_google.jsonl')]
j=pd.DataFrame([json.loads(l) for l in open(D+'/data/judgments.jsonl')])
sc=j[(j.condition=='force_google')&(j.rep==0)].groupby('question').score.mean()
norm=lambda s: re.sub(r'[^a-z0-9 ]',' ',str(s).lower())
out=[]
for r in rows:
    q=r['query']; g=gt.get(q); p=norm(r.get('final_prompt',''))
    toks=[t for t in norm(g).split() if len(t)>2]
    inctx = bool(toks) and all(t in p for t in toks)
    out.append(dict(q=q, gold=g, inctx=inctx, ok=sc.get(q)))
d=pd.DataFrame(out).dropna(subset=['ok'])
print('gold answer present in search context:', round(d.inctx.mean(),2))
print(d.groupby('inctx').ok.agg(['mean','count']))
```

### C5. Failure types and clean set (14 Oct, about 18:55–19:05)

```python
import json, numpy as np, pandas as pd, os, subprocess
D='/content/drive/MyDrive/tool-utility-reliability'
rows=[json.loads(l) for l in open(D+'/runs/force_google.jsonl')]
def bad(r):
    p=str(r.get('final_prompt',''))+str(r.get('response',''))
    return ('unexpected keyword' in p or 'not expecting' in p or '_execute_web_search' in p or '429' in p)
qs=pd.Series([r['query'] for r in rows])
pron=qs.str.contains(r'\b(?:his|her|he|she|they|their|its|the team|the album|the club|the company|the school|the city|this|that)\b', case=False, regex=True) & (qs.str.split().str.len()<9)
drop={r['query'] for r,p in zip(rows,pron) if bad(r) or p}
J=[json.loads(l) for l in open(D+'/data/judgments.jsonl')]
os.makedirs('/content/clean',exist_ok=True)
with open('/content/clean/j.jsonl','w') as f:
    for r in J:
        if r['question'] not in drop: f.write(json.dumps(r)+'\n')
H=dict(np.load(D+'/data/hidden.npz',allow_pickle=True)); keep=np.array([q not in drop for q in H['questions']])
np.savez('/content/clean/h.npz', **{k:(v[keep] if len(v)==len(keep) else v) for k,v in H.items()})
json.dump(sorted(drop), open(D+'/data/excluded_questions_clean.json','w'))
print('kept for probes:', int(keep.sum()))
r=subprocess.run(['python',D+'/src/analyze.py','--runs',D+'/runs','--judgments','/content/clean/j.jsonl','--hidden','/content/clean/h.npz','--out',D+'/results_clean'],capture_output=True,text=True,cwd=D+'/src')
```

(The counts by type, 60 / 2 / 22, come from an earlier version of this cell that grouped the same `bad` and pronoun flags. Accuracy per type is rep 0.)

### C6. Sample for the human check (14 Oct, about 18:55)

```python
import json, pandas as pd
D='/content/drive/MyDrive/tool-utility-reliability'
gt=dict(pd.read_csv('https://raw.githubusercontent.com/QinyuanWu0710/ToCall_or_NotToCall/main/data/BFCL.csv').values)
j=pd.DataFrame([json.loads(l) for l in open(D+'/data/judgments.jsonl')])
j0=j[j.rep==0].groupby(['condition','question']).score.mean()
rows=[]
for cond in ['force_google','no_tool']:
    for l in open(f'{D}/runs/{cond}.jsonl'):
        r=json.loads(l); q=r['query']
        if (cond,q) in j0.index: rows.append(dict(condition=cond, question=q, gold_answer=gt.get(q), model_answer=' '.join(str(r['response']).split())[:400], judge_says=int(j0[(cond,q)])))
d=pd.DataFrame(rows)
g=d[d.condition=='force_google']; n=d[d.condition=='no_tool']
s=pd.concat([g[g.judge_says==1].sample(20,random_state=1), g[g.judge_says==0].sample(20,random_state=1), n.sample(10,random_state=1)]).sample(frac=1,random_state=2).reset_index(drop=True)
s.insert(0,'id',range(1,51)); s['your_verdict_1correct_0wrong']=''; s['note']=''
s_blind=s.drop(columns=['judge_says'])
s.to_csv(D+'/human_check_with_judge.csv',index=False); s_blind.to_csv(D+'/human_check_BLIND.csv',index=False)
```

### C7. Per-AUROC bootstrap CIs, clean set (14 Oct, about 19:15)

```python
import sys, json, numpy as np, pandas as pd
D='/content/drive/MyDrive/tool-utility-reliability'; sys.path.insert(0, D+'/src')
from probes import oof_probe_scores
from reliability import score_matrix, utility_labels
from sklearn.metrics import roc_auc_score
j=pd.DataFrame([json.loads(l) for l in open('/content/clean/j.jsonl')]); j['score']=j.score.astype(int)
H=np.load('/content/clean/h.npz', allow_pickle=True); qs=list(H['questions'])
y=utility_labels(score_matrix(j,'no_tool'), score_matrix(j,'force_google')).reindex(qs)[0].values.astype(int)
sa=oof_probe_scores(H['pre'],y); sb=oof_probe_scores(H['post'],y)
rng=np.random.default_rng(0); A=[];B=[]
for _ in range(2000):
    i=rng.integers(0,len(y),len(y))
    if len(set(y[i]))<2: continue
    A.append(roc_auc_score(y[i],sa[i])); B.append(roc_auc_score(y[i],sb[i]))
print('PRE ',round(roc_auc_score(y,sa),3), np.round(np.percentile(A,[2.5,97.5]),3))
print('POST',round(roc_auc_score(y,sb),3), np.round(np.percentile(B,[2.5,97.5]),3))
np.savez(D+'/results_clean/oof_scores.npz', y=y, pre=sa, post=sb, questions=np.array(qs))
```

### C8. Second-model cross-check (my workspace, 14 Oct, about 19:50)

```python
# J = Qwen judge rep-0 verdicts for ids 1..50 (from human_check_with_judge.csv):
# 0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,1,0,1,1,0,0,1,0,1,1
# C = Claude's labels from human_check/human_check_SUGGESTED.csv, with id 23 -> 1 and id 37 -> 0
# agreement = mean(C == J) = 0.90 ; cohen_kappa_score(C, J) = 0.799
```

---

## D. Known gaps

- `data/clean/` never existed as a saved folder; it lived only in `/content/clean/`. Export cell D rebuilds it from `data/excluded_questions_clean.json` and checks that it gives 229.
- `src/diagnostics.py` has not been run end to end. Its output should be checked against C5 (232 / 229).
- "114 failed with HTTP 429" is a broad-filter count (C2), not a verified 429 count.
- The 47% / 0.74 / 0.09 numbers are over all 314 runs, so the "absent" group includes failed searches.
- Brave Search, a human-labelled check and a text-only baseline were planned but **not run**.
