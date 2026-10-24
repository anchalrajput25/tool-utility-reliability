# Colab export cells for the raw files of tool-utility-reliability.
# Paste each "# %% CELL" block into its own Colab cell and run them in order.
# No cell prints a file's contents or the API key; they print only paths, sizes and counts.

# %% CELL A: mount Drive and list what exists
from google.colab import drive
drive.mount('/content/drive')
import os
D = '/content/drive/MyDrive/tool-utility-reliability'
for root, dirs, files in os.walk(D):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"{os.path.getsize(p)/1e6:8.2f} MB  {os.path.relpath(p, D)}")

# %% CELL B: copy the wanted files to /content/export and redact API keys
import os, re, glob, shutil
D = '/content/drive/MyDrive/tool-utility-reliability'
OUT = '/content/export/tool-utility-reliability'
shutil.rmtree('/content/export', ignore_errors=True)

want = [
    'runs/no_tool.jsonl', 'runs/force_google.jsonl', 'runs/det_a.jsonl', 'runs/det_b.jsonl',
    'runs/force_google_with429.jsonl',          # force_google BEFORE the retry merge
    'data/judgments.jsonl', 'data/hidden.npz',
    'data/excluded_questions_clean.json',       # clean-set exclusion list (14 Oct)
    'data/failed_search_questions.json',        # the 114 retried questions (broad filter)
    'data/judgments_valid.jsonl', 'data/hidden_valid.npz',   # interim 194-question files
]
want += [os.path.relpath(p, D) for p in glob.glob(D + '/runs/fg_retry/**/*.jsonl', recursive=True)]
# Colab-side results and the human-check sheets go under data/ so git ignores them
extra = {os.path.relpath(p, D): 'data/colab_' + os.path.relpath(p, D)
         for p in glob.glob(D + '/results*/**/*', recursive=True) if os.path.isfile(p)}
for f in ['human_check_BLIND.csv', 'human_check_with_judge.csv']:
    extra[f] = 'data/human_check/' + f
plan = {rel: rel for rel in want}
plan.update(extra)

KEY_URL = re.compile(r'(api_key=)[A-Za-z0-9_\-]+')         # ...&api_key=XXXX in request URLs
KEY_JSON = re.compile(r'("api_key"\s*:\s*")[^"]*(")')       # "api_key": "XXXX" if it ever appears
HEX64 = re.compile(r'\b[0-9a-f]{64}\b')                     # bare SerpAPI-style key

def redact(text):
    text, a = KEY_URL.subn(r'\1REDACTED', text)
    text, b = KEY_JSON.subn(r'\1REDACTED\2', text)
    text, c = HEX64.subn('REDACTED_HEX64', text)
    return text, a + b + c

missing, counts = [], {}
for src_rel, dst_rel in plan.items():
    src, dst = os.path.join(D, src_rel), os.path.join(OUT, dst_rel)
    if not os.path.exists(src):
        missing.append(src_rel); continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src_rel.endswith(('.jsonl', '.json', '.csv', '.txt')):
        text, n = redact(open(src, encoding='utf-8').read())
        open(dst, 'w', encoding='utf-8').write(text)
        counts[dst_rel] = n
    else:
        shutil.copy2(src, dst)
print('not on Drive:', missing)
print('redactions per file:', {k: v for k, v in counts.items() if v})

# %% CELL C: verify nothing key-like is left and every JSONL line still parses
import os, re, json, numpy as np
from google.colab import userdata
OUT = '/content/export/tool-utility-reliability'
try:
    cur = userdata.get('SERPAPI_API_KEY') or ''    # current key; the OLD key is caught by the regex scans
except Exception:
    cur = ''
problems = 0
for root, _, files in os.walk(OUT):
    for f in files:
        p = os.path.join(root, f); rel = os.path.relpath(p, OUT)
        if f.endswith('.npz'):
            z = np.load(p, allow_pickle=True)
            for k in z.files:
                if z[k].dtype == object or z[k].dtype.kind in 'US':
                    s = ' '.join(map(str, z[k].ravel()))
                    if 'api_key=' in s or re.search(r'\b[0-9a-f]{64}\b', s) or (cur and cur in s):
                        problems += 1; print('LOOK AT', rel, 'array', k)
            continue
        t = open(p, encoding='utf-8', errors='ignore').read()
        n = len(re.findall(r'api_key=(?!REDACTED)', t)) + len(re.findall(r'\b[0-9a-f]{64}\b', t)) + (t.count(cur) if cur else 0)
        if n:
            problems += 1; print('LOOK AT', rel, n, 'hits')
        if f.endswith('.jsonl'):
            for line in t.splitlines():
                if line.strip(): json.loads(line)
print('files with possible keys:', problems, '(must be 0 before you download)')

# %% CELL D: rebuild data/clean/ (the 14 Oct version lived only in /content/clean and is lost)
import json, os, numpy as np
OUT = '/content/export/tool-utility-reliability'
drop = set(json.load(open(OUT + '/data/excluded_questions_clean.json')))
os.makedirs(OUT + '/data/clean', exist_ok=True)
with open(OUT + '/data/clean/judgments.jsonl', 'w') as f:
    for line in open(OUT + '/data/judgments.jsonl'):
        if line.strip() and json.loads(line)['question'] not in drop:
            f.write(line.rstrip('\n') + '\n')
H = dict(np.load(OUT + '/data/hidden.npz', allow_pickle=True))
keep = np.array([q not in drop for q in H['questions']])
np.savez(OUT + '/data/clean/hidden.npz', **{k: (v[keep] if len(v) == len(keep) else v) for k, v in H.items()})
json.dump(sorted(drop), open(OUT + '/data/clean/excluded_questions.json', 'w'), indent=1)
print('excluded:', len(drop), '| kept with hidden states:', int(keep.sum()), '(14 Oct reported 229)')

# %% CELL E: zip and download
import os, shutil
from google.colab import files
z = shutil.make_archive('/content/tool-utility-reliability-raw', 'zip', '/content/export')
print(round(os.path.getsize(z) / 1e6, 1), 'MB')
files.download(z)
