"""Synthetic correctness test for control_experiments (CPU, no model, no real data).

Proves the file-level analysis code runs end-to-end and behaves sensibly on
data with a KNOWN structure:
  - answer-correctness `at` is encoded strongly in POST states, weakly in PRE,
  - utility = (at > nt) differs from correctness only where nt = 1,
  - gold-in-snippets is correlated with correctness.
Expected: POST predicts utility and correctness well; PRE predicts utility less
well; the lexical AUROC is clearly above chance. This tests the plumbing and
the metric wiring, NOT any scientific claim.
"""
import json, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import control_experiments as ce  # noqa: E402

def _build(tmp, n=140, d=48, seed=1):
    rng = np.random.default_rng(seed)
    at = rng.binomial(1, 0.5, n)           # force-search answer correct
    nt = rng.binomial(1, 0.12, n)          # no-tool correct (mostly 0 -> high need)
    util = (at > nt).astype(int)
    sig = rng.normal(size=d)
    post = at[:, None] * sig * 1.6 + rng.normal(size=(n, d))        # strong
    pre = at[:, None] * sig * 0.5 + rng.normal(size=(n, d)) * 1.3   # weak
    qs = [f"q{i}" for i in range(n)]
    np.savez(tmp / "hidden.npz", questions=np.array(qs, dtype=object), pre=pre, post=post)

    with open(tmp / "judgments.jsonl", "w") as f:
        for i, q in enumerate(qs):
            f.write(json.dumps({"condition": "no_tool", "question": q, "rep": 0, "score": int(nt[i])}) + "\n")
            f.write(json.dumps({"condition": "force_google", "question": q, "rep": 0, "score": int(at[i])}) + "\n")

    gold_present = np.where(rng.random(n) < 0.15, 1 - at, at)  # correlated with correctness
    runs = tmp / "runs"; runs.mkdir(exist_ok=True)
    gt_lines = []
    with open(runs / "force_google.jsonl", "w") as f:
        for i, q in enumerate(qs):
            tok = f"zzqk{i}"                      # unique gold token
            snip = f"Search results: {tok} relevant fact." if gold_present[i] else "Search results: unrelated filler text."
            fp = f"Question: describe entity {i}. {snip} Answer:"
            f.write(json.dumps({"query": q, "final_prompt": fp, "tool_result": snip}) + "\n")
            gt_lines.append(f'"{q}","{tok}"')
    (tmp / "gt.csv").write_text("question,correct_response\n" + "\n".join(gt_lines) + "\n")
    return qs

def test():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); _build(tmp)
        dec = ce.run_decompose(tmp / "hidden.npz", tmp / "judgments.jsonl", out=None)
        lex = ce.run_lexical(tmp / "runs", tmp / "gt.csv", tmp / "judgments.jsonl", out=None)
        mb, rep = ce.build_mismatch(tmp / "runs", out_prompts=str(tmp / "mm.jsonl"))

        # plumbing
        for k in ["auroc_POST_to_utility", "auroc_POST_to_correctness", "auroc_PRE_to_utility"]:
            assert dec[k] is not None and 0.0 <= dec[k] <= 1.0, (k, dec[k])
        assert lex["auroc_gold_in_snippets_for_utility"] is not None
        assert rep["built"] >= 1 and rep["skipped_no_splice"] == 0, rep
        # sanity on the KNOWN structure
        assert dec["auroc_POST_to_utility"] > dec["auroc_PRE_to_utility"], dec
        assert dec["auroc_POST_to_correctness"] > 0.7, dec
        assert lex["auroc_gold_in_snippets_for_utility"] > 0.6, lex
        print("PASS")
        print("decompose:", json.dumps({k: dec[k] for k in dec if k.startswith("auroc") or k.endswith("frac")}))
        print("lexical:", lex["auroc_gold_in_snippets_for_utility"], "| mismatch built:", rep["built"])

if __name__ == "__main__":
    test()
