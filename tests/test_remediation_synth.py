"""Synthetic tests for the remediation code (CPU, no model, no real data).

They check plumbing and known-structure behaviour only, NOT any scientific claim.
Run: python tests/test_remediation_synth.py
"""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import remediation as rm          # noqa: E402
import judge_audit as ja          # noqa: E402
import repair_failed_searches as rf   # noqa: E402
from diagnostics import flag, flag_structured   # noqa: E402
from probes import oof_probe_scores  # noqa: E402

FAST = rm.FAST_GRID


def build(tmp: Path, n=150, d=24, seed=3):
    rng = np.random.default_rng(seed)
    qs = [f"Who founded company number {i} in Paris originally?" if i % 17 else f"Where is his hometown {i}?" for i in range(n)]
    gold = [f"Founder{i}x" for i in range(n)]
    nt_true = rng.binomial(1, 0.12, n)
    present = rng.random(n) < 0.5
    at_true = np.where(present, rng.random(n) < 0.8, rng.random(n) < 0.1).astype(int)
    argerr = rng.random(n) < 0.15
    at_true[argerr] = 0
    sig = rng.normal(size=d)
    pre = (1 - nt_true)[:, None] * sig * 0.8 + rng.normal(size=(n, d))
    post = pre + at_true[:, None] * np.roll(sig, 3) * 1.5 + rng.normal(size=(n, d)) * 0.5
    np.savez(tmp / "hidden.npz", questions=np.array(qs, dtype=object), pre=pre, post=post)
    (tmp / "gt.csv").write_text("question,correct_response\n" + "\n".join(f'"{q}",{g}' for q, g in zip(qs, gold)) + "\n")
    runs = tmp / "runs"; runs.mkdir()
    with open(runs / "force_google.jsonl", "w") as f:
        for i, q in enumerate(qs):
            if argerr[i]:
                res = {"error": "Tool execution failed: _execute_web_search() got an unexpected keyword argument 'search_query'"}
                ctx = f"Search error: {res['error']}"
            else:
                body = f"1. Page\n   {gold[i] if present[i] else 'unrelated'} founded it in {1429 if i % 10 == 0 else 1999}.\n"
                res = json.dumps({"results": [{"title": "Page", "snippet": "x"}]})
                ctx = "Search Results:\n\n" + body
            fp = f"<user>Based on the following tool result, please answer the question.\n{ctx}\nQuestion: {q}<assistant>"
            f.write(json.dumps({"query": q, "final_prompt": fp, "response": "ans",
                                "tool_calls": [{"type": "tool_execution", "result": res}]}) + "\n")
    with open(runs / "no_tool.jsonl", "w") as f:
        for q in qs:
            f.write(json.dumps({"query": q, "final_prompt": f"<user>{q}<assistant>", "response": "ans"}) + "\n")
    for nm in ("det_a", "det_b"):
        with open(runs / f"{nm}.jsonl", "w") as f:
            for q in qs[:20]:
                f.write(json.dumps({"query": q, "response": "same"}) + "\n")
    with open(tmp / "judgments.jsonl", "w") as f:
        for i, q in enumerate(qs):
            for rep in range(3):
                flipn = rng.random() < 0.02; flipa = rng.random() < 0.02
                f.write(json.dumps({"condition": "no_tool", "question": q, "rep": rep, "judge_model": "m",
                                    "score": int(nt_true[i] ^ flipn)}) + "\n")
                f.write(json.dumps({"condition": "force_google", "question": q, "rep": rep, "judge_model": "m",
                                    "score": int(at_true[i] ^ flipa)}) + "\n")
    return qs, argerr


def test_fold_helper_matches_probes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6)); y = (X[:, 0] + rng.normal(size=80) > 0).astype(int)
    s1 = oof_probe_scores(X, y, grid=FAST); s2, fold = rm.oof_scores_and_folds(X, y, grid=FAST)
    assert np.allclose(s1, s2) and set(fold) == set(range(5))


def test_flags_and_overmatch():
    row_ok = {"final_prompt": "Search Results: founded in 1429", "tool_calls": [{"type": "tool_execution", "result": '{"results": []}'}]}
    assert flag(row_ok) == "quota_429"            # the legacy substring match over-fires
    assert flag_structured(row_ok) == "ok"         # the structured flag does not
    row_err = {"final_prompt": "", "tool_calls": [{"type": "tool_execution",
               "result": {"error": "Tool execution failed: _execute_web_search() got an unexpected keyword argument 'search_query'"}}]}
    assert flag_structured(row_err) == "tool_arg_error"


def test_retry_audit():
    import retry_audit as ra
    arg = {"type": "tool_execution", "result": {"error": "_execute_web_search() got an unexpected keyword argument 'search_query'"}}
    quota = {"type": "tool_execution", "result": {"error": "HTTP 429 Too Many Requests"}}
    ok = {"type": "tool_execution", "result": '{"results": [1]}'}
    before = [{"query": "a", "final_prompt": "Search error: Tool execution failed", "tool_calls": [arg]},
              {"query": "b", "final_prompt": "Search error: 429", "tool_calls": [quota]},
              {"query": "c", "final_prompt": "Search Results: fine", "tool_calls": [ok]}]
    after = [dict(before[0]), {"query": "b", "final_prompt": "Search Results: x", "tool_calls": [ok]}, before[2]]
    res = ra.audit(before, after, {"a", "b"})
    assert res["broad_filter_on_before_reproduces_list"]
    assert res["counted_fixed_by_original_check"] == 2      # the original '429'-only check calls both fixed
    assert res["actually_ok_after"] == 1 and res["arg_error_before_and_after"] == 1
    assert res["true_quota_fixed"] == 1


def test_remap():
    assert rf.remap_web_search_args({"search_query": "abc"}) == {"query": "abc"}
    assert rf.remap_web_search_args({"query": "a", "count": "5"}) == {"query": "a", "count": 5}
    assert rf.remap_web_search_args({"foo": 1}, "fallback") == {"query": "fallback"}


def test_judge_audit():
    assert ja.audit([{"condition": "c", "question": "q", "rep": 0, "score": 0}])["status"] == "not_countable"
    rows = [{"condition": "c", "question": "q1", "rep": 0, "score": 1, "raw": '{"is_correct": true, "reason": "ok"}'},
            {"condition": "c", "question": "q2", "rep": 0, "score": 0, "raw": '{"is_correct": false, "reason": "long reas'}]
    res = ja.audit(rows)
    assert res["parse_failures"] == 1 and res["of_which_truncated_json"] == 1


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); qs, argerr = build(tmp)
        st = rm.run_stratify(tmp / "hidden.npz", tmp / "judgments.jsonl", grid=FAST)
        assert st["utility_equals_correct_times_need"] is True
        assert st["need0"]["utility_positives"] == 0
        assert st["auroc_POST_to_correct"] > st["auroc_PRE_to_correct"]
        tx = rm.run_textonly(tmp / "runs", tmp / "gt.csv", tmp / "judgments.jsonl", hidden=tmp / "hidden.npz")
        assert tx["auroc"]["correct"]["REFERENCE_ONLY_gold_in_snippets"] > 0.7, tx
        sd = rm.run_seeds(tmp / "hidden.npz", tmp / "judgments.jsonl", n_seeds=2, n_perm=3, grid=FAST)
        assert len(sd["per_seed"]) == 2 and 0.3 < sd["permutation_POST_auroc"]["mean"] < 0.7
        fm = rm.run_foldmean(tmp / "hidden.npz", tmp / "judgments.jsonl", grid=FAST)
        assert len(fm["LUE_x_utility_from_PRE"]["per_fold"]) == 5

        # repair: prepare finds exactly the injected argument errors; merge tags them
        subprocess.run([sys.executable, str(ROOT / "src/repair_failed_searches.py"), "prepare", "--runs", str(tmp / "runs"),
                        "--gt", str(tmp / "gt.csv"), "--out", str(tmp / "repair")], check=True)
        ids = json.loads((tmp / "repair/failed_ids.json").read_text())
        assert ids["n_failed_rows"] == int(argerr.sum())
        fixed = tmp / "fixed.jsonl"
        with open(fixed, "w") as f:
            for q in ids["questions"]:
                f.write(json.dumps({"query": q, "final_prompt": "Search Results: x", "response": "a",
                                    "tool_calls": [{"type": "tool_execution", "result": '{"results": [1]}'}]}) + "\n")
        subprocess.run([sys.executable, str(ROOT / "src/repair_failed_searches.py"), "merge", "--runs", str(tmp / "runs"),
                        "--repaired", str(fixed), "--judgments", str(tmp / "judgments.jsonl"), "--out", str(tmp / "runs_repaired")], check=True)
        log = json.loads((tmp / "runs_repaired/repair_log.json").read_text())
        assert log["n_replaced"] == int(argerr.sum()) and log["replaced_now_ok"] == log["n_replaced"]
        assert len((tmp / "runs/force_google.jsonl").read_text().splitlines()) == len(qs)   # original untouched

        # make_results: first run defines "reported"; second run must match it exactly
        gen1, rep = tmp / "gen1", tmp / "reported"
        subprocess.run([sys.executable, str(ROOT / "src/make_results.py"), "--runs", str(tmp / "runs"),
                        "--judgments", str(tmp / "judgments.jsonl"), "--gt", str(tmp / "gt.csv"),
                        "--hidden", str(tmp / "hidden.npz"), "--reported", str(ROOT / "results"), "--out", str(gen1)],
                       capture_output=True)
        diag = json.loads((gen1 / "diagnostics/diagnostics.json").read_text())
        n429 = sum(1 for i in range(len(qs)) if i % 10 == 0 and not argerr[i])
        assert len(diag["flag_disagreements"]) == n429, (len(diag["flag_disagreements"]), n429)
        rep.mkdir(); shutil.copy(gen1 / "full314_final.json", rep / "full314_final.json")
        shutil.copy(gen1 / "clean229.json", rep / "clean229.json")
        p = subprocess.run([sys.executable, str(ROOT / "src/make_results.py"), "--runs", str(tmp / "runs"),
                            "--judgments", str(tmp / "judgments.jsonl"), "--gt", str(tmp / "gt.csv"),
                            "--hidden", str(tmp / "hidden.npz"), "--reported", str(rep), "--out", str(tmp / "gen2")])
        assert p.returncode == 0, "regeneration is not deterministic"
        c = json.loads(rep.joinpath("clean229.json").read_text()); c["clean229"]["q2"]["post"] += 0.05
        rep.joinpath("clean229.json").write_text(json.dumps(c))
        p = subprocess.run([sys.executable, str(ROOT / "src/make_results.py"), "--runs", str(tmp / "runs"),
                            "--judgments", str(tmp / "judgments.jsonl"), "--gt", str(tmp / "gt.csv"),
                            "--hidden", str(tmp / "hidden.npz"), "--reported", str(rep), "--out", str(tmp / "gen3")],
                           capture_output=True, text=True)
        assert p.returncode == 1 and "MISMATCH" in p.stdout, p.stdout


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
