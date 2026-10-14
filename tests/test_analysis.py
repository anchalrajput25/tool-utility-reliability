"""Sanity tests on synthetic data with known ground truth.
Run: python -m pytest -q tests/   (or python tests/test_analysis.py)
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
from reliability import (score_matrix, need_labels, utility_labels, flip_rate, fleiss_kappa,
                         empirical_ceiling_auroc, utility_variance_components, reliability_report)
from probes import oof_probe_scores, paired_bootstrap_auc_diff, gap_closure, budgeted_gain, oracle_gain


def synth_judgments(n=300, k=3, judge_flip=0.0, engines=("google", "brave"), engine_flip=0.0, seed=0):
    """True no-tool / always-tool correctness, then observed scores with noise."""
    rng = np.random.default_rng(seed)
    true_nt = (rng.random(n) < 0.2).astype(int)
    true_at = (rng.random(n) < 0.65).astype(int)
    rows = []
    for q in range(n):
        for rep in range(k):
            flip = lambda v: int(v ^ (rng.random() < judge_flip))
            rows.append(("no_tool", f"q{q}", rep, flip(true_nt[q])))
        for e in engines:
            at = int(true_at[q] ^ (rng.random() < engine_flip))   # engine changes the answer
            for rep in range(k):
                rows.append((f"force_{e}", f"q{q}", rep, int(at ^ (rng.random() < judge_flip))))
    return pd.DataFrame(rows, columns=["condition", "question", "rep", "score"])


def test_noiseless_labels_are_perfectly_reliable():
    j = synth_judgments(judge_flip=0.0)
    nt, at = score_matrix(j, "no_tool"), score_matrix(j, "force_google")
    for lab in (need_labels(nt), utility_labels(nt, at)):
        assert flip_rate(lab) == 0.0
        assert abs(fleiss_kappa(lab) - 1) < 1e-9
        assert abs(empirical_ceiling_auroc(lab) - 1) < 1e-9


def test_judge_noise_hits_utility_harder_than_need():
    # utility combines two noisy scores, so it should flip more than need
    j = synth_judgments(judge_flip=0.08, n=2000)
    nt, at = score_matrix(j, "no_tool"), score_matrix(j, "force_google")
    fn, fu = flip_rate(need_labels(nt)), flip_rate(utility_labels(nt, at))
    assert fu > fn, (fn, fu)
    assert empirical_ceiling_auroc(utility_labels(nt, at)) < 1.0


def test_variance_split_attributes_noise_to_correct_source():
    judge_only = utility_variance_components(synth_judgments(judge_flip=0.1, n=1000), ["google", "brave"])
    search_only = utility_variance_components(synth_judgments(engine_flip=0.2, n=1000), ["google", "brave"])
    assert judge_only["judge_share"] > 0.9, judge_only
    assert search_only["search_share"] > 0.9, search_only


def test_post_probe_with_more_information_wins_and_ci_detects_it():
    rng = np.random.default_rng(1)
    n, d = 300, 16
    y = (rng.random(n) < 0.4).astype(int)
    X_pre = rng.normal(size=(n, d)); X_pre[:, 0] += 0.4 * y        # weak signal
    X_post = X_pre.copy(); X_post[:, 1] += 1.5 * y                  # tool output adds signal
    small = {"mlp__hidden_layer_sizes": [()], "mlp__learning_rate_init": [1e-3]}
    s_pre = oof_probe_scores(X_pre, y, grid=small)
    s_post = oof_probe_scores(X_post, y, grid=small)
    res = paired_bootstrap_auc_diff(y, s_pre, s_post, n_boot=500)
    assert res["diff"] > 0.1 and res["significant"], res
    assert 0 < gap_closure(res["auc_a"], res["auc_b"], 1.0) <= 1


def test_budget_oracle_upper_bounds_any_policy():
    rng = np.random.default_rng(2)
    delta = rng.choice([-1, 0, 1], size=200, p=[0.05, 0.55, 0.40]).astype(float)
    for b in (0.2, 0.4, 0.8):
        assert budgeted_gain(rng.random(200), delta, b) <= oracle_gain(delta, b) + 1e-9


def test_report_runs_end_to_end():
    df = reliability_report(synth_judgments(judge_flip=0.05, n=200))
    assert set(df["label"]) == {"need", "utility"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
