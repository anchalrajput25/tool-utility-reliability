"""Draw the report and README figures from the regenerated result files.

  python scripts/make_figures.py                 # reads results/generated, results/remediation
  python scripts/make_figures.py --runs runs --judgments data/judgments.jsonl   # also refreshes outcome counts

Every plotted number is read from a JSON file written by src/make_results.py,
src/remediation.py or src/retry_audit.py; nothing is typed in by hand except the
paper's Table 4 values. Writes figures/*.pdf (for the report) and figures/*.png.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
GEN, REM, FIG = ROOT / "results/generated", ROOT / "results/remediation", ROOT / "figures"
PAPER_LUE = (0.723, 0.024)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
C_PRE, C_POST, C_REF, C_GREY = "#8c8c8c", "#1f4e79", "#b85c00", "#c9c9c9"


def load(p): return json.loads(Path(p).read_text())


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf"); fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)


def fig_auroc():
    """Main result: PRE vs POST utility AUROC with 95% CIs, clean and full."""
    sets = [("Clean (n = 229)", load(GEN / "clean229.json")["clean229"]["q2"]),
            ("Full (n = 308)", load(GEN / "full314_final.json")["q2_single_run"])]
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    w = 0.34
    for i, (lab, q) in enumerate(sets):
        for j, (k, c, name) in enumerate([("pre", C_PRE, "Pre-search state"), ("post", C_POST, "Post-search state")]):
            x = i + (j - 0.5) * w
            lo, hi = q[f"{k}_ci"]
            ax.bar(x, q[k], w * 0.92, color=c, label=name if i == 0 else None)
            ax.errorbar(x, q[k], yerr=[[q[k] - lo], [hi - q[k]]], color="black", lw=0.7, capsize=2)
            ax.text(x, hi + 0.012, f"{q[k]:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.axhspan(PAPER_LUE[0] - PAPER_LUE[1], PAPER_LUE[0] + PAPER_LUE[1], color=C_REF, alpha=0.12, lw=0)
    ax.axhline(PAPER_LUE[0], color=C_REF, lw=0.8, ls="--", label=r"Paper LUE$_x$ (mean $\pm$ SD)")
    ax.axhline(0.5, color="black", lw=0.5, ls=":")
    ax.set_xticks(range(len(sets)), [s[0] for s in sets])
    ax.set_ylim(0.45, 0.95); ax.set_ylabel("Utility AUROC")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.4, columnspacing=1.0)
    save(fig, "fig1_auroc")


def fig_seeds():
    """PRE vs POST for 10 probe seeds, paired lines."""
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.1), sharey=True)
    for ax, (tag, title) in zip(axes, [("clean", "Clean"), ("full", "Full")]):
        s = load(REM / f"seeds_{tag}.json")
        for r in s["per_seed"]:
            ax.plot([0, 1], [r["pre"], r["post"]], color=C_GREY, lw=0.7, zorder=1)
        ax.scatter([0] * 10, [r["pre"] for r in s["per_seed"]], s=10, color=C_PRE, zorder=2)
        ax.scatter([1] * 10, [r["post"] for r in s["per_seed"]], s=10, color=C_POST, zorder=2)
        ax.set_xticks([0, 1], ["Pre", "Post"]); ax.set_xlim(-0.35, 1.35)
        ax.set_title(f"{title}: $\\Delta$ = {s['diff_mean']:.2f} $\\pm$ {s['diff_sd']:.2f}")
        ax.axhline(s["permutation_POST_auroc"]["mean"], color="black", lw=0.5, ls=":")
    axes[0].set_ylabel("Utility AUROC"); axes[0].set_ylim(0.45, 0.9)
    axes[1].text(1.3, load(REM / "seeds_full.json")["permutation_POST_auroc"]["mean"] + 0.01,
                 "shuffled labels", ha="right", va="bottom", fontsize=7)
    save(fig, "fig2_seeds")


def fig_decomposition():
    """What predicts utility: text baselines, probes, and the need x correctness product."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharex=True)
    for ax, tag, title in [(axes[0], "clean", "Clean set (n = 229)"), (axes[1], "full", "Full set (n = 308)")]:
        st, tx = load(REM / f"stratify_{tag}.json"), load(REM / f"textonly_{tag}.json")["auroc"]["utility"]
        rows = [("Prompt length (text)", tx["prompt_length_chars"], C_GREY),
                ("Question-snippet overlap (text)", tx["question_snippet_overlap"], C_GREY),
                ("TF-IDF + logistic reg. (text)", tx["tfidf_logreg_nestedcv"], C_GREY),
                ("Utility probe, pre-search", st["auroc_PRE_to_utility"], C_PRE),
                ("Utility probe, post-search", st["auroc_POST_to_utility"], C_POST),
                ("Need(pre) $\\times$ correct(post) probes", st["auroc_product_PREneed_x_POSTcorrect_to_utility"], C_REF),
                ("Gold answer in snippets (oracle)", tx["REFERENCE_ONLY_gold_in_snippets"], "white")]
        y = list(range(len(rows)))[::-1]
        for yi, (lab, v, c) in zip(y, rows):
            ax.barh(yi, v - 0.5, left=0.5, color=c, edgecolor="black" if c == "white" else c, lw=0.6,
                    hatch="////" if c == "white" else None, height=0.68)
            ax.text(v + 0.006, yi, f"{v:.2f}", va="center", fontsize=7.5)
        ax.set_yticks(y, [r[0] for r in rows] if ax is axes[0] else [""] * len(rows))
        ax.set_xlim(0.45, 0.92); ax.axvline(0.5, color="black", lw=0.5)
        ax.set_title(title); ax.set_xlabel("AUROC for utility")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig3_decomposition")


def outcome_counts(runs, judgments):
    sys.path.insert(0, str(ROOT / "src"))
    from io_utils import load_jsonl, question_of
    from diagnostics import flag_structured
    J = [r for r in load_jsonl(judgments) if r["condition"] == "force_google" and r.get("rep", 0) == 0]
    sc = {r["question"]: float(r["score"]) for r in J}
    groups = {}
    for r in load_jsonl(Path(runs) / "force_google.jsonl"):
        f = flag_structured(r)
        f = {"other_tool_error": "empty_results"}.get(f, f)
        groups.setdefault(f, []).append(sc.get(question_of(r)))
    return {k: {"n": len(v), "accuracy": round(sum(x for x in v if x is not None) / max(1, sum(x is not None for x in v)), 3)}
            for k, v in groups.items()}


def fig_outcomes(counts):
    """What happened to the 314 always-search runs, and accuracy in each group."""
    retry = load(REM / "retry_audit.json")
    order = [("ok", "Results returned"), ("tool_arg_error", "Wrong argument name"), ("empty_results", "Empty result")]
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    for i, (k, lab) in enumerate(order):
        c = counts.get(k, {"n": 0, "accuracy": 0})
        ax.barh(i, c["n"], color=[C_POST, C_REF, C_GREY][i], height=0.62)
        ax.text(c["n"] + 4, i, f"n = {c['n']}, accuracy {c['accuracy']:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(range(3), [o[1] for o in order]); ax.invert_yaxis()
    ax.set_xlim(0, 360); ax.set_xlabel("Always-search runs (of 314)")
    ax.set_title(f"Quota errors: {retry['true_quota_before']}, of which {retry['true_quota_fixed']} recovered on retry",
                 fontsize=8)
    save(fig, "fig4_outcomes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs"); ap.add_argument("--judgments")
    a = ap.parse_args()
    cache = REM / "accuracy_by_outcome.json"
    if a.runs and a.judgments:
        cache.write_text(json.dumps(outcome_counts(a.runs, a.judgments), indent=1))
    fig_auroc(); fig_seeds(); fig_decomposition()
    if cache.exists():
        fig_outcomes(load(cache))
    print("wrote", sorted(p.name for p in FIG.glob("fig*")))


if __name__ == "__main__":
    main()
