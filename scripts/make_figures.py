"""Regenerate the paper's figures from the reference outputs in outputs/.

Figures follow Nature Machine Intelligence artwork guidance: sans-serif text at
5-7 pt, width <= 180 mm, editable (TrueType) text in vector PDF, no in-panel
titles (the legend carries the title), and a colour-blind-safe palette (Okabe-Ito).

  fig_instrument_validation.pdf  synthetic validation: LAP vs true leakage, detection, example curve
  fig_cto_leakage.pdf            CTO leakage-response (start -> during -> all labeling functions)
  fig_trialbench.pdf             TrialBench provided vs temporal split: AUPRC (a) and AUROC (b)
  fig_llm_memorization.pdf       LLM nested prompt conditions (a), contrasts (b), dating by approval (c)
  fig_causal_balance.pdf         covariate balance before/after overlap weighting (Extended Data)

Usage:  python scripts/make_figures.py [--out figures]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = "outputs"
MM = 1 / 25.4                      # inches per millimetre
SINGLE, DOUBLE = 89 * MM, 180 * MM  # Nature column widths
BLUE, VERMILION, SKY, GREY = "#0072B2", "#D55E00", "#56B4E9", "#7F7F7F"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def _save(fig, path):
    """Save without a creation timestamp so regenerated figures are byte-identical."""
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})


def _load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def _err(point, ci):
    """Asymmetric error-bar half-widths from a point estimate and a [lo, hi] CI."""
    return [[p - c[0] for p, c in zip(point, ci)], [c[1] - p for p, c in zip(point, ci)]]


def _bin_label(b):
    """'<=2021' -> '≤2021'; '2022-2022' -> '2022'; '>2022' -> '≥2023'."""
    if b.startswith("<="):
        return "≤" + b[2:]
    if b.startswith(">"):
        return f"≥{int(b[1:]) + 1}"
    lo, hi = b.split("-")
    return lo if lo == hi else f"{lo}–{hi}"


def _panel_label(ax, s, y=1.04):
    ax.text(-0.16, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def fig_cto(out):
    d = _load("cto_audit.json")
    t = d["tiers"]
    keys = ["h0_start", "h1_during", "h2_post_naive"]
    labels = ["Start-time\n(deployable)", "+ During-trial", "+ Post-completion\n(all)"]
    vals = [t[k]["auprc"] for k in keys]
    fig, ax = plt.subplots(figsize=(SINGLE, 62 * MM))
    x = np.arange(3)
    ax.bar(x, vals, yerr=_err(vals, [t[k]["auprc_ci"] for k in keys]), capsize=2.5,
           color=[BLUE, SKY, VERMILION], width=0.62, error_kw={"lw": 0.7})
    ax.axhline(d["test_base_rate"], ls="--", color=GREY, lw=0.8,
               label=f"Test base rate ({d['test_base_rate']:.2f})")
    cto = d.get("cto_pred_proba_reference")
    if cto:
        ax.axhline(cto["auprc"], ls=":", color="black", lw=0.8,
                   label=f"CTO fused label model ({cto['auprc']:.2f})")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=6)
    for i, v in enumerate(vals):
        ax.text(i, t[keys[i]]["auprc_ci"][1] + 0.02, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUPRC (curated human labels)")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    _save(fig, os.path.join(out, "fig_cto_leakage.pdf"))
    plt.close(fig)


def fig_trialbench(out):
    d = _load("trialbench_audit.json")
    ph = d["phases"]
    names = [p["phase"].replace("Phase", "Phase ") for p in ph]
    x, w = np.arange(len(ph)), 0.36
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.75, 58 * MM))
    for ax, metric, label, tag in ((axes[0], "auprc", "AUPRC", "a"), (axes[1], "auroc", "AUROC", "b")):
        prov = [p["provided_split"][metric] for p in ph]
        temp = [p["temporal_split"][metric] for p in ph]
        ax.bar(x - w / 2, prov, w, color=VERMILION, label="Provided (non-temporal) split",
               yerr=_err(prov, [p["provided_split"][f"{metric}_ci"] for p in ph]),
               capsize=2, error_kw={"lw": 0.7})
        ax.bar(x + w / 2, temp, w, color=BLUE, label="Temporal split (by NCT number)",
               yerr=_err(temp, [p["temporal_split"][f"{metric}_ci"] for p in ph]),
               capsize=2, error_kw={"lw": 0.7})
        if metric == "auprc":  # AUPRC is read against each test set's positive rate
            for i, p in enumerate(ph):
                for dx, key in ((-w / 2, "provided_split"), (w / 2, "temporal_split")):
                    ax.hlines(p[key]["test_base_rate"], x[i] + dx - w / 2, x[i] + dx + w / 2,
                              colors="black", linestyles="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.0)
        _panel_label(ax, tag)
    axes[0].legend(frameon=False, loc="upper left", fontsize=6)
    fig.tight_layout()
    _save(fig, os.path.join(out, "fig_trialbench.pdf"))
    plt.close(fig)


LLM_ORDER = ["ollama:llama3.1:8b", "claude-haiku-4-5-20251001", "claude-sonnet-5",
             "claude-opus-4-8", "claude-opus-5"]
LLM_NAMES = {"ollama:llama3.1:8b": "Llama 3.1 8B", "claude-haiku-4-5-20251001": "Haiku 4.5",
             "claude-sonnet-5": "Sonnet 5", "claude-opus-4-8": "Opus 4.8", "claude-opus-5": "Opus 5"}
COND_STYLE = [("D", "Design only (D)", "#BBBBBB"), ("D+ID", "D + identifier", SKY),
              ("D+Tm", "D + title, intervention masked", "#009E73"), ("D+T", "D + title", BLUE),
              ("D+T+ID+S", "D + title + identifier + sponsor", VERMILION)]
CONTRAST_STYLE = [("D+ID - D", "Identifier recall", SKY),
                  ("D+Tm - D", "Title design semantics", "#009E73"),
                  ("D+T - D+Tm", "Named-intervention knowledge", VERMILION)]


def fig_llm(out):
    d = _load("llm_memorization_study.json")
    order = [m for m in LLM_ORDER if m in d["models"] and "conditions" in d["models"][m]["all"]]
    conds = [c for c in COND_STYLE if c[0] in d["models"][order[0]]["all"]["conditions"]]
    contrasts = [c for c in CONTRAST_STYLE if c[0] in d["models"][order[0]]["all"]["contrasts"]]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 72 * MM),
                             gridspec_kw={"width_ratios": [1.9, 1.3, 1.1]})
    two_line = [LLM_NAMES[m].replace(" ", "\n", 1) for m in order]
    # a: AUPRC per prompt condition
    ax = axes[0]
    x, w = np.arange(len(order)), 0.8 / len(conds)
    for k, (cond, label, color) in enumerate(conds):
        v = [d["models"][m]["all"]["conditions"][cond]["auprc"] for m in order]
        ci = [d["models"][m]["all"]["conditions"][cond]["auprc_ci"] for m in order]
        ax.bar(x + (k - (len(conds) - 1) / 2) * w, v, w, color=color, label=label, yerr=_err(v, ci),
               capsize=1.5, error_kw={"lw": 0.6})
    ax.axhline(d["base_rate"], ls="--", color=GREY, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(two_line, fontsize=6)
    ax.set_ylabel("AUPRC (curated human labels)")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=5.5, ncol=2,
              handlelength=1.2, columnspacing=0.8)
    _panel_label(ax, "a")
    # b: paired contrasts
    ax = axes[1]
    cw = 0.8 / len(contrasts)
    for k, (key, label, color) in enumerate(contrasts):
        v = [d["models"][m]["all"]["contrasts"][key]["auprc"] for m in order]
        ci = [d["models"][m]["all"]["contrasts"][key]["auprc_ci"] for m in order]
        ax.bar(x + (k - (len(contrasts) - 1) / 2) * cw, v, cw, color=color, label=label, yerr=_err(v, ci),
               capsize=1.5, error_kw={"lw": 0.6})
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(two_line, fontsize=6)
    ax.set_ylabel("Paired AUPRC difference")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=5.5, ncol=1,
              handlelength=1.2)
    _panel_label(ax, "b")
    # c: recency (largest model), full gap by completion year
    # c: named-intervention knowledge by the drug's approval history (Claude models)
    ax = axes[2]
    dating = _load("llm_knowledge_dating.json")
    strata = [("approved_before_start", "Approved\nbefore"),
              ("never_approved_code", "Never\napproved"),
              ("approved_after_start", "Approved\nafter")]
    claude = [m for m in order if m.startswith("claude")]
    shades = ["#9ECAE1", "#6BAED6", "#2171B5", "#08306B"]
    sw = 0.8 / len(claude)
    for k, m in enumerate(claude):
        cs = [dating["models"][m][st]["D+T - D+Tm"] for st, _ in strata]
        v = [c["auprc"] for c in cs]
        ax.bar(np.arange(len(strata)) + (k - (len(claude) - 1) / 2) * sw, v, sw,
               color=shades[k % len(shades)], label=LLM_NAMES[m],
               yerr=_err(v, [c["auprc_ci"] for c in cs]), capsize=1.2, error_kw={"lw": 0.5})
    first = dating["models"][claude[0]]
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(np.arange(len(strata)))
    ax.set_xticklabels([f"{lab}\nn = {first[st]['D+T - D+Tm']['n']}" for st, lab in strata],
                       fontsize=5.5)
    ax.set_xlabel("Drug approval vs trial start")
    ax.set_ylabel("Named-intervention knowledge\n(AUPRC difference)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=5.2, ncol=2,
              handlelength=1.0, columnspacing=0.6)
    _panel_label(ax, "c")
    fig.tight_layout()
    _save(fig, os.path.join(out, "fig_llm_memorization.pdf"))
    plt.close(fig)


def fig_validation(out):
    d = _load("instrument_validation.json")
    lv = d["levels"]
    x = [v["leak_strength"] for v in lv]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 55 * MM))
    ax = axes[0]
    m = [v["mean_lap"] for v in lv]
    ax.errorbar(x, m, yerr=_err(m, [v["lap_range_95"] for v in lv]), fmt="o-", color=BLUE,
                ms=3, lw=0.9, capsize=2, elinewidth=0.7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("True leakage (leak strength)")
    ax.set_ylabel("Estimated LAP (AUPRC)")
    _panel_label(ax, "a")
    ax = axes[1]
    series = [("Score mechanism", lv, VERMILION, "o")]
    if d.get("count_levels"):
        series.append(("Count mechanism", d["count_levels"], BLUE, "s"))
    for label, levels, color, marker in series:
        mx = [v["mean_lap"] for v in levels]
        ax.plot(mx, [v["detection_rate"] for v in levels], marker + "-", color=color, ms=3, lw=0.9,
                label=f"{label}: detection")
        ax.plot(mx, [v["coverage_of_mean_lap"] for v in levels], marker + ":", color=color, ms=3,
                lw=0.8, mfc="white", label=f"{label}: coverage")
    ax.axhline(0.025, ls=":", color="black", lw=0.6)
    ax.axhline(0.95, ls=":", color="black", lw=0.6)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Mean estimated LAP (effect size)")
    ax.set_ylabel("Fraction of replicates")
    ax.legend(frameon=False, loc="center right", fontsize=5)
    _panel_label(ax, "b")
    ax = axes[2]
    c = d["example_curve"]["curve"]
    hs = [str(v["horizon"]) if v["horizon"] is not None else "∞" for v in c]
    vals = [v["auprc_mean"] for v in c]
    ax.errorbar(range(len(c)), vals, yerr=_err(vals, [v["auprc_ci"] for v in c]), fmt="o-",
                color=BLUE, ms=3, lw=0.9, capsize=2, elinewidth=0.7)
    ax.axhline(d["example_curve"]["test_base_rate"], ls="--", color=GREY, lw=0.8)
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels(hs)
    ax.set_xlabel("Hindsight horizon h (years)")
    ax.set_ylabel("AUPRC")
    _panel_label(ax, "c")
    fig.tight_layout()
    _save(fig, os.path.join(out, "fig_instrument_validation.pdf"))
    plt.close(fig)


def fig_balance(out):
    d = _load("censored_benchmark_20disease.json")
    per = d["causal"]["covariate_balance"]["per_covariate_smd_raw_then_weighted"]
    items = sorted(per.items(), key=lambda kv: abs(kv[1][0]), reverse=True)[:15]
    pretty = {"sponsor_tier": "sponsor tier", "phase_from": "phase", "info_time": "entry year"}
    names = [pretty.get(k, k.replace("area_", "area: ").replace("mod_", "modality: ")
                        .replace("_", " "))[:34] for k, _ in items]
    raw = [abs(v[0]) for _, v in items]
    wtd = [abs(v[1]) for _, v in items]
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(SINGLE, 70 * MM))
    for yi, r, wv in zip(y, raw, wtd):
        ax.plot([wv, r], [yi, yi], color=GREY, lw=0.6, zorder=0)
    ax.scatter(raw, y, s=16, facecolors="none", edgecolors=VERMILION, linewidths=0.8,
               label="Unweighted")
    ax.scatter(wtd, y, s=16, color=BLUE, label="Overlap-weighted")
    ax.axvline(0.1, ls="--", color=GREY, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=5.5)
    ax.set_xlabel("|Standardized mean difference|")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, os.path.join(out, "fig_causal_balance.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Regenerate the paper's figures.")
    ap.add_argument("--out", default="figures", help="output directory for the PDFs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_validation(args.out)
    fig_cto(args.out)
    fig_trialbench(args.out)
    fig_llm(args.out)
    fig_balance(args.out)
    print("wrote 5 figures to", args.out)


if __name__ == "__main__":
    main()
