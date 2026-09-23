"""Regenerate the paper's figures from the reference outputs in outputs/.

Figures follow Nature Machine Intelligence artwork guidance: sans-serif text at
5-7 pt, width <= 180 mm, editable (TrueType) text in vector PDF, no in-panel
titles (the legend carries the title), and a colour-blind-safe palette (Okabe-Ito).

  fig_cto_leakage.pdf       CTO leakage-response (start -> during -> all labeling functions)
  fig_trialbench.pdf        TrialBench provided vs temporal split: AUPRC (a) and AUROC (b)
  fig_llm_memorization.pdf  LLM identified vs de-identified AUPRC by model (a), recency (b)
  fig_causal_balance.pdf    covariate balance before/after overlap weighting (Love plot)

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


def _panel_label(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


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
    fig.savefig(os.path.join(out, "fig_cto_leakage.pdf"))
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
    fig.savefig(os.path.join(out, "fig_trialbench.pdf"))
    plt.close(fig)


def fig_llm(out):
    d = _load("llm_memorization_study.json")
    order = ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-5"]
    disp = {"claude-haiku-4-5-20251001": "Haiku 4.5", "claude-sonnet-5": "Sonnet 5",
            "claude-opus-4-8": "Opus 4.8", "claude-opus-5": "Opus 5"}
    order = [m for m in order if m in d["models"]]
    mods = [d["models"][m] for m in order]
    deid = [m["deident_auprc"] for m in mods]
    iden = [m["identified_auprc"] for m in mods]
    has_ci = all("identified_auprc_ci" in m for m in mods)
    rec = d.get("recency") or []
    rec = rec.get("bins", []) if isinstance(rec, dict) else rec

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE * 0.75, 58 * MM),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    x, w = np.arange(len(order)), 0.36
    kw = dict(capsize=2, error_kw={"lw": 0.7})
    ax.bar(x - w / 2, deid, w, color=BLUE, label="De-identified prompt",
           yerr=_err(deid, [m["deident_auprc_ci"] for m in mods]) if has_ci else None, **kw)
    ax.bar(x + w / 2, iden, w, color=VERMILION, label="Identified prompt",
           yerr=_err(iden, [m["identified_auprc_ci"] for m in mods]) if has_ci else None, **kw)
    ax.axhline(d["test_base_rate"], ls="--", color=GREY, lw=0.8,
               label=f"Test base rate ({d['test_base_rate']:.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([disp[m] for m in order])
    ax.set_ylabel("AUPRC (curated human labels)")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    _panel_label(ax, "a")

    ax = axes[1]
    if rec:
        gaps = [r["leakage"] if "leakage" in r else r["memorization_leakage"] for r in rec]
        ax.bar(np.arange(len(rec)), gaps, 0.6, color=VERMILION)
        if all("memorization_leakage_ci" in r for r in rec):
            ax.errorbar(np.arange(len(rec)), gaps,
                        yerr=_err(gaps, [r["memorization_leakage_ci"] for r in rec]),
                        fmt="none", ecolor="black", lw=0.7, capsize=2)
        ax.set_xticks(np.arange(len(rec)))
        ax.set_xticklabels([f"{_bin_label(r['bin'])}\n(n = {r['n']})" for r in rec])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("Trial start year")
    ax.set_ylabel("Identified − de-identified AUPRC")
    _panel_label(ax, "b")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_llm_memorization.pdf"))
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
    fig.savefig(os.path.join(out, "fig_causal_balance.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Regenerate the paper's figures.")
    ap.add_argument("--out", default="figures", help="output directory for the PDFs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_cto(args.out)
    fig_trialbench(args.out)
    fig_llm(args.out)
    fig_balance(args.out)
    print("wrote 4 figures to", args.out)


if __name__ == "__main__":
    main()
