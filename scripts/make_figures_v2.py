"""Figures for the revised analyses (v2), built from the reference outputs in outputs/.

Reuses the style of make_figures.py (Okabe-Ito palette, 5-7 pt sans-serif, editable
TrueType text, timestamp-free PDFs):

  fig_instrument_validation_v2.pdf  synthetic validation at the default setting (a, b) and at a
                                    setting matched to the 20-disease benchmark (c)
  fig_cto_leakage_v2.pdf            CTO labeling signals tiered by public availability (a) and
                                    the gain from each signal added alone to the start-time set (b)
  fig_trialbench_v2.pdf             TrialBench post-start features on the temporal test set (a, b)
                                    and the effect of training on test-era trials (c)
  fig_llm_memorization_v2.pdf       LLM nested prompts (a), paired contrasts (b) and the
                                    named-intervention gain by curated approval history (c)
  fig_physionet2012_v2.pdf          patient-level demonstration on ICU records (PhysioNet 2012): the
                                    leakage-response curve (a), per-source and per-variable gains (b)
                                    and the decision curve of the 24-hour model (c)
  fig_method_comparison.pdf         the method against permutation importance and a univariate screen
  fig_physionet2019_sepsis.pdf      sepsis onset in the ICU (PhysioNet 2019; train hospital A, test B):
                                    leakage-response curves (a) and per-source and per-variable gains (b)

Usage:  python scripts/make_figures_v2.py [--out figures]
"""
import argparse
import os
import sys

import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figures as mf  # noqa: E402  (sets rcParams and provides the helpers)

plt = mf.plt
BLUE, VERMILION, SKY, GREY, GREEN = mf.BLUE, mf.VERMILION, mf.SKY, mf.GREY, "#009E73"


def fig_validation(out):
    d = mf._load("instrument_validation.json")
    mt = mf._load("instrument_validation_matched.json")
    lv = d["levels"]
    fig, axes = plt.subplots(1, 3, figsize=(mf.DOUBLE, 57 * mf.MM))
    # a: estimated vs true leakage (default setting)
    ax = axes[0]
    x = [v["leak_strength"] for v in lv]
    m = [v["mean_lap"] for v in lv]
    ax.errorbar(x, m, yerr=mf._err(m, [v["lap_range_95"] for v in lv]), fmt="o-", color=BLUE,
                ms=3, lw=0.9, capsize=2, elinewidth=0.7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("True leakage (leak strength)")
    ax.set_ylabel("Estimated LAP (AUPRC)")
    mf._panel_label(ax, "a")

    def _detcov(ax, series, title):
        for label, levels, color, marker, *joined in series:
            mx = [v["mean_lap"] for v in levels]
            ls_d, ls_c = ("-", ":") if (not joined or joined[0]) else ("", "")
            ax.plot(mx, [v["detection_rate"] for v in levels], marker + ls_d, color=color, ms=2.6,
                    lw=0.9, label=f"{label}: detection")
            ax.plot(mx, [v["coverage_of_mean_lap"] for v in levels], marker + ls_c, color=color,
                    ms=2.6, lw=0.8, mfc="white", label=f"{label}: coverage")
        ax.axhline(0.025, ls=":", color="black", lw=0.6)
        ax.axhline(0.95, ls=":", color="black", lw=0.6)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Mean estimated LAP (effect size)")
        ax.set_ylabel("Fraction of replicates")
        ax.set_title(title, fontsize=6.5, pad=3)

    # b: default setting (about 245 test programs, prevalence about 0.5)
    series = [("Score", lv, VERMILION, "o")]
    if d.get("count_levels"):
        series.append(("Count", d["count_levels"], BLUE, "s", False))   # no levels between 0 and 0.11
    _detcov(axes[1], series, "Default setting")
    axes[1].legend(frameon=False, loc="center right", fontsize=5)
    mf._panel_label(axes[1], "b")
    # c: matched to the 20-disease benchmark (about 205 test programs, prevalence about 0.15)
    r = mt["results"]
    null = r["null"]
    series = [("Score", null + r["score"], VERMILION, "o"), ("Count", null + r["count"], BLUE, "s"),
              ("Score, target effect", r["re_null"] + r["re_score"], GREEN, "^")]
    _detcov(axes[2], series, "Matched to 20-disease benchmark")
    obs = mt["real_benchmark"]["observed_LAP_auprc"]
    axes[2].axvline(obs, color=GREY, lw=0.8, ls="--")
    axes[2].text(obs - 0.006, 0.60, f"observed\nLAP {obs:.3f}", fontsize=5.2, color=GREY, ha="right")
    axes[2].legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.07), fontsize=4.6)
    mf._panel_label(axes[2], "c")
    fig.tight_layout()
    mf._save(fig, os.path.join(out, "fig_instrument_validation_v2.pdf"))
    plt.close(fig)


CTO_PRETTY = {"status": "Final status", "pvalues": "Results P values", "num_patients": "Participants analysed",
              "gpt": "LLM verdict on abstracts", "all_ae": "Adverse events (all)", "serious_ae": "Serious AEs",
              "death_ae": "Deaths", "patient_drop": "Dropout", "results_reported": "Results reported",
              "update_more_recent": "Record updates", "amendments": "Amendments", "linkage": "Later-phase linkage",
              "stock_price": "Stock price", "new_headlines": "News", "hint_train": "HINT/TOP label"}


def fig_cto(out):
    v2 = mf._load("cto_audit_v2.json")
    ref = mf._load("cto_audit.json").get("cto_pred_proba_reference")
    a = v2["A_corrected_tiers"]
    t = a["tiers"]
    keys = ["t0_start", "t1_start_plus_completion_registry", "t2_all"]
    labels = ["Start-time\n(deployable)", "+ Registry fields\nat completion", "+ External\npost-completion"]
    fig, axes = plt.subplots(1, 2, figsize=(mf.DOUBLE * 0.82, 62 * mf.MM),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    vals = [t[k]["auprc"] for k in keys]
    x = np.arange(3)
    ax.bar(x, vals, yerr=mf._err(vals, [t[k]["auprc_ci"] for k in keys]), capsize=2.5,
           color=[BLUE, SKY, VERMILION], width=0.62, error_kw={"lw": 0.7})
    ex = v2["B_sensitivity"]["excl_withdrawn_refit"]["tiers"]["t0_start"]
    ax.errorbar([0.18], [ex["auprc"]], yerr=mf._err([ex["auprc"]], [ex["auprc_ci"]]), fmt="D",
                color="black", ms=2.6, capsize=2, elinewidth=0.7,
                label=f"Start-time, withdrawn trials excluded (AUROC {ex['auroc']:.2f})")
    ax.axhline(a["test_base_rate"], ls="--", color=GREY, lw=0.8,
               label=f"Test base rate ({a['test_base_rate']:.2f})")
    if ref:
        ax.axhline(ref["auprc"], ls=":", color="black", lw=0.8, label=f"CTO fused label model ({ref['auprc']:.2f})")
    for i, val in enumerate(vals):
        top = max(t[keys[i]]["auprc_ci"][1], ex["auprc_ci"][1] if i == 0 else 0)
        ax.text(i, top + 0.02, f"{val:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("AUPRC (curated human labels)")
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["left"].set_bounds(0, 1.0)
    ax.legend(frameon=False, loc="upper left", fontsize=5.2)
    mf._panel_label(ax, "a")
    # b: gain from each post-start signal added alone to the start-time set
    ax = axes[1]
    c = v2["C_per_lf_marginal_over_corrected_start_exploratory"]
    items = sorted(((k, v) for k, v in c.items() if k != "hint_train"),  # abstains for all test trials
                   key=lambda kv: kv[1]["marginal_auprc"]["point"], reverse=True)
    y = np.arange(len(items))[::-1]
    pts = [it[1]["marginal_auprc"]["point"] for it in items]
    cis = [it[1]["marginal_auprc"]["ci"] for it in items]
    cols = [SKY if it[1]["tier"] == "completion_registry" else VERMILION for it in items]
    ax.barh(y, pts, xerr=mf._err(pts, cis), color=cols, height=0.7, capsize=1.5, error_kw={"lw": 0.6})
    ax.set_yticks(y)
    ax.set_yticklabels([CTO_PRETTY.get(k, k) for k, _ in items], fontsize=5.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("AUPRC gain added alone to start-time set")
    ax.legend(handles=[Patch(color=SKY, label="Registry field at completion"),
                       Patch(color=VERMILION, label="External, post-completion")],
              frameon=False, loc="lower right", fontsize=5.4)
    mf._panel_label(ax, "b", y=1.02)
    fig.tight_layout()
    mf._save(fig, os.path.join(out, "fig_cto_leakage_v2.pdf"))
    plt.close(fig)


def fig_trialbench(out):
    lap = mf._load("trialbench_lap_poststart.json")
    v2 = mf._load("trialbench_audit_v2.json")
    phases = ["Phase1", "Phase2", "Phase3"]
    names = ["Phase I", "Phase II", "Phase III"]
    x, w = np.arange(3), 0.36
    fig, axes = plt.subplots(1, 3, figsize=(mf.DOUBLE, 58 * mf.MM),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.1]})
    for ax, metric, ylab, tag in ((axes[0], "auprc", "AUPRC, temporal test set", "a"),
                                  (axes[1], "auroc", "AUROC, temporal test set", "b")):
        arms = [lap["phases"][p]["temporal_split"]["arms"] for p in phases]
        allf = [a_["all_features"][metric] for a_ in arms]
        abl = [a_["ablated_non_start_removed"][metric] for a_ in arms]
        ax.bar(x - w / 2, allf, w, color=VERMILION, label="All features (incl. actual enrolment)",
               yerr=mf._err(allf, [a_["all_features"][f"{metric}_ci"] for a_ in arms]), capsize=2,
               error_kw={"lw": 0.7})
        ax.bar(x + w / 2, abl, w, color=BLUE, label="Start-time features only",
               yerr=mf._err(abl, [a_["ablated_non_start_removed"][f"{metric}_ci"] for a_ in arms]),
               capsize=2, error_kw={"lw": 0.7})
        if metric == "auprc":
            for i, p in enumerate(phases):
                br = lap["phases"][p]["temporal_split"]["test_base_rate"]
                ax.hlines(br, x[i] - w, x[i] + w, colors="black", linestyles="--", lw=0.8)
        else:
            ax.axhline(0.5, ls="--", color=GREY, lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.0)
        mf._panel_label(ax, tag)
    axes[0].legend(frameon=False, loc="upper left", fontsize=5.4)
    # c: adding test-era trials to training, fixed temporal test set (start-time features)
    ax = axes[2]
    for k, (metric, color, label) in enumerate((("auprc", VERMILION, "ΔAUPRC"), ("auroc", BLUE, "ΔAUROC"))):
        pt, ci = [], []
        for p in phases:
            dd = v2["T3_fixed_test_set"][p]["ablated_non_start_removed"]["iii_temporal_test_crossfit"]["b_minus_a"][metric]
            pt.append(dd["point"])
            ci.append(dd["ci"])
        ax.errorbar(x + (k - 0.5) * 0.22, pt, yerr=mf._err(pt, ci), fmt="o", color=color, ms=3,
                    capsize=2, elinewidth=0.7, label=label)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mixed minus past-only training\n(same test set)")
    ax.set_ylim(-0.04, 0.06)
    ax.legend(frameon=False, loc="upper right", fontsize=5.4)
    mf._panel_label(ax, "c")
    fig.tight_layout()
    mf._save(fig, os.path.join(out, "fig_trialbench_v2.pdf"))
    plt.close(fig)


def fig_llm(out):
    d = mf._load("llm_memorization_study.json")
    val = mf._load("llm_validity_checks.json")
    order = [m for m in mf.LLM_ORDER if m in d["models"] and "conditions" in d["models"][m]["all"]]
    cond_style = [("D", "Design + enrolment bin (D)", "#BBBBBB")] + mf.COND_STYLE[1:]
    conds = [c for c in cond_style if c[0] in d["models"][order[0]]["all"]["conditions"]]
    contrasts = [c for c in mf.CONTRAST_STYLE if c[0] in d["models"][order[0]]["all"]["contrasts"]]
    fig, axes = plt.subplots(1, 3, figsize=(mf.DOUBLE, 72 * mf.MM),
                             gridspec_kw={"width_ratios": [1.8, 1.25, 1.25]})
    two_line = [mf.LLM_NAMES[m].replace(" ", "\n", 1) for m in order]
    ax = axes[0]
    x, w = np.arange(len(order)), 0.8 / len(conds)
    for k, (cond, label, color) in enumerate(conds):
        v = [d["models"][m]["all"]["conditions"][cond]["auprc"] for m in order]
        ci = [d["models"][m]["all"]["conditions"][cond]["auprc_ci"] for m in order]
        ax.bar(x + (k - (len(conds) - 1) / 2) * w, v, w, color=color, label=label, yerr=mf._err(v, ci),
               capsize=1.5, error_kw={"lw": 0.6})
    ax.axhline(d["base_rate"], ls="--", color=GREY, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(two_line, fontsize=6)
    ax.set_ylabel("AUPRC (curated human labels)")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=5.5, ncol=2,
              handlelength=1.2, columnspacing=0.8)
    mf._panel_label(ax, "a")
    ax = axes[1]
    cw = 0.8 / len(contrasts)
    for k, (key, label, color) in enumerate(contrasts):
        v = [d["models"][m]["all"]["contrasts"][key]["auprc"] for m in order]
        ci = [d["models"][m]["all"]["contrasts"][key]["auprc_ci"] for m in order]
        ax.bar(x + (k - (len(contrasts) - 1) / 2) * cw, v, cw, color=color, label=label, yerr=mf._err(v, ci),
               capsize=1.5, error_kw={"lw": 0.6})
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(two_line, fontsize=5.4)
    ax.set_ylabel("Paired AUPRC difference")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=5.5, ncol=1, handlelength=1.2)
    mf._panel_label(ax, "b")
    # c: named-intervention gain by curated approval history (Claude models)
    ax = axes[2]
    strata = [("established_all_pre", "Approved\nbefore\nstart"), ("mixed", "Mixed"),
              ("novel_dev_code_only", "Never-\napproved\ncode")]
    claude = [m for m in order if m.startswith("claude")]
    shades = ["#9ECAE1", "#6BAED6", "#2171B5", "#08306B"]
    sw = 0.8 / len(claude)
    R = val["chembl_refined"]["models"]
    for k, m in enumerate(claude):
        cs = [R[m][st]["D+T - D+Tm"] for st, _ in strata]
        v = [c["auprc"] for c in cs]
        ax.bar(np.arange(len(strata)) + (k - (len(claude) - 1) / 2) * sw, v, sw, color=shades[k % 4],
               label=mf.LLM_NAMES[m], yerr=mf._err(v, [c["auprc_ci"] for c in cs]), capsize=1.2,
               error_kw={"lw": 0.5})
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(np.arange(len(strata)))
    ax.set_xticklabels([f"{lab}\nn = {R[claude[0]][st]['D+T - D+Tm']['n']}" for st, lab in strata], fontsize=5.3)
    ax.set_ylabel("Named-intervention knowledge\n(AUPRC difference)")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=5.2, ncol=2,
              handlelength=1.0, columnspacing=0.6)
    mf._panel_label(ax, "c")
    fig.tight_layout()
    mf._save(fig, os.path.join(out, "fig_llm_memorization_v2.pdf"))
    plt.close(fig)


EHR_PRETTY = {"GCS": "Glasgow Coma Scale", "Urine": "Urine output", "Lactate": "Lactate", "HR": "Heart rate",
              "BUN": "Blood urea nitrogen", "pH": "pH", "PaO2": "PaO2", "SysABP": "Systolic BP (invasive)",
              "FiO2": "FiO2", "ALP": "Alkaline phosphatase", "Bilirubin": "Bilirubin"}


def fig_physionet(out):
    d = mf._load("physionet2012_audit.json")
    fig, axes = plt.subplots(1, 3, figsize=(mf.DOUBLE, 60 * mf.MM),
                             gridspec_kw={"width_ratios": [1.0, 1.15, 0.95]})
    ax = axes[0]
    # both designs refit the same learner on the same data at shared cut-offs, so one curve
    # (the union of their points) shows both; the decision times mark the deployable ends
    pts = {}
    for key in ("t12", "t24"):
        for v in d[key]["curve"]:
            pts[v["data_up_to_hours"]] = v
    x = sorted(pts)
    y = [pts[h]["auprc"] for h in x]
    ax.errorbar(x, y, yerr=mf._err(y, [pts[h]["auprc_ci"] for h in x]), fmt="o-", color=BLUE, ms=3,
                lw=0.9, capsize=2, elinewidth=0.7)
    naive = pts[48]["auprc"]
    for key, color, dx in (("t12", VERMILION, -1.6), ("t24", GREEN, -1.6)):
        t_h = d[key]["decision_time_hours"]
        dep = pts[t_h]["auprc"]
        ax.axvline(t_h, color=color, lw=0.7, ls=":")
        ax.annotate("", xy=(t_h + dx, naive), xytext=(t_h + dx, dep),
                    arrowprops=dict(arrowstyle="<->", lw=0.7, color=color))
        ax.text(t_h + 0.8, dep - (0.12 if key == "t12" else 0.10),
                f"decision\nat {t_h} h:\nLAP\n{d[key]['LAP']['auprc']:+.3f}", fontsize=5.0, color=color,
                ha="left", va="top")
        ax.hlines(naive, t_h + dx - 0.5, 48, colors=GREY, linestyles=":", lw=0.5)
    ax.axhline(d["t24"]["test_base_rate"], ls="--", color=GREY, lw=0.8)
    ax.text(47.5, d["t24"]["test_base_rate"] + 0.012, "test base rate", fontsize=5.2, color=GREY, ha="right")
    ax.set_xlabel("Hours of ICU data admitted")
    ax.set_ylabel("AUPRC, in-hospital death (test set)")
    ax.set_xticks([12, 18, 24, 30, 36, 42, 48])
    ax.set_xlim(4, 50)
    ax.set_ylim(0, 0.7)
    mf._panel_label(ax, "a")
    ax = axes[1]
    r = d["t24"]
    items = [("All measured values", r["per_source_placebo"]["measured_values"], BLUE),
             ("All measurement counts", r["per_source_placebo"]["measurement_counts"], SKY)]
    top = list(r["per_variable_placebo_exploratory"].items())[:8]
    items += [(EHR_PRETTY.get(k, k), v, GREY) for k, v in top]
    y = np.arange(len(items))[::-1]
    pts = [v["auprc"] for _, v, _ in items]
    ax.barh(y, pts, xerr=mf._err(pts, [v["auprc_ci"] for _, v, _ in items]), color=[c for *_, c in items],
            height=0.7, capsize=1.5, error_kw={"lw": 0.6})
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in items], fontsize=5.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("AUPRC gain from admitting hours 24-48\n(one group or variable; decision at 24 h)")
    mf._panel_label(ax, "b", y=1.02)
    # c: decision curve of the 24-hour model, deployable versus evaluated with the 48-hour record
    ax = axes[2]
    dc = mf._load("decision_curve_icu.json")
    th = [c["threshold"] for c in dc["curve"]]
    for key, ci_key, color, label in (("net_benefit_naive", "ci_naive", VERMILION, "Evaluated with 48 h"),
                                      ("net_benefit_deployable", "ci_deployable", BLUE, "Deployable (24 h)")):
        yv = [c[key] for c in dc["curve"]]
        ax.fill_between(th, [c[ci_key][0] for c in dc["curve"]], [c[ci_key][1] for c in dc["curve"]],
                        color=color, alpha=0.15, lw=0)
        ax.plot(th, yv, "o-", color=color, ms=2.5, lw=0.9, label=label)
    ax.plot(th, [c["net_benefit_treat_all"] for c in dc["curve"]], color=GREY, lw=0.8, ls="--", label="Treat all")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylim(-0.02, 0.14)
    ax.set_xlim(0.03, 0.52)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.legend(frameon=False, loc="upper right", fontsize=5.4, handlelength=1.6)
    mf._panel_label(ax, "c")
    fig.tight_layout()
    mf._save(fig, os.path.join(out, "fig_physionet2012_v2.pdf"))
    plt.close(fig)


def fig_method_comparison(out):
    mc = mf._load("method_comparison.json")
    lapj = mf._load("trialbench_lap_poststart.json")["phases"]
    fig, axes = plt.subplots(1, 2, figsize=(mf.DOUBLE * 0.9, 62 * mf.MM),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    # a: LAP, joint permutation importance and LAP left after removing the top-ranked feature
    ax = axes[0]
    rows = [("CTO", mc["cto"]["LAP"], mc["cto"]["LAP_ci_published"],
             mc["cto"]["permutation_importance"]["joint_drop"],
             mc["cto"]["after_dropping_the_flagged_feature"]["residual_LAP"],
             mc["cto"]["after_dropping_the_flagged_feature"]["residual_LAP_ci"], "final status")]
    for ph, name in (("Phase1", "TrialBench I"), ("Phase2", "TrialBench II"), ("Phase3", "TrialBench III")):
        v = mc["trialbench"][ph]
        res = lapj[ph]["temporal_split"]["paired_differences"]["LAP_city_minus_enrollment_minus_ablated"]["auprc"]
        rows.append((name, v["LAP"], v["LAP_ci_published"], v["permutation_importance"]["joint_drop"],
                     res["point"], res["ci"], "enrollment"))
    ypos = np.arange(len(rows))[::-1] * 1.0
    hgt = 0.26
    ax.barh(ypos + hgt, [r[1] for r in rows], height=hgt, color=BLUE,
            xerr=mf._err([r[1] for r in rows], [r[2] for r in rows]), capsize=1.5, error_kw={"lw": 0.6},
            label="LAP (this method)")
    ax.barh(ypos, [r[3] for r in rows], height=hgt, color=VERMILION, label="Joint permutation importance")
    ax.barh(ypos - hgt, [r[4] for r in rows], height=hgt, color=SKY,
            xerr=mf._err([r[4] for r in rows], [r[5] for r in rows]), capsize=1.5, error_kw={"lw": 0.6},
            label="LAP after removing the top-ranked feature")
    for yy, r in zip(ypos, rows):
        ax.text(max(r[5][1], 0) + 0.02, yy - hgt, f"({r[6]} removed)",
                fontsize=4.8, va="center", color="#333333")
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlim(-0.05, 1.0)
    ax.set_xlabel("AUPRC attributed to post-decision information")
    ax.legend(frameon=False, loc="lower right", fontsize=5.2, handlelength=1.2)
    mf._panel_label(ax, "a")
    # b: what a univariate screen sees versus the leakage the method measures
    ax = axes[1]
    cto = mf._load("cto_audit_v2.json")["A_corrected_tiers"]
    share = {"CTO": mc["cto"]["LAP"] / (cto["tiers"]["t2_all"]["auprc"] - cto["test_base_rate"])}
    for ph, name in (("Phase1", "TB I"), ("Phase2", "TB II"), ("Phase3", "TB III")):
        s_ = lapj[ph]["temporal_split"]
        share[name] = mc["trialbench"][ph]["LAP"] / (s_["arms"]["all_features"]["auprc"] - s_["test_base_rate"])
    p12 = mf._load("physionet2012_audit.json")["t24"]
    share["ICU"] = mc["icu_physionet2012_decision_24h"]["LAP"] / (p12["curve"][-1]["auprc"] - p12["test_base_rate"])
    cb = mf._load("censored_benchmark_20disease.json")["leakage_response_curve"]
    share["Censored"] = mc["censored_benchmark_20disease"]["LAP"] / (cb["naive_auprc"] - cb["test_base_rate"])
    pts = [("CTO", mc["cto"]["univariate_screen"]["post_decision_max_auroc"], share["CTO"], BLUE)]
    for ph, name in (("Phase1", "TB I"), ("Phase2", "TB II"), ("Phase3", "TB III")):
        v = mc["trialbench"][ph]
        pts.append((name, v["univariate_screen"]["post_decision_max_auroc"], share[name], BLUE))
    icu = mc["icu_physionet2012_decision_24h"]
    cen = mc["censored_benchmark_20disease"]
    pts.append(("ICU", icu["univariate_screen_naive_48h"]["max_auroc"], share["ICU"], GREEN))
    pts.append(("Censored", cen["univariate_screen_naive"]["max_auroc"], share["Censored"], GREEN))
    offsets = {"CTO": (-0.008, -0.06, "right"), "TB I": (0.006, 0.025, "left"), "TB II": (-0.006, 0.03, "right"),
               "TB III": (0.006, 0.025, "left"), "ICU": (0.006, -0.06, "left"),
               "Censored": (-0.004, 0.035, "left")}
    for name, xv, yv, color in pts:
        ax.scatter([xv], [yv], s=14, color=color, zorder=3)
        dx, dy, ha = offsets[name]
        ax.text(xv + dx, yv + dy, name, fontsize=5.4, ha=ha)
    ax.scatter([], [], s=14, color=BLUE, label="post-decision information\nin separate columns")
    ax.scatter([], [], s=14, color=GREEN, label="post-decision information\nin aggregate columns")
    ax.legend(frameon=False, loc="upper left", fontsize=5.0, handletextpad=0.3, borderaxespad=0.2)
    for thr, lab in ((0.80, "0.80"), (0.90, "0.90")):
        ax.axvline(thr, color=GREY, lw=0.7, ls="--")
        ax.text(thr + 0.004, 1.02, f"flag at {lab}", fontsize=5.0, color=GREY, rotation=90, va="top")
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Largest single-feature AUROC\n(training block)")
    ax.set_ylabel("LAP as a share of the naive\nmodel's lift over the base rate")
    mf._panel_label(ax, "b")
    fig.tight_layout()
    mf._save(fig, os.path.join(out, "fig_method_comparison.pdf"))
    plt.close(fig)


SEPSIS_PRETTY = {"Lactate": "Lactate", "HR": "Heart rate", "Resp": "Respiratory rate", "Temp": "Temperature",
                 "WBC": "White cell count", "MAP": "Mean arterial pressure", "SBP": "Systolic BP",
                 "O2Sat": "Oxygen saturation", "Creatinine": "Creatinine", "BUN": "Blood urea nitrogen",
                 "Platelets": "Platelets", "FiO2": "FiO2", "Glucose": "Glucose", "HCO3": "Bicarbonate",
                 "pH": "pH", "PaCO2": "PaCO2", "Bilirubin_total": "Bilirubin", "Hct": "Haematocrit",
                 "Hgb": "Haemoglobin", "Potassium": "Potassium", "Chloride": "Chloride", "DBP": "Diastolic BP",
                 "Magnesium": "Magnesium", "Phosphate": "Phosphate", "Calcium": "Calcium", "PTT": "PTT",
                 "BaseExcess": "Base excess", "AST": "AST", "Alkalinephos": "Alkaline phosphatase",
                 "Fibrinogen": "Fibrinogen", "SaO2": "SaO2", "TroponinI": "Troponin I",
                 "Bilirubin_direct": "Direct bilirubin", "EtCO2": "EtCO2"}


def fig_sepsis(out):
    path = os.path.join(mf.RES, "physionet2019_sepsis_audit.json")
    if not os.path.exists(path):
        return False
    d = mf._load("physionet2019_sepsis_audit.json")
    fig, axes = plt.subplots(1, 2, figsize=(mf.DOUBLE * 0.82, 60 * mf.MM),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0]
    # each decision time defines its own cohort and outcome window, so the two curves are separate
    for key, color, marker in (("t12", VERMILION, "s"), ("t24", BLUE, "o")):
        r = d[key]
        x = [v["data_up_to_hours"] for v in r["curve"]]
        y = [v["auprc"] for v in r["curve"]]
        ax.errorbar(x, y, yerr=mf._err(y, [v["auprc_ci"] for v in r["curve"]]), fmt=marker + "-", color=color,
                    ms=3, lw=0.9, capsize=2, elinewidth=0.7,
                    label=f"decision at {r['decision_time_hours']} h (LAP {r['LAP']['auprc']:+.3f})")
        ax.axhline(r["test_base_rate"], ls="--", color=color, lw=0.6, alpha=0.7)
        ax.axvline(r["decision_time_hours"], color=color, lw=0.6, ls=":")
    ax.set_xlabel("Hours of ICU data admitted")
    ax.set_ylabel("AUPRC, sepsis onset within 24 h\n(external test hospital)")
    ax.set_xticks([12, 18, 24, 30, 36, 42, 48])
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc="upper left", fontsize=5.4)
    ax.text(47.5, max(d["t24"]["test_base_rate"], d["t12"]["test_base_rate"]) + 0.004, "test base rates (dashed)",
            fontsize=5.0, color=GREY, ha="right")
    mf._panel_label(ax, "a")
    ax = axes[1]
    r = d["t24"]
    items = [("All measured values", r["per_source_placebo"]["measured_values"], BLUE),
             ("All measurement counts", r["per_source_placebo"]["measurement_counts"], SKY)]
    top = list(r["per_variable_placebo_exploratory"].items())[:8]
    items += [(SEPSIS_PRETTY.get(k, k), v, GREY) for k, v in top]
    yy = np.arange(len(items))[::-1]
    pts = [v["auprc"] for _, v, _ in items]
    ax.barh(yy, pts, xerr=mf._err(pts, [v["auprc_ci"] for _, v, _ in items]), color=[c for *_, c in items],
            height=0.7, capsize=1.5, error_kw={"lw": 0.6})
    ax.set_yticks(yy)
    ax.set_yticklabels([n for n, _, _ in items], fontsize=5.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("AUPRC gain from admitting hours 24-48\n(one group or variable; decision at 24 h)")
    mf._panel_label(ax, "b", y=1.02)
    fig.tight_layout()
    mf._save(fig, os.path.join(out, "fig_physionet2019_sepsis.pdf"))
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Regenerate the revised (v2) figures.")
    ap.add_argument("--out", default="figures", help="output directory for the PDFs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_validation(args.out)
    fig_cto(args.out)
    fig_trialbench(args.out)
    fig_llm(args.out)
    fig_physionet(args.out)
    fig_method_comparison(args.out)
    n = 6 + int(fig_sepsis(args.out))
    print(f"wrote {n} figures to", args.out)


if __name__ == "__main__":
    main()
