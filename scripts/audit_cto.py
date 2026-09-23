"""Temporal-leakage audit of the CTO benchmark (Gao et al., arXiv:2406.10292).

CTO ("Automatically Labeling Clinical Trial Outcomes") fuses ~20 labeling functions
(LFs) into a Random-Forest success probability. By CTO's own source, several LFs are
computed on POST-COMPLETION windows and therefore cannot exist at a real go/no-go
decision (trial start):
  * stock_price  : sponsor stock slope over (completion_date, completion_date+7d)
                   -- stock_price/get_stocks.py L68,L96
  * new_headlines: news sentiment restricted to COMPLETED trials, centered on completion
                   -- news_headlines/get_news2.py L95,L264
  * gpt/gpt2     : GPT verdict on PubMed abstracts up to 5y AFTER completion
                   -- llm_prediction_on_pubmed/support_functions.py L128-134
  * pvalues, results_reported, status/status2, *_ae, patient_drop, amendments,
    num_patients, update_more_recent : results-section / final-status signals.

Post-hoc signals are appropriate for CTO's stated *labeling* purpose. They become
temporal leakage when CTO is used -- as its "benchmark for drug development" framing
invites, and as downstream outcome predictors do -- for *prospective* prediction.

We group the LFs by information vintage and trace the leakage-response curve: AUPRC/AUROC
against the CURATED human labels (not CTO's own RF labels, so no circularity) as
post-decision evidence is admitted. h0 = start-time only (deployable); the top tier is
the full fusion of all LFs (comparable to what CTO reports). LAP = all - start is the
post-START gap; it is decomposed into its during-trial (h1 - h0) and post-completion
(h2 - h1) components. CIs are sponsor-clustered percentile bootstraps (1,000 resamples,
paired for contrasts); the split is temporal (train: start year <= 60th percentile).

Usage:  python scripts/audit_cto.py [--data-dir data/cto] [--out outputs/cto_audit.json]
Data:   four CTO tables, downloaded on first use from
        https://huggingface.co/datasets/chufangao/CTO
        (phase{1,2,3}_CTO_rf.csv, human_labels_2020_2024.csv).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

# LF vintage tiers (from CTO source inspection).
START = ["hint_train", "hint_train2", "hint_train3", "sites", "num_sponsors"]
DURING = ["linkage", "linkage2", "amendments", "patient_drop", "num_patients",
          "serious_ae", "death_ae", "all_ae"]
POST = ["status", "status2", "results_reported", "update_more_recent",
        "gpt", "gpt2", "stock_price", "new_headlines", "pvalues"]
TIERS = [("h0_start", START),
         ("h1_during", START + DURING),
         ("h2_post_naive", START + DURING + POST)]

# CTO tables are streamed on demand from the public Hugging Face dataset rather than
# committed to the repo (keeps the project data-file-free). basename -> remote path.
CTO_HF_BASE = "https://huggingface.co/datasets/chufangao/CTO/resolve/main"
CTO_HF_FILES = {
    "phase1_CTO_rf.csv": "phase1_CTO_rf.csv",
    "phase2_CTO_rf.csv": "phase2_CTO_rf.csv",
    "phase3_CTO_rf.csv": "phase3_CTO_rf.csv",
    "human_labels_2020_2024.csv": "human_labels_2020_2024/human_labels_2020_2024.csv",
}


def _ensure_cto_data(data_dir):
    """Download the CTO tables we need if they are not already present locally."""
    import urllib.request
    os.makedirs(data_dir, exist_ok=True)
    for local, remote in CTO_HF_FILES.items():
        path = os.path.join(data_dir, local)
        if os.path.exists(path):
            continue
        print(f"  streaming {local} from Hugging Face ...")
        urllib.request.urlretrieve(f"{CTO_HF_BASE}/{remote}", path)


def _auprc(y, p):
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _auroc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def load(data_dir):
    _ensure_cto_data(data_dir)  # stream the tables if they are not present locally
    hl = pd.read_csv(os.path.join(data_dir, "human_labels_2020_2024.csv"), low_memory=False)
    lf = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in
                    ("phase1_CTO_rf.csv", "phase2_CTO_rf.csv", "phase3_CTO_rf.csv")],
                   ignore_index=True).drop_duplicates("nct_id")
    keep = ["nct_id", "labels", "start_date", "completion_date", "phase", "source"]
    j = hl[keep].merge(lf, on="nct_id", how="inner")
    j = j[j["labels"].isin([0.0, 1.0])].copy()
    j["labels"] = j["labels"].astype(int)
    j["start_year"] = pd.to_datetime(j["start_date"], errors="coerce").dt.year
    j = j.dropna(subset=["start_year"])
    j["start_year"] = j["start_year"].astype(int)
    j["source"] = j["source"].fillna("UNKNOWN")
    return j


def _fit_predict(tr, te, cols):
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=300,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(tr[cols].values, tr["labels"].values)
    return clf.predict_proba(te[cols].values)[:, 1]


def _clustered_samples(yte, preds_by_key, clusters, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    c2r = {c: np.where(clusters == c)[0] for c in uniq}
    samp = {k: [] for k in preds_by_key}
    for _ in range(n_boot):
        rows = np.concatenate([c2r[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
        yb = yte[rows]
        if len(np.unique(yb)) < 2:
            continue
        for k, p in preds_by_key.items():
            samp[k].append(_auprc(yb, p[rows]))
    return samp


def main():
    ap = argparse.ArgumentParser(description="Temporal-leakage audit of the CTO benchmark.")
    ap.add_argument("--data-dir", default="data/cto")
    ap.add_argument("--out", default="outputs/cto_audit.json")
    args = ap.parse_args()
    j = load(args.data_dir)
    cut = int(j["start_year"].quantile(0.6))
    tr, te = j[j["start_year"] <= cut], j[j["start_year"] > cut]
    yte = te["labels"].values
    base = float(te["labels"].mean())

    preds, points = {}, {}
    for name, cols in TIERS:
        p = _fit_predict(tr, te, cols)
        preds[name] = p
        points[name] = {"auprc": round(_auprc(yte, p), 4), "auroc": round(_auroc(yte, p), 4),
                        "n_lfs": len(cols)}
    cto_ref = None
    if "pred_proba" in te.columns:
        pp = te["pred_proba"].values
        cto_ref = {"auprc": round(_auprc(yte, pp), 4), "auroc": round(_auroc(yte, pp), 4)}

    samp = _clustered_samples(yte, preds, te["source"].values, n_boot=1000, seed=0)

    def ci(a):
        a = np.array([x for x in a if np.isfinite(x)])
        return [round(float(np.quantile(a, .025)), 4), round(float(np.quantile(a, .975)), 4)]

    lap = np.array([n - s for n, s in zip(samp["h2_post_naive"], samp["h0_start"])
                    if np.isfinite(n) and np.isfinite(s)])
    lap_pt = points["h2_post_naive"]["auprc"] - points["h0_start"]["auprc"]
    frac = lap_pt / points["h2_post_naive"]["auprc"]

    # Decompose the post-start gap into its during-trial and post-completion parts
    # (paired, on the same sponsor-clustered resamples).
    def paired(a_key, b_key):
        draws = np.array([a - b for a, b in zip(samp[a_key], samp[b_key])
                          if np.isfinite(a) and np.isfinite(b)])
        return {"auprc": round(points[a_key]["auprc"] - points[b_key]["auprc"], 4),
                "ci": [round(float(np.quantile(draws, .025)), 4),
                       round(float(np.quantile(draws, .975)), 4)]}
    lap_components = {"during_trial": paired("h1_during", "h0_start"),
                      "post_completion": paired("h2_post_naive", "h1_during")}

    # Per-labeling-function decomposition: the marginal AUPRC each post-start LF adds to the
    # start-time set (i.e. which signals carry the leakage), with paired sponsor-clustered CIs.
    leaky_lfs = DURING + POST
    lf_preds = {"__start__": preds["h0_start"]}
    for lf in leaky_lfs:
        lf_preds[lf] = _fit_predict(tr, te, START + [lf])
    lf_samp = _clustered_samples(yte, lf_preds, te["source"].values, n_boot=1000, seed=0)
    per_lf = {}
    for lf in leaky_lfs:
        pt = _auprc(yte, lf_preds[lf]) - points["h0_start"]["auprc"]
        marg = [a - b for a, b in zip(lf_samp[lf], lf_samp["__start__"])
                if np.isfinite(a) and np.isfinite(b)]
        per_lf[lf] = {"marginal_auprc": round(float(pt), 4), "ci": ci(marg),
                      "p_gt_0": round(float(np.mean(np.array(marg) > 0)), 3) if marg else None}
    per_lf = dict(sorted(per_lf.items(), key=lambda kv: -kv[1]["marginal_auprc"]))

    out = {
        "benchmark": "CTO (Gao et al. 2024, arXiv:2406.10292)",
        "eval_against": "curated human labels (human_labels_2020_2024), not CTO's RF labels",
        "n": len(j), "n_train": len(tr), "n_test": len(te),
        "train_start_year_le": cut, "test_base_rate": round(base, 4),
        "n_sponsors": int(j["source"].nunique()),
        "tiers": {name: {**points[name], "auprc_ci": ci(samp[name])} for name, _ in TIERS},
        "cto_pred_proba_reference": cto_ref,
        "LAP_auprc": round(lap_pt, 4),
        "LAP_auprc_ci": [round(float(np.quantile(lap, .025)), 4), round(float(np.quantile(lap, .975)), 4)],
        "LAP_p_gt_0": round(float(np.mean(lap > 0)), 4),
        "leakage_fraction_of_auprc": round(float(frac), 4),
        "LAP_components": lap_components,
        "per_lf_marginal_leakage": per_lf,
        "vintage_groups": {"start": START, "during": DURING, "post_completion": POST},
        "interpretation": (
            "Restricting CTO's labeling functions to start-time signals collapses AUPRC "
            f"{points['h2_post_naive']['auprc']:.3f} -> {points['h0_start']['auprc']:.3f} "
            f"(AUROC {points['h2_post_naive']['auroc']:.3f} -> {points['h0_start']['auroc']:.3f}); "
            f"~{100*frac:.0f}% of the apparent AUPRC depends on signals unavailable at trial "
            "start (during-trial + post-completion). Legitimate for CTO's labeling purpose; "
            "severe leakage if used for prospective prediction."
        ),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"n={len(j)} base_rate(test)={base:.3f} split start_year<={cut} "
          f"(train {len(tr)} / test {len(te)}) sponsors={j['source'].nunique()}")
    for name, _ in TIERS:
        c = ci(samp[name])
        print(f"  {name:16s} AUPRC={points[name]['auprc']:.3f} CI[{c[0]:.3f},{c[1]:.3f}]  "
              f"AUROC={points[name]['auroc']:.3f}")
    if cto_ref:
        print(f"  CTO pred_proba   AUPRC={cto_ref['auprc']:.3f} AUROC={cto_ref['auroc']:.3f} (their RF)")
    print(f"  LAP={lap_pt:+.3f} CI[{out['LAP_auprc_ci'][0]:+.3f},{out['LAP_auprc_ci'][1]:+.3f}] "
          f"P(>0)={out['LAP_p_gt_0']:.3f}  leakage={100*frac:.0f}% of AUPRC")
    for k, v in lap_components.items():
        print(f"    {k:16s} {v['auprc']:+.3f} CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]")
    print("  per-LF marginal AUPRC over start-only (which signals carry the leakage):")
    for lf, v in list(per_lf.items())[:6]:
        print(f"    {lf:18s} +{v['marginal_auprc']:.3f}  CI[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]  P(>0)={v['p_gt_0']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
