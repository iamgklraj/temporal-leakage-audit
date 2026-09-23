"""Sensitivity of the leakage estimate to the evidence DATING RULE.

The timestamp attached to each evidence datum is the load-bearing modelling choice in this
work: it decides what counts as "knowable at decision time" and therefore sets LAP. The two
defensible rules are:

  * pubmed_year          -- earliest PubMed publication year of the datum's PMIDs
                            (drop the datum if no PMID resolves), and
  * ot_publication_year  -- Open Targets' own publicationYear field.

The primary 20-disease benchmark uses pubmed_year; the 41-disease benchmark uses
ot_publication_year. This script holds the PROGRAMS, the split, the literature cap and the
sampling seed fixed, varies only the dating rule (the ot_publication_year arm also disables
the PMID-year fallback, as in config/benchmark_20disease_otyear.yaml), and reports the
leakage-response under each rule.

Usage:
    # 1. build the alternative-dated evidence for an existing program set
    python scripts/dating_rule_sensitivity.py --build data/benchmark_20disease data/benchmark_20disease_otyear
    # 2. compare
    python scripts/dating_rule_sensitivity.py --compare data/benchmark_20disease data/benchmark_20disease_otyear

Writes outputs/dating_rule_sensitivity.json.
"""
import json, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temporal_leakage_audit import features, leakage, splits          # noqa: E402
from temporal_leakage_audit.config import get_config                  # noqa: E402
from temporal_leakage_audit.data import connectors as C               # noqa: E402

CAP = 400  # identical literature cap + seed in both arms, so only the dating differs


def build(src_dir, out_dir, policy="ot_publication_year"):
    """Re-date an existing program set's evidence under `policy` (reuses the warm cache)."""
    cfg = get_config("config/benchmark_20disease_otyear.yaml")
    cfg["real"]["literature_policy"] = policy
    cfg["real"]["use_pmid_year_fallback"] = False
    cfg["real"]["max_literature_per_pair"] = CAP
    programs = pd.read_csv(os.path.join(src_dir, "programs.csv"))
    ev = C.load_open_targets_evidence(programs, cfg)
    os.makedirs(out_dir, exist_ok=True)
    ev.to_csv(os.path.join(out_dir, "evidence.csv"), index=False)
    programs.to_csv(os.path.join(out_dir, "programs.csv"), index=False)
    print(f"wrote {len(ev)} evidence rows ({policy}) -> {out_dir}")


def _leakage_for(d, cfg, n_boot=500):
    programs = pd.read_csv(os.path.join(d, "programs.csv"))
    evidence = pd.read_csv(os.path.join(d, "evidence.csv"))
    _, _, _, meta, _ = features.build(programs, evidence)
    yrs = programs["info_time"]
    tmax, hi = int(yrs.quantile(0.60)), int(yrs.max())
    tr, te, info = splits.temporal_split(meta, tmax, (tmax + 1, hi), enforce_target_disjoint=True)
    lrc = leakage.leakage_response_curve(
        programs, evidence, tr, te, clusters=meta.loc[te, "target_id"].values,
        n_boot=n_boot, budget_frac=cfg["decision"]["budget_frac"])
    return {"evidence_rows": int(len(evidence)), "n_test": info["n_test"],
            "test_base_rate": lrc["test_base_rate"], "deployable_auprc": lrc["deployable_auprc"],
            "naive_auprc": lrc["naive_auprc"], "LAP": lrc["total_LAP_auprc"],
            "LAP_ci": lrc["total_LAP_auprc_ci"], "LAP_p_gt_0": lrc["total_LAP_auprc_p_gt_0"],
            "peak_LAP": lrc["peak_LAP_auprc"]}


def compare(dir_pubmed, dir_ot):
    cfg = get_config()
    out = {"pubmed_year": _leakage_for(dir_pubmed, cfg),
           "ot_publication_year": _leakage_for(dir_ot, cfg)}
    out["LAP_difference"] = round(out["pubmed_year"]["LAP"] - out["ot_publication_year"]["LAP"], 4)
    out["interpretation"] = (
        "Identical programs and split; only the evidence dating rule differs. A small "
        "LAP_difference with the same sign means the leakage conclusion is not an artifact of "
        "how evidence is timestamped.")
    os.makedirs("outputs", exist_ok=True)
    json.dump(out, open("outputs/dating_rule_sensitivity.json", "w"), indent=2)
    for k in ("pubmed_year", "ot_publication_year"):
        v = out[k]
        print(f"{k:22s} rows={v['evidence_rows']:,} deployable={v['deployable_auprc']:.3f} "
              f"naive={v['naive_auprc']:.3f} LAP={v['LAP']:+.3f} CI{v['LAP_ci']} "
              f"P(>0)={v['LAP_p_gt_0']} peak={v['peak_LAP']:+.3f}")
    print(f"LAP difference between dating rules: {out['LAP_difference']:.3f}")
    print("wrote outputs/dating_rule_sensitivity.json")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--build":
        build(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
