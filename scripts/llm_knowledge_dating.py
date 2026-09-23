"""Can the language models' named-intervention gain be dated? (analysis of saved predictions)

The nested-prompt study (scripts/llm_memorization_study.py) finds that revealing the
named intervention (full title vs masked title, "D+T - D+Tm") raises AUPRC. That knowledge
may be legitimate (what was public about the drug when the trial started) or hindsight
(what became known later, e.g. the programme's fate). This script stratifies trials by how
much could have been public at trial start and recomputes the gain within each stratum:

  approved_before_start  an intervention matches a ChEMBL molecule first approved before the
                         trial's start year (established drug: efficacy knowledge largely
                         predates the trial)
  approved_after_start   an intervention was first approved in or after the start year (its
                         success became public only in hindsight) and none before
  never_approved_code    an intervention carries a development code (e.g. KM-819) and no
                         intervention has ever been approved (novel compound)
  other                  everything else (including approved drugs with no recorded date)

A gain confined to established drugs is consistent with legitimate prior knowledge; a gain
for drugs approved only after the trial started points to hindsight. ChEMBL lookups are cached in outputs/llm_sample_chembl.json.

Usage:  python scripts/llm_knowledge_dating.py   -> outputs/llm_knowledge_dating.json
"""
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_memorization_study as L  # noqa: E402

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
CACHE = "outputs/llm_sample_chembl.json"
OUT = "outputs/llm_knowledge_dating.json"
CONTRASTS = [("D+T", "D+Tm"), ("D+ID", "D")]
N_BOOT, SEED = 1000, 0


def candidates(name):
    """Strings to look up for one intervention name: the name, the name without dose or
    parenthetical text, and its distinctive words."""
    out = {name.strip()} if len(name.split()) <= 3 else set()
    base = re.sub(r"\(.*?\)|\[.*?\]", " ", name)
    base = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|µg|g|ml|mL|iu|IU|units?|%|mg/kg|mg/m2)\b", " ", base)
    base = " ".join(base.split())
    if len(base.split()) <= 3:
        out.add(base)
    for tok in re.split(r"[^A-Za-z0-9-]+", name):
        if len(tok) >= 5 and tok.lower() not in L._STOP and not tok.isdigit():
            out.add(tok)
    return {c for c in out if len(c) >= 3 and c.lower() not in L._STOP}


def lookup(term):
    """ChEMBL molecules whose preferred name or a synonym equals the term (case-insensitive)."""
    # Synonym match also covers preferred names and development codes (e.g. SY-1425).
    hits = []
    url = CHEMBL + "?" + urllib.parse.urlencode(
        {"molecule_synonyms__molecule_synonym__iexact": term,
         "only": "molecule_chembl_id,pref_name,first_approval,max_phase", "limit": 5})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                hits = json.load(r).get("molecules", [])
            break
        except Exception:  # noqa: BLE001 -- transient API errors: back off
            time.sleep(2 ** attempt)
    uniq = {h["molecule_chembl_id"]: h for h in hits}
    return [{"chembl_id": k, "pref_name": v.get("pref_name"),
             "first_approval": v.get("first_approval"), "max_phase": v.get("max_phase")}
            for k, v in uniq.items()]


def chembl_table(interventions):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    terms = sorted({c for v in interventions.values() for n in v["names"] for c in candidates(n)})
    todo = [t for t in terms if t not in cache]
    print(f"ChEMBL: {len(terms)} lookup terms, {len(todo)} uncached", flush=True)
    def save():
        tmp = CACHE + ".tmp"
        json.dump(cache, open(tmp, "w"), indent=0)
        os.replace(tmp, CACHE)

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, (t, res) in enumerate(zip(todo, ex.map(lookup, todo)), 1):
            cache[t] = res
            if i % 200 == 0:
                save()
                print(f"  {i}/{len(todo)}", flush=True)
    save()
    return cache


def stratify(samp, interventions, cache):
    rows = []
    for _, r in samp.iterrows():
        info = interventions.get(r["nct_id"], {"names": []})
        approvals, codes, matched, ever_approved = [], False, False, False
        for n in info["names"]:
            if L._CODE.search(n):
                codes = True
            for c in candidates(n):
                for m in cache.get(c, []):
                    matched = True
                    if str(m.get("max_phase") or "").startswith("4"):
                        ever_approved = True
                    if m.get("first_approval"):
                        approvals.append(int(m["first_approval"]))
        start = int(r["start_year"])
        if any(a < start for a in approvals):
            stratum = "approved_before_start"
        elif any(a >= start for a in approvals):
            stratum = "approved_after_start"
        elif codes and not ever_approved:
            stratum = "never_approved_code"
        else:
            stratum = "other"
        rows.append({"nct_id": r["nct_id"], "stratum": stratum, "chembl_matched": matched})
    return pd.DataFrame(rows)


def _ap(y, p):
    return float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan")


def _roc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")


def contrast(w, a, b, seed=SEED):
    y = w["label"].values.astype(int)
    pa, pb = w[a].values, w[b].values
    if len(set(y)) < 2:
        return {"n": int(len(w)), "n_pos": int(y.sum())}
    rng = np.random.default_rng(seed)
    d, dr = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        d.append(_ap(y[i], pa[i]) - _ap(y[i], pb[i]))
        dr.append(_roc(y[i], pa[i]) - _roc(y[i], pb[i]))
    d, dr = np.array(d), np.array(dr)
    return {"n": int(len(w)), "n_pos": int(y.sum()),
            "auprc": round(_ap(y, pa) - _ap(y, pb), 4),
            "auprc_ci": [round(float(np.quantile(d, .025)), 4), round(float(np.quantile(d, .975)), 4)],
            "auprc_p_gt_0": round(float(np.mean(d > 0)), 3),
            "auroc": round(_roc(y, pa) - _roc(y, pb), 4),
            "auroc_ci": [round(float(np.quantile(dr, .025)), 4), round(float(np.quantile(dr, .975)), 4)]}


def main():
    samp = L.load_sample(L.N)
    interventions = json.load(open(L.INTERVENTIONS_CACHE))
    cache = chembl_table(interventions)
    strata = stratify(samp, interventions, cache)
    preds = pd.read_csv(L.PRED_PATH)
    out = {"strata_counts": strata["stratum"].value_counts().to_dict(),
           "chembl_match_rate": round(float(strata["chembl_matched"].mean()), 4),
           "definition": __doc__.split("\n\n")[1].strip(), "models": {}}
    for model in [m for m in L.MODELS if m in set(preds["model"])]:
        wide = preds[preds["model"] == model].pivot(index=["nct_id", "label"], columns="condition",
                                                    values="p").reset_index().merge(strata, on="nct_id")
        res = {}
        for s in ["approved_before_start", "approved_after_start", "never_approved_code", "other"]:
            w = wide[wide["stratum"] == s].dropna(subset=L.CONDITIONS)
            res[s] = {f"{a} - {b}": contrast(w, a, b) for a, b in CONTRASTS}
        out["models"][model] = res
        line = [f"{model:28s}"]
        for s_ in ["approved_before_start", "approved_after_start", "never_approved_code", "other"]:
            c = res[s_]["D+T - D+Tm"]
            line.append(f"{s_}: {c.get('auprc', float('nan')):+.3f} {c.get('auprc_ci')} "
                        f"(n={c['n']}, pos={c['n_pos']})")
        print("\n    ".join(line), flush=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print("strata:", out["strata_counts"], "| ChEMBL match rate:", out["chembl_match_rate"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
