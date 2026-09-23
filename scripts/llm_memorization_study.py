"""Training-data memorization in LLM trial-outcome prediction.

Three axes, on CTO trials that started after 2020, scored against CTO's curated
human labels:
  (1) MODEL TIER -- the same experiment across Claude models of different product
      tiers (Haiku < Sonnet < Opus); parameter counts are not public.
  (2) IDENTIFIED vs DE-IDENTIFIED prompts -- with vs without the trial identifier,
      verbatim title (which names the drug and indication) and sponsor. The gap is an
      upper bound on memorization: identified prompts also carry the title's
      legitimate design information.
  (3) RECENCY -- the gap stratified by trial start year (tertiles). Recall of trials
      seen in training should weaken for trials near the models' training cutoffs.

Every call's parsed probability is saved (outputs/llm_predictions.csv), so all
summaries -- including paired trial-level bootstrap CIs (1,000 resamples) for the
identified, de-identified and gap AUPRC -- can be recomputed offline with
``--analyze-only``. Metrics are computed on trials with a valid prediction in BOTH
conditions for a model, so the identified/de-identified contrast is paired.

Requires the Claude Code CLI (``claude``) for querying; models are called with the
CLI's default sampling settings, so outputs are stochastic.

Usage:
    python scripts/llm_memorization_study.py                 # query models + analyse
    python scripts/llm_memorization_study.py --analyze-only  # recompute from saved predictions
"""
import argparse
import datetime
import json
import os
import re
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

CTO = "data/cto"
N, BATCH, SEED, N_BOOT = 250, 25, 0, 1000
OUT_PATH = "outputs/llm_memorization_study.json"
PRED_PATH = "outputs/llm_predictions.csv"
# Product-tier ladder (ascending): does memorization grow with model tier?
MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-5"]
RECENCY_MODEL = "claude-opus-5"


def _call(prompt, model, timeout=200):
    try:
        return subprocess.run(["claude", "-p", prompt, "--model", model],
                              capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def _json(text):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    return json.loads(m.group(0)) if m else json.loads(text)


def _bucket(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "unknown"
    return "<50" if v < 50 else "50-200" if v < 200 else "200-500" if v < 500 else ">=500"


def prompt_identified(chunk):
    L = ["You are a clinical-development analyst. For each trial, estimate the probability "
         "(0-1) the drug program SUCCEEDS at this phase, using ONLY pre-registration design "
         "info. Return ONLY a JSON array of {\"id\":\"Tnn\",\"p\":0.NN}.\n"]
    for j, (_, r) in enumerate(chunk.iterrows()):
        L.append(f"id=T{j:02d} nct={r['nct_id']} phase={r['phase']} "
                 f"title={str(r['brief_title'])[:150]} enroll={r['enrollment']} "
                 f"arms={r['number_of_arms']} sponsor={r['source']}")
    return "\n".join(L)


def prompt_deident(chunk):
    L = ["You are a clinical-development analyst. Estimate the probability (0-1) each "
         "DE-IDENTIFIED trial's program SUCCEEDS at its phase, from anonymised structural "
         "design only (reason from population base rates, not any specific trial). Return "
         "ONLY a JSON array of {\"id\":\"Tnn\",\"p\":0.NN}.\n"]
    for j, (_, r) in enumerate(chunk.iterrows()):
        L.append(f"id=T{j:02d} phase={r['phase']} enroll_bucket={_bucket(r['enrollment'])} "
                 f"arms={r['number_of_arms'] if not pd.isna(r['number_of_arms']) else 'NA'} "
                 f"sponsor_class={r['source_class'] if not pd.isna(r['source_class']) else 'NA'}")
    return "\n".join(L)


PROMPTS = {"identified": prompt_identified, "deidentified": prompt_deident}


def run(samp, model, builder):
    """Query ``model`` in batches; return one probability per trial (NaN if unparsed)."""
    preds = {}
    for i in range(0, len(samp), BATCH):
        chunk = samp.iloc[i:i + BATCH]
        for _ in range(2):  # one retry on an unparseable response
            try:
                for o in _json(_call(builder(chunk), model)):
                    j = int(re.sub(r"\D", "", str(o.get("id", ""))))
                    if 0 <= j < len(chunk):
                        preds[i + j] = min(max(float(o.get("p")), 0.0), 1.0)
                break
            except (ValueError, TypeError, AttributeError):
                continue
    return np.array([preds.get(k, np.nan) for k in range(len(samp))])


def _auprc(y, p):
    return float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan")


def _q(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return [None, None]
    return [round(float(np.quantile(a, .025)), 4), round(float(np.quantile(a, .975)), 4)]


def paired_summary(y, p_id, p_de, seed=SEED):
    """Identified / de-identified / gap AUPRC with paired trial-level bootstrap CIs."""
    ok = np.isfinite(p_id) & np.isfinite(p_de)
    y, p_id, p_de = y[ok], p_id[ok], p_de[ok]
    ai, ad = _auprc(y, p_id), _auprc(y, p_de)
    rng = np.random.default_rng(seed)
    bi, bd = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        bi.append(_auprc(y[idx], p_id[idx]))
        bd.append(_auprc(y[idx], p_de[idx]))
    bi, bd = np.array(bi), np.array(bd)
    return {"n_paired": int(ok.sum()), "base_rate": round(float(y.mean()), 4),
            "identified_auprc": round(ai, 4), "identified_auprc_ci": _q(bi),
            "deident_auprc": round(ad, 4), "deident_auprc_ci": _q(bd),
            "memorization_leakage": round(ai - ad, 4), "memorization_leakage_ci": _q(bi - bd),
            "memorization_leakage_p_gt_0": round(float(np.mean(bi - bd > 0)), 3)}


def load_sample():
    hl = pd.read_csv(os.path.join(CTO, "human_labels_2020_2024.csv"), low_memory=False)
    hl = hl[hl["labels"].isin([0.0, 1.0])].copy()
    hl["labels"] = hl["labels"].astype(int)
    hl["start_year"] = pd.to_datetime(hl["start_date"], errors="coerce").dt.year
    test = hl[hl["start_year"] > 2020].dropna(subset=["brief_title"])
    return test.sample(n=min(N, len(test)), random_state=SEED).reset_index(drop=True)


def query_models(samp):
    rows = []
    for model in MODELS:
        for cond, builder in PROMPTS.items():
            p = run(samp, model, builder)
            rows.append(pd.DataFrame({"nct_id": samp["nct_id"], "start_year": samp["start_year"],
                                      "label": samp["labels"], "model": model,
                                      "condition": cond, "p": p}))
            print(f"  {model:28s} {cond:12s} parsed {np.isfinite(p).sum()}/{len(p)}", flush=True)
        pd.concat(rows).to_csv(PRED_PATH, index=False)  # checkpoint after each model
    return pd.concat(rows, ignore_index=True)


def analyze(preds, meta):
    out = {**meta, "models": {}, "recency": None}
    wide = preds.pivot(index=["nct_id", "start_year", "label", "model"],
                       columns="condition", values="p").reset_index()
    for model in [m for m in MODELS if m in set(wide["model"])]:
        w = wide[wide["model"] == model]
        s = paired_summary(w["label"].values, w["identified"].values, w["deidentified"].values)
        out["models"][model] = s
        print(f"  {model:28s} n={s['n_paired']} identified {s['identified_auprc']:.3f} "
              f"{s['identified_auprc_ci']}  de-ident {s['deident_auprc']:.3f} {s['deident_auprc_ci']}  "
              f"gap {s['memorization_leakage']:+.3f} {s['memorization_leakage_ci']}", flush=True)

    w = wide[wide["model"] == RECENCY_MODEL]
    if len(w):
        yrs = w["start_year"]
        q1, q2 = (int(v) for v in yrs.quantile([1 / 3, 2 / 3]))
        bins = [(f"<={q1}", yrs <= q1), (f"{q1 + 1}-{q2}", (yrs > q1) & (yrs <= q2)),
                (f">{q2}", yrs > q2)]
        rec = []
        for name, mask in bins:
            m = mask.values
            if m.sum() < 8:
                continue
            s = paired_summary(w["label"].values[m], w["identified"].values[m],
                               w["deidentified"].values[m])
            rec.append({"bin": name, "n": int(m.sum()), **s})
            print(f"  recency {name:8s} n={int(m.sum())} gap {s['memorization_leakage']:+.3f} "
                  f"{s['memorization_leakage_ci']}", flush=True)
        out["recency"] = {"model": RECENCY_MODEL, "bins": rec}
    return out


def main():
    ap = argparse.ArgumentParser(description="LLM memorization study on CTO trials.")
    ap.add_argument("--analyze-only", action="store_true",
                    help=f"recompute summaries from {PRED_PATH} without querying models")
    args = ap.parse_args()
    os.makedirs("outputs", exist_ok=True)

    if args.analyze_only:
        preds = pd.read_csv(PRED_PATH)
        meta = json.load(open(OUT_PATH)).get("run", {}) if os.path.exists(OUT_PATH) else {}
    else:
        samp = load_sample()
        print(f"n={len(samp)} base_rate={samp['labels'].mean():.3f} start-year range "
              f"[{int(samp['start_year'].min())},{int(samp['start_year'].max())}]")
        try:
            cli = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
        except OSError:
            cli = "unavailable"
        meta = {"run": {"date": datetime.date.today().isoformat(), "claude_cli_version": cli,
                        "models": MODELS, "n_sample": len(samp), "sample_seed": SEED,
                        "batch_size": BATCH}}
        preds = query_models(samp)

    out = analyze(preds, {"run": meta.get("run", meta), "n": int(preds["nct_id"].nunique()),
                          "test_base_rate": round(float(preds.drop_duplicates("nct_id")["label"].mean()), 4)})
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print("DONE ->", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
