"""Memorization and knowledge leakage in language-model trial-outcome prediction.

Design (nested prompts). Every trial is presented to every model under five prompt
conditions that differ ONLY in the information supplied; the instructions are identical:

  D           structural design: phase, enrolment bin, number of arms, sponsor class
  D+ID        D + the ClinicalTrials.gov identifier (NCT number)
  D+Tm        D + the title with every intervention name, alias and trial acronym masked
  D+T         D + the trial title (first 150 characters)
  D+T+ID+S    D + title + identifier + sponsor name ("fully identified")

Contrasts (paired, on trials with a valid prediction in every condition):
  D+ID - D         pure identifier recall: an NCT number carries no design information, so
                   any gain requires the model to have seen the trial (memorization)
  D+Tm - D         design semantics of the title (legitimate at decision time)
  D+T - D+Tm       knowledge tied to the named intervention or trial acronym, which for a
                   model trained after the trial started can include its later fate
  D+T - D          title content (design semantics plus intervention knowledge)
  D+T+ID+S - D     the full identified-minus-de-identified gap
  D+T+ID+S - D+T   what identifiers and sponsor add once the title is known

Recency: gaps are stratified by trial completion year; for models with a published
training cutoff, trials completed before and after the cutoff are compared.

Backends: Claude models through the Claude Code CLI with every tool and connector disabled
(--tools "" --strict-mcp-config), a neutral system prompt, and a clean working directory,
so answers come from model weights only; open-weight models through a local Ollama server
at temperature 0. Every raw response is saved (outputs/llm_raw/), so runs resume after
interruption and all summaries can be recomputed offline with --analyze-only.

Usage:
    python scripts/llm_memorization_study.py --models claude-opus-5 [--workers 2]
    python scripts/llm_memorization_study.py --models ollama:llama3.1:8b
    python scripts/llm_memorization_study.py --analyze-only
"""
import argparse
import concurrent.futures as cf
import datetime
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

CTO = "data/cto"
N, BATCH, SEED, N_BOOT = 600, 25, 0, 1000
OUT_PATH = "outputs/llm_memorization_study.json"
PRED_PATH = "outputs/llm_predictions.csv"
RAW_DIR = "outputs/llm_raw"
CONDITIONS = ["D", "D+ID", "D+Tm", "D+T", "D+T+ID+S"]
CONTRASTS = [("D+ID", "D"), ("D+Tm", "D"), ("D+T", "D+Tm"), ("D+T", "D"), ("D+T+ID+S", "D"),
             ("D+T+ID+S", "D+T")]
INTERVENTIONS_CACHE = "outputs/llm_sample_interventions.json"  # tracked: offline reproducibility
# Generic comparator terms are design information, not identities, so they are never masked.
GENERIC = {"placebo", "saline", "vehicle", "sham", "control", "standard of care", "usual care",
           "best supportive care", "normal saline", "no intervention", "observation"}
MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-5",
          "ollama:llama3.1:8b"]
# Published training-data cutoffs (year, month) where the developer documents one.
CUTOFFS = {"ollama:llama3.1:8b": (2023, 12)}
SYSTEM = ("You are a clinical-development analyst. Answer only from your own knowledge and "
          "return only the requested JSON.")
INSTRUCTION = ("Estimate the probability (0-1) that each clinical trial below meets its primary "
               "objective (trial success), using only the information given for that trial. "
               "Return ONLY a JSON array of objects {\"id\": \"Tnn\", \"p\": 0.NN}, one per trial.\n")
_lock = threading.Lock()


def _bucket(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "unknown"
    return "<50" if v < 50 else "50-200" if v < 200 else "200-500" if v < 500 else ">=500"


def _na(v):
    return "NA" if pd.isna(v) else v


def describe(r, cond):
    parts = []
    if "ID" in cond:
        parts.append(f"nct={r['nct_id']}")
    parts += [f"phase={_na(r['phase'])}", f"enroll_bucket={_bucket(r['enrollment'])}",
              f"arms={_na(r['number_of_arms'])}", f"sponsor_class={_na(r['source_class'])}"]
    tokens = cond.split("+")
    if "T" in tokens:
        parts.append(f"title={str(r['brief_title'])[:150]}")
    if "Tm" in tokens:
        parts.append(f"title={str(r['title_masked'])[:150]}")
    if cond.endswith("+S"):
        parts.append(f"sponsor={_na(r['source'])}")
    return " ".join(parts)


def build_prompt(chunk, cond):
    lines = [INSTRUCTION]
    for j, (_, r) in enumerate(chunk.iterrows()):
        lines.append(f"id=T{j:02d} {describe(r, cond)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- backends
_CLI_CWD = tempfile.mkdtemp(prefix="llm_study_")  # no project settings or CLAUDE.md


def call_claude(prompt, model, timeout=300):
    cmd = ["claude", "-p", prompt, "--model", model, "--tools", "", "--strict-mcp-config",
           "--no-session-persistence", "--system-prompt", SYSTEM]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=_CLI_CWD)
    return r.stdout.strip()


def call_ollama(prompt, model, timeout=600):
    body = {"model": model, "system": SYSTEM, "prompt": prompt, "stream": False,
            # num_predict caps runaway generations (a full answer needs ~400 tokens)
            "options": {"temperature": 0, "seed": SEED, "num_ctx": 8192, "num_predict": 1500}}
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp).get("response", "").strip()


def call(prompt, model):
    if model.startswith("ollama:"):
        return call_ollama(prompt, model.split(":", 1)[1])
    return call_claude(prompt, model)


_PAIR = re.compile(r'T(\d{1,3})"?\s*[,:]?\s*(?:"p"|p)\s*"?\s*:\s*"?([01](?:\.\d+)?|\.\d+)')


def parse(text, n):
    """Parse the JSON array of {id, p}; fall back to a tolerant regex for near-JSON output
    (e.g. a missing "id" key), which deterministic (temperature-0) retries would repeat."""
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    out = {}
    try:
        m = re.search(r"\[.*\]", text, flags=re.DOTALL)
        for o in json.loads(m.group(0) if m else text):
            j = int(re.sub(r"\D", "", str(o.get("id", ""))))
            if 0 <= j < n:
                out[j] = min(max(float(o.get("p")), 0.0), 1.0)
    except (ValueError, TypeError, AttributeError):
        for j, p in _PAIR.findall(text):
            if 0 <= int(j) < n:
                out[int(j)] = min(max(float(p), 0.0), 1.0)
    if not out:
        raise ValueError("no predictions parsed")
    return out


# --------------------------------------------------------------------------- masking
def fetch_interventions(ncts):
    """Intervention names, aliases and trial acronyms from ClinicalTrials.gov (cached)."""
    cache = json.load(open(INTERVENTIONS_CACHE)) if os.path.exists(INTERVENTIONS_CACHE) else {}
    todo = [x for x in ncts if x not in cache]
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        url = ("https://clinicaltrials.gov/api/v2/studies?" + urllib.parse.urlencode(
            {"filter.ids": ",".join(chunk), "pageSize": 1000, "format": "json",
             "fields": "NCTId,Acronym,InterventionName,InterventionOtherName"}))
        with urllib.request.urlopen(url, timeout=60) as resp:
            studies = json.load(resp).get("studies", [])
        for st in studies:
            ps = st.get("protocolSection", {})
            nct = ps.get("identificationModule", {}).get("nctId")
            names = []
            for iv in (ps.get("armsInterventionsModule", {}) or {}).get("interventions", []) or []:
                names.append(iv.get("name") or "")
                names += iv.get("otherNames") or []
            cache[nct] = {"acronym": ps.get("identificationModule", {}).get("acronym"),
                          "names": [x for x in names if x]}
        for x in chunk:
            cache.setdefault(x, {"acronym": None, "names": []})
        time.sleep(0.2)
    if todo:  # write atomically, and only when something new was fetched (parallel runs read it)
        os.makedirs(os.path.dirname(INTERVENTIONS_CACHE), exist_ok=True)
        tmp = f"{INTERVENTIONS_CACHE}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(cache, fh, indent=1)
        os.replace(tmp, INTERVENTIONS_CACHE)
    return cache


# Words that describe formulation, route or design rather than identity (never masked).
_STOP = GENERIC | {"injection", "injections", "tablet", "tablets", "capsule", "capsules", "oral",
                   "intravenous", "subcutaneous", "infusion", "topical", "solution", "cream",
                   "ointment", "spray", "patch", "dose", "doses", "dosing", "therapy", "treatment",
                   "combination", "placebo", "matching", "standard", "active", "comparator",
                   "group", "arm", "daily", "weekly", "extended", "release", "vaccine",
                   "vaccination", "surgery", "procedure", "device", "behavioral", "training",
                   "program", "education", "exercise", "diet", "supplement", "radiation",
                   "chemotherapy", "immunotherapy", "cells", "study", "drug", "product", "high",
                   "low", "healthy", "volunteers", "patients", "subjects", "versus", "plus",
                   "alone", "single", "multiple", "ascending", "phase", "trial"}
# Development codes such as HH-003, SY-1425, ABBV-951, BNT162b2.
_CODE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,6}[- ]?\d{2,6}[A-Za-z0-9]*(?![A-Za-z0-9])")


def mask_title(title, info):
    """Mask intervention identities in a title: full intervention names and aliases, their
    distinctive words (formulation/route/design words are kept), development codes, and the
    trial acronym. Design information such as 'placebo-controlled' or the indication stays."""
    terms = set()
    for name in info.get("names", []):
        name = name.strip()
        if len(name) < 3 or name.lower() in _STOP:
            continue
        terms.add(name)
        for tok in re.split(r"[^A-Za-z0-9-]+", name):
            if len(tok) >= 5 and tok.lower() not in _STOP and not tok.isdigit():
                terms.add(tok)
    out = str(title)
    for t in sorted(terms, key=len, reverse=True):
        out = re.sub(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", "[INTERVENTION]",
                     out, flags=re.IGNORECASE)
    out = _CODE.sub("[INTERVENTION]", out)
    acr = (info.get("acronym") or "").strip()
    if len(acr) >= 2:
        out = re.sub(r"(?<![A-Za-z0-9])" + re.escape(acr) + r"(?![A-Za-z0-9])", "[ACRONYM]", out,
                     flags=re.IGNORECASE)
    return re.sub(r"(\[INTERVENTION\][\s,/+-]*){2,}", "[INTERVENTION] ", out).strip()


# --------------------------------------------------------------------------- querying
def load_sample(n=N):
    hl = pd.read_csv(os.path.join(CTO, "human_labels_2020_2024.csv"), low_memory=False)
    hl = hl[hl["labels"].isin([0.0, 1.0])].copy()
    hl["labels"] = hl["labels"].astype(int)
    hl["start_year"] = pd.to_datetime(hl["start_date"], errors="coerce").dt.year
    hl["completion_date"] = pd.to_datetime(hl["completion_date"], errors="coerce")
    pool = hl[hl["start_year"] > 2020].dropna(subset=["brief_title", "completion_date"])
    samp = pool.sample(n=min(n, len(pool)), random_state=SEED).reset_index(drop=True)
    info = fetch_interventions(list(samp["nct_id"]))
    samp["title_masked"] = [mask_title(t, info.get(x, {})) for t, x in zip(samp["brief_title"], samp["nct_id"])]
    return samp


def _raw_path(model):
    return os.path.join(RAW_DIR, re.sub(r"[^A-Za-z0-9_.-]", "_", model) + ".jsonl")


def _done(model):
    done = {}
    if os.path.exists(_raw_path(model)):
        for line in open(_raw_path(model)):
            rec = json.loads(line)
            if rec.get("preds") is not None:
                done[(rec["condition"], rec["batch"])] = rec
    return done


def query_model(samp, model, workers=1, conditions=None):
    os.makedirs(RAW_DIR, exist_ok=True)
    done = _done(model)
    jobs = [(c, b) for c in (conditions or CONDITIONS) for b in range(0, len(samp), BATCH)
            if (c, b) not in done]
    print(f"{model}: {len(done)} batches cached, {len(jobs)} to query", flush=True)

    def work(job):
        cond, b = job
        chunk = samp.iloc[b:b + BATCH]
        prompt = build_prompt(chunk, cond)
        rec = {"model": model, "condition": cond, "batch": b, "nct_ids": list(chunk["nct_id"]),
               "prompt": prompt, "response": None, "preds": None, "attempts": 0,
               "time": datetime.datetime.now().isoformat(timespec="seconds")}
        for attempt in range(3):
            rec["attempts"] = attempt + 1
            try:
                rec["response"] = call(prompt, model)
                rec["preds"] = {str(k): v for k, v in parse(rec["response"], len(chunk)).items()}
                break
            except Exception as e:  # noqa: BLE001 -- timeouts, empty or malformed output: retry
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(5 * (attempt + 1))
        with _lock, open(_raw_path(model), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return cond, b, rec["preds"] is not None

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (cond, b, ok) in enumerate(ex.map(work, jobs), 1):
            if i % 10 == 0 or not ok:
                print(f"  {model} {i}/{len(jobs)} {cond} batch {b} {'ok' if ok else 'FAILED'}",
                      flush=True)


def collect(samp, models):
    rows = []
    for model in models:
        for (cond, b), rec in _done(model).items():
            ids = rec["nct_ids"]
            for j, nct in enumerate(ids):
                p = rec["preds"].get(str(j))
                rows.append({"model": model, "condition": cond, "nct_id": nct,
                             "p": np.nan if p is None else p})
    preds = pd.DataFrame(rows)
    meta = samp[["nct_id", "labels", "start_year", "completion_date"]].rename(columns={"labels": "label"})
    return preds.merge(meta, on="nct_id", how="left")


# --------------------------------------------------------------------------- analysis
def _auprc(y, p):
    return float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan")


def _auroc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")


def _q(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return [round(float(np.quantile(a, .025)), 4), round(float(np.quantile(a, .975)), 4)] if len(a) else [None, None]


def summarize(w, seed=SEED):
    """Per-condition AUPRC/AUROC and paired contrasts with trial-level bootstrap CIs."""
    w = w.dropna(subset=CONDITIONS)
    y = w["label"].values.astype(int)
    P = {c: w[c].values for c in CONDITIONS}
    out = {"n": int(len(w)), "n_pos": int(y.sum()), "base_rate": round(float(y.mean()), 4) if len(y) else None}
    if len(w) < 20 or len(set(y)) < 2:
        return out
    rng = np.random.default_rng(seed)
    boot = {c: [] for c in CONDITIONS}
    boot_roc = {c: [] for c in CONDITIONS}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        for c in CONDITIONS:
            boot[c].append(_auprc(y[idx], P[c][idx]))
            boot_roc[c].append(_auroc(y[idx], P[c][idx]))
    boot = {c: np.array(v) for c, v in boot.items()}
    boot_roc = {c: np.array(v) for c, v in boot_roc.items()}
    out["conditions"] = {c: {"auprc": round(_auprc(y, P[c]), 4), "auprc_ci": _q(boot[c]),
                             "auroc": round(_auroc(y, P[c]), 4), "auroc_ci": _q(boot_roc[c])}
                         for c in CONDITIONS}
    out["contrasts"] = {}
    for a, b in CONTRASTS:
        d = boot[a] - boot[b]
        dr = boot_roc[a] - boot_roc[b]
        out["contrasts"][f"{a} - {b}"] = {
            "auprc": round(_auprc(y, P[a]) - _auprc(y, P[b]), 4), "auprc_ci": _q(d),
            "auprc_p_gt_0": round(float(np.mean(d > 0)), 3),
            "auroc": round(_auroc(y, P[a]) - _auroc(y, P[b]), 4), "auroc_ci": _q(dr)}
    return out


def analyze(preds, run_meta):
    out = {"run": run_meta, "design": {"conditions": CONDITIONS,
                                       "contrasts": [f"{a} - {b}" for a, b in CONTRASTS]},
           "n_sample": int(preds["nct_id"].nunique()),
           "base_rate": round(float(preds.drop_duplicates("nct_id")["label"].mean()), 4),
           "models": {}}
    for model in [m for m in MODELS if m in set(preds["model"])] + \
                 sorted(set(preds["model"]) - set(MODELS)):
        wide = preds[preds["model"] == model].pivot(
            index=["nct_id", "label", "completion_date"], columns="condition", values="p").reset_index()
        coverage = {c: round(float(wide[c].notna().mean()), 4) for c in CONDITIONS if c in wide}
        res = {"coverage": coverage, "all": summarize(wide)}
        yrs = pd.to_datetime(wide["completion_date"]).dt.year
        res["by_completion_year"] = {
            label: summarize(wide[mask]) for label, mask in
            (("2021-2022", yrs <= 2022), ("2023", yrs == 2023), ("2024", yrs == 2024))}
        if model in CUTOFFS:
            cy, cm = CUTOFFS[model]
            cut = pd.Timestamp(year=cy, month=cm, day=1) + pd.offsets.MonthEnd(0)
            cd = pd.to_datetime(wide["completion_date"])
            res["cutoff"] = f"{cy}-{cm:02d}"
            res["before_cutoff"] = summarize(wide[cd <= cut])
            res["after_cutoff"] = summarize(wide[cd > cut])
        out["models"][model] = res
        a = res["all"]
        if "contrasts" in a:
            c = a["contrasts"]
            print(f"{model:28s} n={a['n']} D {a['conditions']['D']['auprc']:.3f}  "
                  f"ID-gain {c['D+ID - D']['auprc']:+.3f} {c['D+ID - D']['auprc_ci']}  "
                  f"full-gap {c['D+T+ID+S - D']['auprc']:+.3f} {c['D+T+ID+S - D']['auprc_ci']}",
                  flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="LLM memorization study on CTO trials.")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--workers", type=int, default=1, help="concurrent requests per model")
    ap.add_argument("--conditions", nargs="*", default=None, help="subset of prompt conditions")
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    os.makedirs("outputs", exist_ok=True)
    samp = load_sample(args.n)
    if not args.analyze_only:
        print(f"n={len(samp)} base_rate={samp['labels'].mean():.3f}", flush=True)
        for model in args.models:
            query_model(samp, model, workers=args.workers, conditions=args.conditions)
    models = [m for m in MODELS + args.models if os.path.exists(_raw_path(m))]
    preds = collect(samp, list(dict.fromkeys(models)))
    preds.to_csv(PRED_PATH, index=False)
    try:
        cli = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    except OSError:
        cli = "unavailable"
    run_meta = {"analyzed": datetime.date.today().isoformat(), "claude_cli_version": cli,
                "n_sample": len(samp), "sample_seed": SEED, "batch_size": BATCH,
                "n_successes": int(samp["labels"].sum()),
                "titles_masked_fraction": round(float((samp["title_masked"] != samp["brief_title"]).mean()), 4),
                "claude_flags": "--tools '' --strict-mcp-config --no-session-persistence --system-prompt",
                "ollama_options": "temperature 0, seed 0"}
    out = analyze(preds, run_meta)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print("DONE ->", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
