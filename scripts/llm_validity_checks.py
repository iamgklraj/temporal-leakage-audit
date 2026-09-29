"""Offline validity checks of the language-model study (response to a domain review).

No language model is queried. Everything is recomputed from saved artefacts:
outputs/llm_predictions.csv, outputs/llm_raw/*.jsonl (prompt provenance only),
outputs/llm_sample_interventions.json, outputs/llm_sample_chembl.json and the CTO human-label
table. Metrics and bootstraps reuse llm_memorization_study.summarize / holm / primary_tests
(trial-level paired bootstrap, 1,000 resamples, seed 0) and llm_knowledge_dating.contrast.

Sections of outputs/llm_validity_checks.json
  provenance       regenerated prompts vs the prompts actually sent (outputs/llm_raw)
  claims           L1 enrolment type, L2 tabular model on the design-only fields, L3 label
                   semantics, L5 completion-date range vs documented training cutoffs,
                   L7 identifier+sponsor increment
  masking_audit    unchanged titles, automated residual-name scan (all 600 titles), manual
                   review of 100 masked titles (seed 0), over-masking by the code/acronym rules
  contrasts        named-intervention (D+T - D+Tm) and masked-title (D+Tm - D) contrasts per
                   model: as published, excluding COVID-19 trials, excluding titles that still
                   leak an intervention name, excluding withdrawn trials, all exclusions
  chembl_refined   refined approval strata (curated; see the constants below) and the
                   named-intervention gain per model per stratum

Usage:  OMP_NUM_THREADS=2 python scripts/llm_validity_checks.py
"""
import difflib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_knowledge_dating as KD  # noqa: E402
import llm_memorization_study as L  # noqa: E402

OUT = "outputs/llm_validity_checks.json"
SEED = 0
KEY = ["D+T - D+Tm", "D+Tm - D"]
COVID_RE = r"covid|sars-cov|coronavirus|ncov"

# Training-data cutoffs documented by Anthropic (model pages on platform.claude.com, read
# 2026-09-28) and Meta (Llama 3.1). Year, month.
DOC_CUTOFFS = {
    "claude-haiku-4-5-20251001": ((2025, 7), "https://platform.claude.com/docs/en/models/haiku-4-5/overview"),
    "claude-sonnet-5": ((2026, 1), "https://platform.claude.com/docs/en/models/sonnet-5/overview"),
    "claude-opus-4-8": ((2026, 1), "https://platform.claude.com/docs/en/models/opus-4-8/overview"),
    "claude-opus-5": ((2026, 5), "https://platform.claude.com/docs/en/models/opus-5/overview"),
    "ollama:llama3.1:8b": ((2023, 12), "llm_memorization_study.CUTOFFS"),
}


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


# --------------------------------------------------------------------------- provenance
def prompt_provenance(samp):
    """Rebuild every prompt line from the sample and compare with the saved raw prompts."""
    out = {}
    for model in L.MODELS:
        sent = {}
        for line in open(L._raw_path(model)):
            rec = json.loads(line)
            rows = [x for x in rec["prompt"].split("\n") if x.startswith("id=T")]
            for j, nct in enumerate(rec["nct_ids"]):
                sent.setdefault((rec["condition"], nct), set()).add(rows[j])
        mism = 0
        for cond in L.CONDITIONS:
            for b in range(0, len(samp), L.BATCH):
                chunk = samp.iloc[b:b + L.BATCH]
                rows = [x for x in L.build_prompt(chunk, cond).split("\n") if x.startswith("id=T")]
                mism += sum(rows[j] not in sent.get((cond, n), set()) for j, n in enumerate(chunk["nct_id"]))
        out[model] = {"prompt_lines_checked": len(samp) * len(L.CONDITIONS), "mismatches": int(mism)}
    return out


# --------------------------------------------------------------------------- L2 tabular
def design_frame(samp):
    return pd.DataFrame({"phase": samp["phase"].fillna("NA").astype(str),
                         "enroll_bucket": samp["enrollment"].map(L._bucket),
                         "arms": samp["number_of_arms"].map(lambda v: "NA" if pd.isna(v) else str(v)),
                         "sponsor_class": samp["source_class"].fillna("NA").astype(str)})


def tabular_design_model(samp):
    X, y = design_frame(samp), samp["labels"].values.astype(int)
    sets = {"all_D_fields": ["phase", "enroll_bucket", "arms", "sponsor_class"],
            "without_enrol_bin": ["phase", "arms", "sponsor_class"],
            "enrol_bin_only": ["enroll_bucket"]}
    models = {"logistic_regression": lambda: LogisticRegression(max_iter=2000),
              "gradient_boosting": lambda: GradientBoostingClassifier(random_state=SEED)}
    out = {"cv": "5-fold stratified, shuffled; one-hot encoding; pooled out-of-fold predictions",
           "results": {}}
    for sname, cols in sets.items():
        for mname, mk in models.items():
            reps = []
            for rep in range(5):
                oof = np.zeros(len(y))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X, y):
                    pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"), mk())
                    pipe.fit(X.iloc[tr][cols], y[tr])
                    oof[te] = pipe.predict_proba(X.iloc[te][cols])[:, 1]
                reps.append((L._auroc(y, oof), L._auprc(y, oof)))
            reps = np.array(reps)
            out["results"][f"{sname}|{mname}"] = {
                "seed0_auroc": round(reps[0, 0], 4), "seed0_auprc": round(reps[0, 1], 4),
                "mean5seeds_auroc": round(float(reps[:, 0].mean()), 4),
                "mean5seeds_auprc": round(float(reps[:, 1].mean()), 4)}
    return out


# --------------------------------------------------------------------------- masking audit
def mask_title_traced(title, info):
    """Same logic as llm_memorization_study.mask_title, recording what each rule replaced."""
    terms = set()
    for name in info.get("names", []):
        name = name.strip()
        if len(name) < 3 or name.lower() in L._STOP:
            continue
        terms.add(name)
        for tok in re.split(r"[^A-Za-z0-9-]+", name):
            if len(tok) >= 5 and tok.lower() not in L._STOP and not tok.isdigit():
                terms.add(tok)
    out, by_name = str(title), []
    for t in sorted(terms, key=len, reverse=True):
        pat = r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])"
        by_name += re.findall(pat, out, flags=re.IGNORECASE)
        out = re.sub(pat, "[INTERVENTION]", out, flags=re.IGNORECASE)
    by_code = L._CODE.findall(out)
    out = L._CODE.sub("[INTERVENTION]", out)
    by_acr = []
    acr = (info.get("acronym") or "").strip()
    if len(acr) >= 2:
        pat = r"(?<![A-Za-z0-9])" + re.escape(acr) + r"(?![A-Za-z0-9])"
        by_acr = re.findall(pat, out, flags=re.IGNORECASE)
        out = re.sub(pat, "[ACRONYM]", out, flags=re.IGNORECASE)
    out = re.sub(r"(\[INTERVENTION\][\s,/+-]*){2,}", "[INTERVENTION] ", out).strip()
    return out, {"name_rule": by_name, "code_rule": by_code, "acronym_rule": by_acr}


# Disease, pathogen, target or biomarker strings that the development-code / acronym rules
# should never mask (classification of every code/acronym replacement in the 600 titles).
OVERMASK_CODE = re.compile(r"^(COVID[- ]?19|CD\d+|HPV-?\d+|V600[A-Z]?|BRAFV600[A-Z]?|[A-Z]\d{2,3}[A-Z]|"
                           r"IL-?\d+|FGF\d+|CCR\d+|CXCR\d+|TLR\d+|CLDN\d+)$", re.I)
OVERMASK_ACR = re.compile(r"^(ALL|AML|CLL|MG|MS|SARS(-CoV-2)?|COVID-?19|BRCA\d?|CAPACITY)$", re.I)

COMMON = {"mri", "cmr", "pet", "ct", "soc", "ns", "pbo", "inj", "hcl", "adc", "car", "dna", "rna",
          "iv", "sc", "im", "po", "art", "and", "with", "for", "the", "gel", "acid", "cell", "free",
          "base", "sodium", "salt", "plus", "mg", "ml", "kg", "part", "arm", "type", "new", "pre",
          "post", "low", "high", "dose", "oral", "eye", "drops", "drop", "mab", "anti", "cap", "tab",
          "tabs", "caps", "inhaler", "mock", "inert", "oil", "seed", "sesame", "powder", "food",
          "ready", "use", "dual", "local", "standard", "care", "support", "module", "switch", "test",
          "kit", "scan", "imaging", "control", "sham", "usual", "water", "normal", "regimen", "agent",
          "agents", "inhibitor", "inhibitors", "antibody", "antibodies", "monoclonal", "receptor",
          "agonist", "antagonist", "cells", "therapy", "therapies", "gene", "virus", "viral",
          "protein", "human", "recombinant", "live", "attenuated", "booster", "adjuvant", "group",
          "cohort", "phase", "study", "trial", "medicine", "medication", "medications",
          "investigational", "product", "products", "drug", "drugs", "pain", "blood", "plasma",
          "serum", "surgery", "hormone", "insulin", "vitamin", "oxygen", "saline", "glucose",
          "nitric", "oxide", "cancer", "tumor", "tumour", "covid-19", "covid", "sars-cov-2", "hiv",
          "hpv", "b-cell", "t-cell", "t-cells", "car-t", "nk", "dc", "dcs", "pd-1", "pd-l1",
          "ctla-4", "her2", "egfr", "vegf", "il", "tnf", "jak", "btk", "ga", "zr", "lu", "tc",
          "f-18", "18f", "68ga", "89zr", "177lu", "f18", "sars", "mrna", "vlp", "aav", "active",
          "comparator", "experimental", "intervention", "immunotherapy", "radiotherapy", "chemo",
          "chemort", "rt", "gy", "ebrt", "sbrt", "imrt", "fu", "5fu", "5-fu", "nsaid", "nsaids",
          "ppi", "ppis", "tki", "tkis", "ivig", "ig", "igg", "ecmo", "cpap", "tms", "tdcs", "rtms",
          "cbt", "act", "app", "web", "online", "video", "virtual", "reality", "vr", "exercise",
          "yoga", "diet", "fasting", "sleep", "light", "music", "massage", "acupuncture", "cream",
          "lotion"}
TARGET = re.compile(r"^(anti-)?(CD\d+|Trop-?2|TRBC1|CEA|PSMA|HER-?2|EGFR|BCMA|CMV|HIV(-1)?|HPV-?\d*|"
                    r"PD-?L?1|CTLA-?4|VEGF|TNF|IL-?\d+|GLP-?1|FLT3|BRCA\d?|KRAS|BRAF|TIGIT|LAG-?3|"
                    r"FGFR\d?|CGRP|PCSK9|SGLT-?2|DPP-?4|GM-CSF|RNA|DNA|mRNA|SARS-CoV-2|COVID-?19|RSV|"
                    r"CAR-T|B-ALL)$", re.I)
CLASS_WORDS = {"vaccine", "corticosteroid", "bronchodilator", "agonist", "insulin", "weight",
               "platelet", "antibody", "antibodie", "inhibitor", "therapy", "therapie", "molecule",
               "drug", "agent"}
INN = re.compile(r"^[A-Za-z-]*(mab|tinib|ciclib|parib|rafenib|lisib|zomib|platin|taxel|rubicin|"
                 r"previr|asvir|buvir|gliflozin|gliptin|glutide|sartan|pril|olol|dipine|vastatin|"
                 r"conazole|floxacin|mycin|cillin|cycline|lukast|setron|triptan|prazole|parin|"
                 r"xaban|gatran|dronate|cept|kinra|leucel|tidine|degib|oxetine|pramine|azepam|"
                 r"azolam|caine|olone|limus|trexate|siran|tegravir|navir|ciclovir|lutamide|relix|"
                 r"glitazone|formin|nidazole|profen|coxib|semide|thiazide|estrant|ribine|citabine|"
                 r"tecan|kinib|imod|vudine|fovir|dotide|tide|stim|ermin|leukin)$", re.I)
INN_WHITELIST = {"peptide", "nucleotide", "oligonucleotide", "polypeptide", "dipeptide",
                 "tripeptide", "glycopeptide", "concept", "except", "accept", "intercept"}
GENERIC_SUBSTANCE = {"ALANINE", "ARGININE", "BUTANE", "CINNAMON", "CYTOSINE", "ESTROGEN", "FIBRIN",
                     "GALLIUM", "INOSITOL", "MALTODEXTRIN", "PHENYLALANINE", "PLATELETS",
                     "SESAME OIL", "TECHNETIUM", "WATER", "OXYGEN", "GLUCOSE", "SODIUM CHLORIDE",
                     "ALBUMIN HUMAN", "NITROGEN", "CALCIUM", "COPPER", "IRON", "ZINC", "MAGNESIUM",
                     "POTASSIUM", "SODIUM", "TESTOSTERONE", "ESTRADIOL", "HYDROCORTISONE"}
DOSE = re.compile(r"\b\d+(\.\d+)?\s*(mg|mcg|µg|ug|g|ml|mL|iu|IU|units?|%|mg/kg|mg/m2|gy|Gy)\b", re.I)
ABBR = re.compile(r"\[INTERVENTION\]\s*\(([A-Za-z][A-Za-z0-9-]{1,9})\)")


def _has(term, text):
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", text, re.I) is not None


def identity_terms(info, chembl):
    """Intervention identities for one trial: CT.gov names/aliases (full strings, list items and
    distinctive tokens, incl. short upper-case/alphanumeric tokens that the masker skips) and the
    ChEMBL synonyms/preferred names they resolve to (from the cache)."""
    full, toks, chem = set(), set(), set()
    for name in info["names"]:
        c = " ".join(DOSE.sub(" ", re.sub(r"[®™©]", "", name)).split())
        for p in [c] + re.findall(r"\(([^)]*)\)", c) + re.split(r",|;| or | and |/|\+|\(|\)", c):
            p = p.strip(" -.")
            if len(p) >= 3 and p.lower() not in L._STOP and p.lower() not in COMMON:
                full.add(p)
        for tok in set(re.split(r"[^A-Za-z0-9-]+", c)) | set(re.split(r"[^A-Za-z0-9]+", c)):
            tok = tok.strip("-")
            if len(tok) < 3 or tok.isdigit() or tok.lower() in L._STOP or tok.lower() in COMMON:
                continue
            if len(tok) <= 4 and not (re.search(r"\d", tok) or tok.isupper()):
                continue
            toks.add(tok)
        for cand in KD.candidates(name):
            for m in chembl.get(cand, []):
                chem.add(cand)
                if m.get("pref_name"):
                    chem.add(m["pref_name"])
    chem = {c for c in chem if len(c) >= 3 and c.lower() not in COMMON and c.lower() not in L._STOP}
    return full, toks, chem


def global_drug_dictionary(chembl):
    """Every clinical-stage ChEMBL name in the cache (any trial): catches drugs named in a title
    that are not listed as that trial's interventions (background, challenge or prior therapy)."""
    g = {}
    for term, hits in chembl.items():
        for m in hits:
            if str(m.get("max_phase") or "") not in {"1.0", "2.0", "3.0", "4.0"}:
                continue
            if (m.get("pref_name") or "") in GENERIC_SUBSTANCE:
                continue
            for t in {term, m.get("pref_name") or ""}:
                if len(t) >= 6 and t.lower() not in COMMON and t.lower() not in L._STOP \
                        and not TARGET.match(t) and t.upper() not in GENERIC_SUBSTANCE:
                    g[t.lower()] = t
    return g


def scan_masked_title(masked, info, chembl, gdict):
    m150 = str(masked)[:150]  # what the model saw
    full, toks, chem = identity_terms(info, chembl)
    ok = lambda t: not TARGET.match(t) and t.lower() not in COMMON  # noqa: E731
    hits = {"ctgov_name": sorted({t for t in full | toks if ok(t) and _has(t, m150)}),
            "chembl_own": sorted({t for t in chem if ok(t) and _has(t, m150)})}
    hits["chembl_any_trial"] = sorted({v for v in gdict.values() if _has(v, m150)}
                                      - set(hits["ctgov_name"]) - set(hits["chembl_own"]))
    words = set(re.findall(r"[A-Za-z][A-Za-z-]{5,}", m150))
    idwords = {w for t in full | toks | chem for w in re.findall(r"[A-Za-z][A-Za-z-]{5,}", t)}
    fz = set()
    for w in words:
        wl = w.lower()
        if wl in COMMON or wl in L._STOP or wl.rstrip("s") in CLASS_WORDS:
            continue
        for t in idwords:
            tl = t.lower()
            if wl == tl or wl.rstrip("s") == tl.rstrip("s") or wl in tl or tl in wl:
                continue
            if difflib.SequenceMatcher(None, wl, tl).ratio() >= 0.85:
                fz.add(f"{w}~{t}")
    hits["misspelling"] = sorted(fz)
    hits["inn_stem"] = sorted({w for w in words if INN.match(w) and not w.isupper()
                               and w.lower() not in INN_WHITELIST and w.upper() not in GENERIC_SUBSTANCE})
    hits["abbreviation"] = sorted({a for a in ABBR.findall(m150) if a.lower() not in COMMON and not TARGET.match(a)})
    return hits


# Manual review (by the auditing agent) of 100 masked titles drawn with
# samp.sample(n=100, random_state=0), each read against its original title and CT.gov
# interventions. leak: "listed" = a listed intervention's name, alias, abbreviation or
# misspelling is still visible; "unlisted" = another named drug is visible (background,
# challenge). trial_name = study name / protocol number visible. class_only = only drug class,
# mechanism, target or modality visible. over = non-identity words that masking removed.
MANUAL = {
    "NCT04566133": {"leak": "listed", "detail": "'(HCQ)' abbreviation of hydroxychloroquine left after '[INTERVENTION]'"},
    "NCT05552859": {"leak": "listed", "detail": "'Gla-300 and IDeg-100' (insulin glargine 300 U/mL, insulin degludec 100 U/mL)", "over": "Insulin(-Naive)"},
    "NCT04873895": {"leak": "listed", "detail": "'Hydroxychlorquine' (misspelling escapes exact matching)"},
    "NCT05492695": {"leak": "unlisted", "detail": "'OnabotulinumtoxinA' (background therapy, not a listed intervention)"},
    "NCT04771143": {"leak": "unlisted", "detail": "'Cocaine' (challenge agent)"},
    "NCT06056310": {"trial_name": "HyperlynX"}, "NCT04598321": {"trial_name": "BrUOG 390"},
    "NCT04602117": {"trial_name": "ISPY-P1.01"}, "NCT04766996": {"trial_name": "PROUD"},
    "NCT05977127": {"class_only": "mRNA vaccine", "over": "COVID-19"},
    "NCT04386252": {"class_only": "dendritic cell vaccine", "over": "COVID-19"},
    "NCT05251233": {"class_only": "proton pump inhibitors"},
    "NCT05717400": {"class_only": "immunotherapy; direct-acting antiviral"},
    "NCT04911790": {"class_only": "inactivated SARS-CoV-2 vaccine", "over": "COVID-19"},
    "NCT04963413": {"class_only": "CMV RNA-pulsed dendritic cell vaccine"},
    "NCT05281562": {"class_only": "immunonutrition"}, "NCT03550352": {"class_only": "cannabinoids"},
    "NCT05060276": {"class_only": "anti-Trop2", "over": "Antibody; Conjugate"},
    "NCT04584034": {"class_only": "bronchodilators"}, "NCT05477498": {"class_only": "iron"},
    "NCT06065059": {"class_only": "PARP inhibitor"}, "NCT04795466": {"class_only": "anti-inflammatory agents"},
    "NCT05362058": {"over": "Basal Insulin"}, "NCT05156645": {"over": "COVID-19"},
    "NCT06484881": {"over": "Androgenetic Alopecia (disease)"}, "NCT04290871": {"over": "COVID-19"},
    "NCT06178367": {"over": "Molecular"}, "NCT05283954": {"over": "Regimen; COVID-19"},
    "NCT05021484": {"over": "Antibody(-Mediated Rejection) (disease)"}, "NCT04894474": {"over": "COVID-19"},
    "NCT05993143": {"over": "COVID-19"}, "NCT05015972": {"over": "CD19"}, "NCT05065411": {"over": "Combo"},
    "NCT05152849": {"over": "COVID-19"}, "NCT04773067": {"over": "COVID-19"}, "NCT04615949": {"over": "COVID-19"},
    "NCT03125187": {"over": "Administration"}, "NCT05554627": {"over": "Depression (disease)"},
    "NCT06329687": {"over": "Nasal (route)"}, "NCT05292755": {"over": "Artificial Tears"},
}
OVER_DISEASE_TARGET = re.compile(r"COVID-19|CD19|disease", re.I)


def masking_audit(samp, iv, chembl):
    gdict = global_drug_dictionary(chembl)
    rows, traced_ok = [], 0
    for _, r in samp.iterrows():
        info = iv.get(r["nct_id"], {"names": []})
        masked, rep = mask_title_traced(r["brief_title"], info)
        traced_ok += masked == r["title_masked"]
        hits = scan_masked_title(r["title_masked"], info, chembl, gdict)
        rows.append({"nct_id": r["nct_id"], "title": r["brief_title"], "masked": r["title_masked"],
                     "unchanged": r["title_masked"] == r["brief_title"],
                     "code_overmask": [c for c in rep["code_rule"] if OVERMASK_CODE.match(c)],
                     "acronym_overmask": [a for a in rep["acronym_rule"] if OVERMASK_ACR.match(a)],
                     "code_rule": rep["code_rule"], **{f"scan_{k}": v for k, v in hits.items()}})
    a = pd.DataFrame(rows)
    strict_cols = ["scan_ctgov_name", "scan_chembl_own", "scan_misspelling", "scan_abbreviation"]
    broad_cols = strict_cols + ["scan_chembl_any_trial", "scan_inn_stem"]
    a["leak_auto_strict"] = a[strict_cols].map(len).sum(axis=1) > 0
    a["leak_auto"] = a[broad_cols].map(len).sum(axis=1) > 0
    covid_title = a["title"].str.contains("covid", case=False)
    covid_masked = covid_title & ~a["masked"].str.contains("covid", case=False)
    tgt = a["code_overmask"].map(lambda v: any(not c.upper().startswith("COVID") for c in v))
    # manual review
    s100 = samp.sample(n=100, random_state=SEED)["nct_id"].tolist()
    lab = {n: MANUAL.get(n, {}) for n in s100}
    assert set(MANUAL) <= set(s100), "manual labels must belong to the seed-0 sample"
    k_listed = sum(v.get("leak") == "listed" for v in lab.values())
    k_any = sum(v.get("leak") in ("listed", "unlisted") for v in lab.values())
    k_tn = sum("trial_name" in v for v in lab.values())
    k_cls = sum("class_only" in v for v in lab.values())
    k_over = sum("over" in v for v in lab.values())
    k_over_dt = sum(bool(OVER_DISEASE_TARGET.search(v.get("over", ""))) for v in lab.values())
    auto_in_sample = set(a.loc[a["nct_id"].isin(s100) & a["leak_auto"], "nct_id"])
    manual_any = {n for n, v in lab.items() if v.get("leak")}
    by_nct = a.set_index("nct_id")
    examples = [{"nct_id": n, "original": by_nct.loc[n, "title"][:150], "masked": by_nct.loc[n, "masked"][:150],
                 **MANUAL[n]} for n in s100 if MANUAL.get(n, {}).get("leak")]
    auto_examples = [{"nct_id": r["nct_id"], "masked": r["masked"][:150],
                      "hits": {k[5:]: r[k] for k in broad_cols if r[k]}} for _, r in a[a["leak_auto"]].iterrows()]
    out = {
        "traced_masker_reproduces_saved_masks": f"{traced_ok}/{len(a)}",
        "unchanged_titles": int(a["unchanged"].sum()),
        "unchanged_titles_first150": int((a["title"].str[:150] == a["masked"].str[:150]).sum()),
        "automated_scan": {
            "definition": "masked title (first 150 characters, as sent) scanned for the trial's own "
                          "CT.gov intervention names/aliases/short codes, the ChEMBL synonyms and "
                          "preferred names they resolve to (cache), close misspellings "
                          "(difflib ratio >= 0.85), an abbreviation left in parentheses after "
                          "[INTERVENTION], clinical-stage ChEMBL names from any trial, and INN stems; "
                          "target/biomarker/class words excluded",
            "n_flagged_strict_listed_identity": int(a["leak_auto_strict"].sum()),
            "n_flagged_any_named_drug": int(a["leak_auto"].sum()),
            "rate_any": round(float(a["leak_auto"].mean()), 4),
            "rate_any_ci95_wilson": wilson(int(a["leak_auto"].sum()), len(a)),
            "rule_counts": {k[5:]: int((a[k].map(len) > 0).sum()) for k in broad_cols},
            "flagged": auto_examples},
        "manual_review": {
            "sample": "samp.sample(n=100, random_state=0)", "n": 100,
            "listed_intervention_name_visible": k_listed, "listed_rate_ci95_wilson": wilson(k_listed, 100),
            "any_named_drug_visible": k_any, "any_rate_ci95_wilson": wilson(k_any, 100),
            "trial_name_or_protocol_visible": k_tn, "class_or_mechanism_only": k_cls,
            "over_masked_any": k_over, "over_masked_any_ci95": wilson(k_over, 100),
            "over_masked_disease_or_target": k_over_dt, "over_masked_disease_or_target_ci95": wilson(k_over_dt, 100),
            "automated_scan_vs_manual": {"manual_any_named": len(manual_any), "auto_flagged": len(auto_in_sample),
                                         "both": len(manual_any & auto_in_sample),
                                         "missed_by_scan": sorted(manual_any - auto_in_sample),
                                         "auto_only": sorted(auto_in_sample - manual_any)},
            "leak_examples": examples, "labels": {n: v for n, v in lab.items() if v}},
        "over_masking_rules_all600": {
            "titles_containing_covid": int(covid_title.sum()),
            "covid_term_masked": int(covid_masked.sum()),
            "titles_with_target_or_biomarker_masked_by_code_rule": int(tgt.sum()),
            "target_examples": sorted({c for v in a["code_overmask"] for c in v if not c.upper().startswith("COVID")}),
            "titles_with_disease_or_word_masked_by_acronym_rule": int((a["acronym_overmask"].map(len) > 0).sum()),
            "acronym_examples": sorted({x for v in a["acronym_overmask"] for x in v}),
            "titles_overmasked_by_code_or_acronym_rule": int(((a["code_overmask"].map(len) > 0) |
                                                              (a["acronym_overmask"].map(len) > 0)).sum())},
    }
    return out, a[["nct_id", "leak_auto", "leak_auto_strict"]]


# --------------------------------------------------------------------------- refined strata
# Curated after reading every unresolved code and every non-approved ChEMBL match in the cache.
VEHICLE = {"SODIUM CHLORIDE", "WATER"}  # placebo diluents, not investigational products
NOT_A_DEV_COMPOUND = {  # generic substances, salt/conjugate components, mis-hits of short terms
    "ALANINE", "ALANYL GLUTAMINE", "ARGININE", "BUTANE", "CARBOXYMETHYLCELLULOSE CALCIUM",
    "CINNAMON", "COPPER GLUCONATE", "CYTOSINE", "ESTROGEN", "FIBRIN", "GALLIUM", "GLP-1",
    "GLUCAGON-LIKE PEPTIDE 1", "INOSITOL", "L-CITRULLINE", "MALTODEXTRIN", "OMEGA-3 FATTY ACIDS",
    "PHENYLALANINE", "PLATELETS", "PROGRAMMED CELL DEATH 1 LIGAND 1", "SESAME OIL", "SOYSTEROL",
    "TECHNETIUM", "YEAST, DRIED", "DUOCARMAZINE", "VEDOTIN", "LORNOXICAM", "REGRAMOSTIM", None}
# Approved before 2021 outside ChEMBL's max_phase (regional approval, prodrug/salt/conjugate
# parent, or licensed vaccine); every sampled trial starts in 2021 or later.
APPROVED_BEFORE_2021 = {
    "CAMRELIZUMAB", "SINTILIMAB", "TUCIDINOSTAT", "TEGOPRAZAN", "LEVOMETHADONE", "YOHIMBINE",
    "YOHIMBINE HYDROCHLORIDE", "NABIXIMOLS", "RINTATOLIMOD", "ISAVUCONAZOLE", "TENOFOVIR",
    "FLUDARABINE", "SACITUZUMAB", "INCLISIRAN", "THROMBOMODULIN ALFA", "ICOSAPENT", "ANGIOTENSIN",
    "HAEMOPHILUS INFLUENZAE TYPE B CAPSULAR POLYSACCHARIDE MENINGOCOCCAL OUTER MEMBRANE PROTEIN CONJUGATE ANTIGEN"}
# Development-code strings that are CT.gov aliases of approved products, or not drugs at all.
APPROVED_ALIAS_CODES = {
    "NSC-10281", "PI3K Inhibitor BAY 80-6946", "BCD-178", "EG1206A", "HLX11", "HS627", "RO4368451",
    "TQB 2440", "TQB-2440", "TQB2440", "ALT02", "CT-P06", "QL 1701", "QL-1701", "QL1701", "ASP9785",
    "formerly CP-675,206", "SCH 900475", "CNTO1959", "GSK 4182136", "VIR 7831", "B1939 Mesylate",
    "ASTX727", "LY3650150", "BG00002", "BXCL501", "CHX-3311", "U 19920", "WR-28453", "SH T 586",
    "CPX-351", "Liposomal AraC-Daunorubicin CPX-351", "WR-19039", "SHP620", "TAK-620", "VP 16",
    "VP 16-213", "WR-19813", "WR 45312", "MPDL 3280A", "MPDL 328OA", "MPDL328OA", "RO5541267",
    "JNJ-67896062", "ACT-064922", "NSC 763760", "TMT212-NXA", "JTP-78296", "JTP-75303", "NSC 763093",
    "HOE901-U300", "CB 2041", "CB-2041", "GT 41", "GT-41", "WR-19508", "JNJ-78436735 Vaccine"}
NON_DRUG_CODES = {"RT 50 Gy", "RT 54 GY", "RT 60 GY", "K171355",
                  "Neoprobe Gamma Detection System NPB11L(Model1102)"}
NONCODE = re.compile(r"^(COVID[- ]?19|CD\d+|P\d{3}|HPV-?\d+|[A-Z]\d{3}[A-Z]|IL-?\d+|B\d{3})$", re.I)
# Investigational cell/gene/microbial products named without a development code and absent
# from ChEMBL (e.g. "Anti-CEA CAR-T cells", "MASE-T", "SQZ-AAC-HPV", "MET4", "T-Guard").
UNCODED_NOVEL = re.compile(r"CAR-?T\b|CAR T|\bT[- ]cells?\b|cell therapy|dendritic|\bNK cells?\b|"
                           r"oncolytic|gene therapy|microbial ecosystem|\bMET-?4\b|T-Guard|SQZ-", re.I)
# ChEMBL first_approval later than the molecule's first clinical use/authorisation before the
# trial started (molecule-level ChEMBL dates; checked for all 18 'approved after start' trials).
MISDATED = {
    "PHENOBARBITAL": "in clinical use since 1912; ChEMBL 2022 is the first FDA NDA (Sezaby)",
    "HEPARIN": "in clinical use since the 1930s (US approval 1939); ChEMBL gives 2023",
    "VONOPRAZAN": "approved in Japan 2014; ChEMBL gives the 2022 US approval",
    "TORIPALIMAB": "approved in China Dec 2018; ChEMBL gives the 2023 US approval",
    "ELASOMERAN": "mRNA-1273 authorised (US EUA) Dec 2020, before the 2021 start",
}
# Code collision: Allay's ATX101 (bupivacaine implant) resolved to Kythera's ATX-101
# (deoxycholic acid) in ChEMBL.
IGNORE_MATCH = {"NCT05260008": {"DEOXYCHOLIC ACID", "SODIUM DEOXYCHOLATE"}}
DEV_CODE_OVERRIDE = {"NCT05260008": ["ATX-101"]}


def refined_strata(samp, iv, chembl):
    rows = []
    for _, r in samp.iterrows():
        nct, start = r["nct_id"], int(r["start_year"])
        pre, post, undated, dev, misdated = set(), set(), set(), set(), set()
        for n in iv.get(nct, {"names": []})["names"]:
            if re.match(r"\s*placebo", n, re.I):  # placebo arms (e.g. "Placebo to BI 690517")
                continue
            mols = {m["chembl_id"]: m for c in KD.candidates(n) for m in chembl.get(c, [])}
            mols = {k: m for k, m in mols.items() if m.get("pref_name") not in IGNORE_MATCH.get(nct, set())
                    and m.get("pref_name") not in VEHICLE}
            for m in mols.values():
                name = m.get("pref_name")
                approved = str(m.get("max_phase") or "").startswith("4") or bool(m.get("first_approval")) \
                    or name in APPROVED_BEFORE_2021
                if not approved:
                    if name not in NOT_A_DEV_COMPOUND:
                        dev.add(name)
                    continue
                fa = m.get("first_approval")
                if name in APPROVED_BEFORE_2021 and not fa:
                    pre.add(name)
                elif not fa:
                    undated.add(name)
                elif int(fa) < start:
                    pre.add(name)
                elif name in MISDATED:
                    pre.add(name)
                    misdated.add(name)
                else:
                    post.add(name)
            codes = [c for c in L._CODE.findall(n) if not NONCODE.match(c)]
            if codes and not mols and n not in APPROVED_ALIAS_CODES | NON_DRUG_CODES:
                dev.add(n)
            elif not mols and UNCODED_NOVEL.search(n):
                dev.add(n)
        dev |= set(DEV_CODE_OVERRIDE.get(nct, []))
        approved_any = pre | post | undated
        if post and not pre:
            s = "approved_after_start_corrected"
        elif pre and (dev or post):
            s = "mixed"
        elif pre:
            s = "established_all_pre"
        elif dev and not approved_any:
            s = "novel_dev_code_only"
        else:
            s = "other"
        rows.append({"nct_id": nct, "refined": s, "pre": sorted(pre), "post": sorted(post),
                     "dev": sorted(dev), "misdated": sorted(misdated)})
    df = pd.DataFrame(rows)
    # same-year ambiguity (first approval in the start year) for the after-start stratum
    sy = []
    for _, r in df.iterrows():
        if r["refined"] != "approved_after_start_corrected":
            sy.append(False)
            continue
        start = int(samp.loc[samp["nct_id"] == r["nct_id"], "start_year"].iloc[0])
        years = {int(m["first_approval"]) for n in iv[r["nct_id"]]["names"] for c in KD.candidates(n)
                 for m in chembl.get(c, []) if m.get("pref_name") in r["post"] and m.get("first_approval")}
        sy.append(bool(years) and max(years) == start)
    df["same_year_only"] = sy
    return df


# --------------------------------------------------------------------------- contrasts
def wide_for(preds, model):
    return preds[preds["model"] == model].pivot(
        index=["nct_id", "label", "completion_date"], columns="condition", values="p").reset_index()


def pick(res):
    if "contrasts" not in res:
        return {"n": res.get("n"), "n_pos": res.get("n_pos"), "note": "too few trials or one class"}
    out = {"n": res["n"], "n_pos": res["n_pos"],
           "conditions": {c: {k: res["conditions"][c][k] for k in ("auprc", "auroc")} for c in ("D", "D+Tm", "D+T")}}
    for k in KEY + ["D+ID - D", "D+T+ID+S - D+T"]:
        c = res["contrasts"][k]
        out[k] = {"auprc": c["auprc"], "auprc_ci": c["auprc_ci"], "auprc_p_gt_0": c["auprc_p_gt_0"],
                  "auroc": c["auroc"], "auroc_ci": c["auroc_ci"]}
    return out


def main():
    os.makedirs("outputs", exist_ok=True)
    samp = L.load_sample(L.N)
    iv = json.load(open(L.INTERVENTIONS_CACHE))
    chembl = json.load(open(KD.CACHE))
    preds = pd.read_csv(L.PRED_PATH)
    models = [m for m in L.MODELS if m in set(preds["model"])]
    y = samp["labels"].values.astype(int)
    covid = (samp["brief_title"].str.contains(COVID_RE, case=False) |
             samp["official_title"].fillna("").str.contains(COVID_RE, case=False))
    flags = pd.DataFrame({"nct_id": samp["nct_id"], "covid": covid.values,
                          "withdrawn": (samp["overall_status"] == "WITHDRAWN").values})
    out = {"note": "Offline re-analysis of saved predictions; no language model was queried.",
           "provenance": prompt_provenance(samp)}

    # ---- claims
    cd = samp["completion_date"]
    claims = {
        "L1_enrollment_type": samp["enrollment_type"].fillna("NA").value_counts().to_dict(),
        "L1_withdrawn_zero_enrolment": int(((samp["overall_status"] == "WITHDRAWN") & (samp["enrollment"] == 0)).sum()),
        "L1_enroll_bucket_by_status": pd.crosstab(samp["enrollment"].map(L._bucket), samp["overall_status"]).to_dict(),
        "L2_tabular_design_only": tabular_design_model(samp),
        "L2_llm_design_only": {m: {k: v for k, v in json.load(open(L.OUT_PATH))["models"][m]["all"]["conditions"]["D"].items()
                                   if k in ("auprc", "auroc")} for m in models},
        "L3_label_by_status": {s: {"n": int((samp["overall_status"] == s).sum()),
                                   "label_1": int(((samp["overall_status"] == s) & (y == 1)).sum())}
                               for s in samp["overall_status"].unique()},
        "L3_non_completed": int((samp["overall_status"] != "COMPLETED").sum()),
        "L5_completion_date_range": [str(cd.min().date()), str(cd.max().date())],
        "L5_completion_date_type": samp["completion_date_type"].value_counts().to_dict(),
        "L5_trials_completed_after_documented_cutoff": {
            m: {"cutoff": f"{c[0][0]}-{c[0][1]:02d}", "source": c[1],
                "n_after": int((cd > pd.Timestamp(year=c[0][0], month=c[0][1], day=1) + pd.offsets.MonthEnd(0)).sum())}
            for m, c in DOC_CUTOFFS.items()},
        "n_covid_trials": int(covid.sum()), "n_covid_successes": int((covid & (y == 1)).sum()),
    }
    out["claims"] = claims

    # ---- masking audit
    audit, leak = masking_audit(samp, iv, chembl)
    out["masking_audit"] = audit
    flags = flags.merge(leak, on="nct_id")

    # ---- contrasts under exclusions (reusing L.summarize; Holm across the 10 primary tests)
    subsets = {"a_as_published": lambda f: pd.Series(True, index=f.index),
               "b_excl_covid": lambda f: ~f["covid"],
               "c_excl_masked_title_leak": lambda f: ~f["leak_auto"],
               "c2_excl_leak_strict_listed_only": lambda f: ~f["leak_auto_strict"],
               "d_excl_withdrawn": lambda f: ~f["withdrawn"],
               "e_excl_all_three": lambda f: ~(f["covid"] | f["leak_auto"] | f["withdrawn"])}
    out["contrasts"] = {}
    for sname, fn in subsets.items():
        keep = set(flags.loc[fn(flags), "nct_id"])
        res_all, block = {}, {}
        for m in models:
            w = wide_for(preds, m)
            res = L.summarize(w[w["nct_id"].isin(keep)])
            res_all[m] = {"all": res}
            block[m] = pick(res)
        prim = L.primary_tests(res_all)
        for r in prim:
            block[r["model"]].setdefault("holm", {})[r["contrast"]] = {"p_two_sided": r["p_two_sided"], "p_holm": r["p_holm"]}
        out["contrasts"][sname] = {"n_trials": len(keep), "models": block}
        print(f"[{sname}] n={len(keep)}", flush=True)
        for m in models:
            b = block[m]
            if "D+T - D+Tm" in b:
                print(f"   {m:28s} D+T-D+Tm {b['D+T - D+Tm']['auprc']:+.4f} {b['D+T - D+Tm']['auprc_ci']} "
                      f"(AUROC {b['D+T - D+Tm']['auroc']:+.4f} {b['D+T - D+Tm']['auroc_ci']}) Holm "
                      f"{b.get('holm', {}).get('D+T - D+Tm', {}).get('p_holm')} | D+Tm-D {b['D+Tm - D']['auprc']:+.4f} "
                      f"{b['D+Tm - D']['auprc_ci']} (AUROC {b['D+Tm - D']['auroc']:+.4f})", flush=True)
    out["claims"]["L7_id_sponsor_increment"] = {
        m: out["contrasts"]["a_as_published"]["models"][m]["D+T+ID+S - D+T"] for m in models}

    # ---- ChEMBL strata
    orig = KD.stratify(samp, iv, chembl)
    ref = refined_strata(samp, iv, chembl).merge(orig.rename(columns={"stratum": "original"}), on="nct_id")
    ref = ref.merge(flags[["nct_id", "covid"]], on="nct_id").merge(
        samp[["nct_id", "labels", "start_year", "brief_title"]], on="nct_id")
    ref.loc[(ref["refined"] == "established_all_pre") & ref["covid"], "refined"] = "established_all_pre_covid"
    b316 = ref[ref["original"] == "approved_before_start"]
    after18 = ref[ref["original"] == "approved_after_start"]
    after_list = []
    for _, r in after18.iterrows():
        mols = sorted({(m["pref_name"], m["first_approval"]) for n in iv[r["nct_id"]]["names"]
                       for c in KD.candidates(n) for m in chembl.get(c, []) if m.get("first_approval")})
        after_list.append({"nct_id": r["nct_id"], "start_year": int(r["start_year"]), "label": int(r["labels"]),
                           "title": r["brief_title"][:90], "approved_molecules": [f"{a} ({b})" for a, b in mols],
                           "misdated": {x: MISDATED[x] for x in r["misdated"]}, "covid": bool(r["covid"]),
                           "same_year_only": bool(r["same_year_only"]), "refined": r["refined"]})
    order = ["established_all_pre", "established_all_pre_covid", "mixed", "novel_dev_code_only",
             "approved_after_start_corrected", "other"]
    cr = {"definitions": {
        "established_all_pre": "every matched approved molecule first approved before the start year "
                               "(misdating corrected), no never-approved/development-code compound, not COVID-19",
        "established_all_pre_covid": "as above, COVID-19 trials (reported separately)",
        "mixed": "an approved-before-start drug plus a never-approved/development-code compound or a drug approved after start",
        "novel_dev_code_only": "a never-approved compound (ChEMBL max_phase < 4 or an unresolved development code) and no approved drug",
        "approved_after_start_corrected": "approved only in/after the start year, misdated approvals removed",
        "other": "remaining trials"},
        "counts": ref["refined"].value_counts().reindex(order).fillna(0).astype(int).to_dict(),
        "positives": ref.groupby("refined")["labels"].sum().reindex(order).fillna(0).astype(int).to_dict(),
        "crosstab_original_vs_refined": pd.crosstab(ref["original"], ref["refined"]).to_dict(),
        "L6_of_316_approved_before_start": {
            "n": int(len(b316)),
            "with_dev_or_never_approved_compound_curated": int((b316["dev"].map(len) > 0).sum()),
            "with_drug_approved_after_start": int((b316["post"].map(len) > 0).sum()),
            "covid": int(b316["covid"].sum()),
            "refined_breakdown": b316["refined"].value_counts().to_dict()},
        "L6_approved_after_start_18": after_list,
        "first_approval_semantics": "ChEMBL MOLECULE_DICTIONARY.FIRST_APPROVAL = 'Earliest known approval year "
                                    "for the drug'; MAX_PHASE is 'across all indications' (ChEMBL schema "
                                    "documentation); the cache stores no indication, so dating is molecule-level.",
        "models": {}}
    strat_sets = {s: ref.loc[ref["refined"] == s, "nct_id"] for s in order}
    strat_sets["approved_after_start_strict_later_year"] = ref.loc[
        (ref["refined"] == "approved_after_start_corrected") & ~ref["same_year_only"], "nct_id"]
    strat_sets["original_approved_before_start"] = ref.loc[ref["original"] == "approved_before_start", "nct_id"]
    strat_sets["original_never_approved_code"] = ref.loc[ref["original"] == "never_approved_code", "nct_id"]
    for m in models:
        w = wide_for(preds, m)
        cr["models"][m] = {}
        for s, ids in strat_sets.items():
            ws = w[w["nct_id"].isin(set(ids))].dropna(subset=L.CONDITIONS)
            cr["models"][m][s] = {k: KD.contrast(ws, *k.split(" - ")) for k in KEY}
        line = " | ".join(f"{s}: {cr['models'][m][s]['D+T - D+Tm'].get('auprc')} "
                          f"{cr['models'][m][s]['D+T - D+Tm'].get('auprc_ci')} n={cr['models'][m][s]['D+T - D+Tm']['n']}"
                          for s in ["established_all_pre", "mixed", "novel_dev_code_only", "approved_after_start_corrected"])
        print(f"   {m:28s} {line}", flush=True)
    out["chembl_refined"] = cr
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
