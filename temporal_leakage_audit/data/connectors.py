"""Real-data connectors: build the drug-program benchmark from public APIs.

Each function returns data in the exact schema the rest of the pipeline expects,
so ``build_real_dataset`` can write ``programs.csv`` and ``evidence.csv`` into
``cfg['data_dir']`` with the columns documented below.

CONTRACT
--------
programs.csv (one row per molecule-target-indication-phase program):
    program_id        : unique str
    target_id         : gene/target id (Ensembl gene id)
    indication        : disease EFO id
    therapeutic_area  : coarse area (str)
    sponsor_tier      : 0/1/2 proxy for sponsor scale-experience (confounder)
    modality          : small_molecule / biologic / other
    info_time         : YEAR the program entered ``phase_from`` (the freeze point)
    phase_from        : 1 / 2 / 3  (phase at info_time)
    label             : 1 if it advanced to phase_from+1 (resolved AFTER info_time)

evidence.csv (long/event log, many rows per program):
    program_id        : matches programs.program_id
    evidence_type     : genetic | somatic | known_drug | pathway |
                        animal_model | expression | literature
    score             : float strength/contribution in [0, 1]
    evidence_date     : YEAR the evidence became available (see the WARNING)

EVIDENCE DATING (the load-bearing modelling choice)
---------------------------------------------------
``evidence_date`` decides what counts as knowable at decision time. Open Targets *literature / text-mining*
evidence frequently has NO reliable timestamp and reflects post-approval
publications -- that is the leakage. This module reconstructs a conservative
``evidence_date`` from the underlying PubMed publication years (the earliest year
across a datum's PMIDs) and *drops* literature it cannot date. The choice is
configurable via ``cfg['real']['literature_policy']`` and every kept/dropped
count is logged so the LAP metric measures exactly how much that choice removes.

DATA SOURCES (all public; no authentication required)
    Open Targets GraphQL  https://api.platform.opentargets.org/api/v4/graphql
        disease.drugAndClinicalCandidates  -> drug x disease x per-trial reports
                                              (this replaced the removed
                                               `knownDrugs` field in rel 26.03)
        disease.evidences                  -> evidence by datatype
        drug.linkedTargets                 -> drug -> Ensembl target join
    ClinicalTrials.gov v2 REST             -> leadSponsor.class -> sponsor_tier
    NCBI E-utilities esummary              -> PubMed publication years

MODELLING DECISIONS (reported in the paper's Methods)
    1. PROGRAM = (drug, disease, phase_from). For each drug-disease pair we read
       the harmonised clinicalStage of every ClinicalTrials.gov-derived report and
       emit one row per observed clinical phase in {1,2,3} that has a datable trial
       start. ``info_time`` is the earliest trial-start YEAR at that phase.
       ``label`` = 1 iff a date-stamped trial at a HIGHER clinical phase
       (phase_from+1 or beyond) started in a year STRICTLY AFTER info_time -- i.e.
       advancement observed after the freeze point. Advancement at/before the freeze
       is not counted, and undated drug-/disease-level max-stage summaries are NOT
       used for the label (they cannot be shown to postdate the freeze and would
       leak cross-indication outcome). Alternative (one row per program at its entry
       phase) is noted but not used.
    2. TARGET is recovered from the drug via ``drug.linkedTargets`` (falling back
       to mechanismsOfAction). Programs whose drug has no Ensembl target are
       dropped by default so target-disjoint splitting stays well defined.
    3. LITERATURE dating: PubMed-min-year, drop-if-undatable (see EVIDENCE DATING).

CAVEATS
    * The integer mapping of the harmonised clinicalStage vocabulary to phase
      numbers is inferred, not officially published by Open Targets. Validate
      against known drugs before trusting numeric thresholds.
    * Open Targets ships bi-monthly; ``check_version`` warns if the live
      dataVersion has drifted from ``expected_data_version`` -- re-verify field
      names against the live schema if you then see GraphQL validation errors.
    * This module performs many single-entity queries. For large studies the
      Open Targets docs steer you to the bulk Parquet/BigQuery downloads instead.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..config import get_config

log = logging.getLogger("connectors")

REQUIRED_PROGRAM_COLS = [
    "program_id", "target_id", "indication", "therapeutic_area",
    "sponsor_tier", "modality", "info_time", "phase_from", "label",
]
REQUIRED_EVIDENCE_COLS = ["program_id", "evidence_type", "score", "evidence_date"]

_UA = "temporal-leakage-audit/1.0 (academic research software)"

# ---------------------------------------------------------------------------
# Vocabularies / mappings
# ---------------------------------------------------------------------------

# Open Targets datatypeId -> this project's evidence_type vocabulary.
DATATYPE_TO_EVIDENCE = {
    "genetic_association": "genetic",
    "somatic_mutation": "somatic",
    "known_drug": "known_drug",
    "affected_pathway": "pathway",
    "animal_model": "animal_model",
    "rna_expression": "expression",
    "literature": "literature",
}

# Open Targets clinicalStage enum -> approximate numeric phase (inferred).
# NOTE: values are the live 26.06 enum (UPPER_SNAKE, e.g. PHASE_3, APPROVAL),
# verified by schema introspection (not the title-case "Phase III" used in older
# releases). Re-verify if a future release renames the enum.
STAGE_TO_NUM = {
    "APPROVAL": 4.0, "PHASE_4": 4.0, "WITHDRAWAL": 4.0,
    "PREAPPROVAL": 3.5,
    "PHASE_3": 3.0, "PHASE_2_3": 2.5,
    "PHASE_2": 2.0, "PHASE_1_2": 1.5,
    "PHASE_1": 1.0, "EARLY_PHASE_1": 0.5,
    "IND": 0.0, "PRECLINICAL": 0.0, "UNKNOWN": None,
}

# Which integer clinical phase a *trial at this stage sits at*. Used both for
# phase_from (a trial at bin p) and for advancement (a later trial at bin >= p+1).
STAGE_BIN = {
    "EARLY_PHASE_1": 1, "PHASE_1": 1, "PHASE_1_2": 1,
    "PHASE_2": 2, "PHASE_2_3": 2,
    "PHASE_3": 3,
    "PHASE_4": 4, "APPROVAL": 4, "PREAPPROVAL": 4, "WITHDRAWAL": 4,
    "IND": 0, "PRECLINICAL": 0, "UNKNOWN": 0,
}

# ClinicalTrials.gov leadSponsor.class -> sponsor_tier (0 small / 1 gov-network / 2 industry).
SPONSOR_CLASS = {
    "INDUSTRY": 2,
    "NIH": 1, "FED": 1, "OTHER_GOV": 1, "NETWORK": 1,
    "OTHER": 0, "INDIV": 0, "UNKNOWN": 0, "AMBIG": 0,
}

# Open Targets drugType -> modality bucket (keys are the live 26.06 drugType values).
DRUGTYPE_TO_MODALITY = {
    "small molecule": "small_molecule",
    "antibody": "biologic", "antibody drug conjugate": "biologic",
    "protein": "biologic", "enzyme": "biologic", "vaccine component": "biologic",
    "oligosaccharide": "biologic",
    "oligonucleotide": "other", "cell": "other", "gene": "other",
    "unknown": "other",
}

NCT_RE = re.compile(r"(NCT\d{8})")
_YEAR_RE = re.compile(r"(\d{4})")

# Defaults so the connector works even if cfg lacks a "real" block (see config.py).
_REAL_DEFAULTS: Dict = {
    "ot_graphql_url": "https://api.platform.opentargets.org/api/v4/graphql",
    "ctgov_v2_url": "https://clinicaltrials.gov/api/v2/studies",
    "pubmed_esummary_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
    "ncbi_api_key": None,
    "ncbi_tool": "temporal-leakage-audit",
    "ncbi_email": None,
    "disease_ids": ["MONDO_0008383"],
    "max_drug_rows": None,          # cap drug candidates per disease (None = all)
    "expected_data_version": {"year": "26", "month": "06"},
    "check_version": True,
    "literature_policy": "pubmed_year",
    "use_pmid_year_fallback": True,
    "max_literature_per_pair": None,  # cap literature evidence per (disease,target); None = all
    "drop_unmapped_targets": True,
    "evidence_size": 3000,
    "pubmed_batch": 200,
    "request_timeout": 60,
    "max_retries": 5,
    "backoff_base": 0.5,
    "sleep_between": 0.1,
    "cache_dir": "data/api_cache",
}

# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------

QUERY_META = "{ meta { dataVersion { year month iteration } } }"

# QUERY A -- clinical drug candidates for a disease (replaces removed `knownDrugs`).
QUERY_DRUGS_BY_DISEASE = """
query kd($efo: String!) {
  disease(efoId: $efo) {
    id
    name
    therapeuticAreas { id name }
    drugAndClinicalCandidates {
      count
      rows {
        maxClinicalStage
        drug { id name drugType maximumClinicalStage }
        clinicalReports {
          id
          source
          type
          url
          clinicalStage
          trialPhase
          phaseFromSource
          trialOverallStatus
          trialStartDate
        }
      }
    }
  }
}
"""

# Drug -> target join. Two variants; we try the first and fall back to the second.
QUERY_DRUG_LINKED_TARGETS = """
query dt($id: String!) {
  drug(chemblId: $id) { id linkedTargets { rows { id approvedSymbol } } }
}
"""
QUERY_DRUG_MOA_TARGETS = """
query dt($id: String!) {
  drug(chemblId: $id) { id mechanismsOfAction { rows { targets { id approvedSymbol } } } }
}
"""

# QUERY B -- evidence for a (disease, target) pair. `size` is capped at 3000; page
# with the opaque `cursor`. We request date-fallback fields and retry without them
# if the live schema rejects any (they are less stable than the core fields).
_EVIDENCE_CORE_FIELDS = """
        datasourceId
        datatypeId
        score
        publicationYear
        literature
"""
_EVIDENCE_DATE_FIELDS = """
        evidenceDate
        studyStartDate
        releaseDate
"""


def _evidences_query(include_dates: bool) -> str:
    fields = _EVIDENCE_CORE_FIELDS + (_EVIDENCE_DATE_FIELDS if include_dates else "")
    return (
        "query ev($efo: String!, $ens: [String!]!, $size: Int!, $cursor: String) {\n"
        "  disease(efoId: $efo) {\n"
        "    evidences(ensemblIds: $ens, size: $size, cursor: $cursor) {\n"
        "      count\n"
        "      cursor\n"
        "      rows {\n" + fields + "      }\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


class GraphQLError(RuntimeError):
    """Open Targets returned a non-empty GraphQL `errors` array."""


# ---------------------------------------------------------------------------
# Config / logging helpers
# ---------------------------------------------------------------------------

def _real_cfg(cfg: Optional[Dict]) -> Dict:
    out = dict(_REAL_DEFAULTS)
    out.update((cfg or {}).get("real") or {})
    return out


def _disease_ids(rc: Dict) -> List[str]:
    """Configured disease universe; ``efo_ids`` is accepted as a legacy alias."""
    return list(rc.get("efo_ids") or rc.get("disease_ids") or [])


def _configure_logging(cfg: Optional[Dict]) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# HTTP + cache (stdlib only -- no new dependencies)
# ---------------------------------------------------------------------------

def _cache_path(rc: Dict, key: str) -> Optional[str]:
    cache_dir = rc.get("cache_dir")
    if not cache_dir:
        return None
    return os.path.join(cache_dir, key + ".json")


def _cache_get(rc: Dict, key: str):
    path = _cache_path(rc, key)
    if path and os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None
    return None


def _cache_put(rc: Dict, key: str, obj) -> None:
    path = _cache_path(rc, key)
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def _http_json(
    url: str,
    cfg: Optional[Dict],
    *,
    method: str = "GET",
    body: Optional[Dict] = None,
    use_cache: bool = True,
    cache_ok=None,
    cache_url: Optional[str] = None,
):
    """GET/POST a URL and parse JSON, with retries, backoff and an on-disk cache.

    ``cache_ok`` is an optional predicate ``obj -> bool``; when given, only responses
    it accepts are written to the cache (used to avoid persisting error bodies).
    ``cache_url`` (default: ``url``) is the URL used to form the cache key, so that
    identification parameters (API key, tool, email) do not change the key.
    """
    rc = _real_cfg(cfg)
    key = None
    if use_cache and rc.get("cache_dir"):
        payload = json.dumps(body, sort_keys=True) if body is not None else ""
        key_url = cache_url or url
        key = hashlib.sha1((method + "|" + key_url + "|" + payload).encode("utf-8")).hexdigest()
        cached = _cache_get(rc, key)
        if cached is not None:
            return cached

    data = None
    headers = {"Accept": "application/json", "User-Agent": _UA}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    last_err: Optional[Exception] = None
    for attempt in range(int(rc["max_retries"])):
        try:
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=rc["request_timeout"]) as resp:
                raw = resp.read()
            try:
                obj = json.loads(raw.decode("utf-8"))
            except ValueError as e:  # includes json.JSONDecodeError -- retry a truncated/garbled body
                last_err = e
                time.sleep(float(rc["backoff_base"]) * (2 ** attempt))
                continue
            if key is not None and (cache_ok is None or cache_ok(obj)):
                _cache_put(rc, key, obj)
            if rc.get("sleep_between"):
                time.sleep(float(rc["sleep_between"]))
            return obj
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(float(rc["backoff_base"]) * (2 ** attempt))
                continue
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise RuntimeError(f"HTTP {e.code} for {url}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            time.sleep(float(rc["backoff_base"]) * (2 ** attempt))
            continue
    raise RuntimeError(f"request failed after {rc['max_retries']} retries: {url}: {last_err}")


def _graphql(query: str, variables: Dict, cfg: Optional[Dict], *, use_cache: bool = True) -> Dict:
    rc = _real_cfg(cfg)
    obj = _http_json(
        rc["ot_graphql_url"], cfg, method="POST",
        body={"query": query, "variables": variables}, use_cache=use_cache,
        # Never cache an error body -- otherwise a transient server-side GraphQL
        # error (HTTP 200 with a non-empty `errors` array) would stick permanently.
        cache_ok=lambda o: not (isinstance(o, dict) and o.get("errors")),
    )
    if obj.get("errors"):
        msgs = "; ".join(str(e.get("message", e)) for e in obj["errors"])
        raise GraphQLError(msgs)
    return obj.get("data") or {}


# ---------------------------------------------------------------------------
# Small pure parsers
# ---------------------------------------------------------------------------

def _extract_nct(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = NCT_RE.search(url)
    return m.group(1) if m else None


def _year_from_date(value) -> Optional[int]:
    """First 4-digit run in a date-ish value ('2020-04-06', '2020', 2020, ...)."""
    if value is None:
        return None
    m = _YEAR_RE.search(str(value))
    return int(m.group(1)) if m else None


_MIN_EVIDENCE_YEAR = 1970


def _sane_year(y: Optional[int]) -> Optional[int]:
    """Reject impossible years. The '(first 4-digit run)' parser can latch onto a
    stray number in a malformed date/id string (e.g. yielding 2119 or 1937), and a
    bogus future year is 'post-decision' for every program -- silently corrupting the
    as-of-time censoring the leakage metric depends on. Drop anything outside
    [1970, current_year+1]."""
    if y is None:
        return None
    hi = time.gmtime().tm_year + 1
    return y if _MIN_EVIDENCE_YEAR <= y <= hi else None


def _sponsor_tier(cls: Optional[str]) -> int:
    return SPONSOR_CLASS.get((cls or "").strip().upper(), 0)


def _modality(drug_type: Optional[str]) -> str:
    return DRUGTYPE_TO_MODALITY.get((drug_type or "").strip().lower(), "other")


def _primary_area(disease: Dict) -> str:
    tas = disease.get("therapeuticAreas") or []
    name = (tas[0].get("name") if tas and tas[0] else None) or "unknown"
    return re.sub(r"\s+", "_", str(name).strip().lower()) or "unknown"


def _parse_ctgov_study(study: Optional[Dict]) -> Dict:
    """Parse a single ClinicalTrials.gov v2 study (top-level protocolSection)."""
    ps = (study or {}).get("protocolSection") or {}
    status_mod = ps.get("statusModule") or {}
    design = ps.get("designModule") or {}
    sponsor_mod = ps.get("sponsorCollaboratorsModule") or {}
    lead = sponsor_mod.get("leadSponsor") or {}
    start = (status_mod.get("startDateStruct") or {}).get("date")
    return {
        "start_year": _year_from_date(start),
        "status": status_mod.get("overallStatus"),
        "phases": design.get("phases") or [],
        "sponsor_class": lead.get("class"),
        "sponsor_name": lead.get("name"),
    }


def _collect_aact_trials(reports: Sequence[Dict]) -> List[Dict]:
    """ClinicalTrials.gov-derived reports -> normalised per-trial records."""
    out = []
    for r in reports or []:
        url = r.get("url") or ""
        nct = _extract_nct(url)
        is_ct = (r.get("source") == "AACT") or (r.get("type") == "CLINICAL_TRIAL") or bool(nct)
        if not is_ct:
            continue
        stage = r.get("clinicalStage") or ""
        out.append({
            "nct": nct,
            "stage": stage,
            "bin": STAGE_BIN.get(stage, 0),
            "start_year": _year_from_date(r.get("trialStartDate")),
            "status": r.get("trialOverallStatus"),
        })
    return out


def _phase_timeline(trials: Sequence[Dict]) -> Tuple[Dict[int, int], List[Tuple[int, int]]]:
    """Return (entry_year, dated_trials) from date-stamped clinical trials.

    entry_year   : {phase in {1,2,3} -> earliest ClinicalTrials.gov start-year of a
                   trial AT that phase}. Each is a candidate freeze point (info_time).
    dated_trials : [(bin, start_year), ...] for every date-stamped trial at a clinical
                   phase in {1,2,3,4}; used to test for post-freeze advancement.

    Only trials with a real start year participate. Undated per-drug / per-disease
    max-stage summaries are deliberately NOT folded in: they carry no date, so they
    cannot be shown to resolve AFTER info_time and would leak the outcome (including
    cross-indication advancement) into the label.
    """
    entry_year: Dict[int, int] = {}
    dated_trials: List[Tuple[int, int]] = []
    for t in trials:
        b, y = t.get("bin"), t.get("start_year")
        if b in (1, 2, 3, 4) and y is not None:
            dated_trials.append((int(b), int(y)))
            if b in (1, 2, 3) and (b not in entry_year or y < entry_year[b]):
                entry_year[b] = int(y)
    return entry_year, dated_trials


# ---------------------------------------------------------------------------
# Live fetchers (each cached + degrade gracefully)
# ---------------------------------------------------------------------------

def _check_data_version(cfg: Optional[Dict]) -> Dict:
    rc = _real_cfg(cfg)
    if not rc.get("check_version", True):
        return {}
    try:
        data = _graphql(QUERY_META, {}, cfg, use_cache=False)
    except Exception as e:  # noqa: BLE001 -- a version check must never be fatal
        log.warning("could not fetch Open Targets data version: %s", e)
        return {}
    dv = (data.get("meta") or {}).get("dataVersion") or {}
    exp = rc.get("expected_data_version") or {}
    if exp and (str(dv.get("year")) != str(exp.get("year"))
                or str(dv.get("month")) != str(exp.get("month"))):
        log.warning(
            "Open Targets data version %s.%s differs from expected %s.%s; "
            "re-verify GraphQL field names against the live schema.",
            dv.get("year"), dv.get("month"), exp.get("year"), exp.get("month"),
        )
    else:
        log.info("Open Targets data version %s.%s", dv.get("year"), dv.get("month"))
    return dv


QUERY_SEARCH_DISEASE = (
    'query sd($q: String!) { search(queryString: $q, entityNames: ["disease"]) '
    '{ hits { id name entity } } }'
)


def _resolve_disease_id(id_or_name: str, cfg: Optional[Dict]) -> Optional[Tuple[str, str]]:
    """Return (current_disease_id, name) for an OT disease id OR a disease name.

    Ontology ids drift between releases (Open Targets migrated many diseases from
    EFO_* to MONDO_* in the 26.x line, so hardcoded EFO ids now resolve to null).
    We try the value as an id first, then fall back to a name search so a stale id
    or a plain name both resolve to the release's current canonical id.
    """
    try:
        d = _graphql("query($e:String!){disease(efoId:$e){id name}}",
                     {"e": id_or_name}, cfg).get("disease")
        if d and d.get("id"):
            return d["id"], d.get("name") or id_or_name
    except (GraphQLError, RuntimeError) as e:
        log.debug("disease id probe failed for %r: %s", id_or_name, e)
    try:
        hits = ((_graphql(QUERY_SEARCH_DISEASE, {"q": id_or_name}, cfg).get("search") or {})
                .get("hits")) or []
    except (GraphQLError, RuntimeError) as e:
        log.debug("disease search failed for %r: %s", id_or_name, e)
        hits = []
    for h in hits:
        if h.get("entity") == "disease" and h.get("id"):
            return h["id"], h.get("name") or id_or_name
    return None


def _fetch_disease_candidates(efo: str, cfg: Optional[Dict]) -> Dict:
    return _graphql(QUERY_DRUGS_BY_DISEASE, {"efo": efo}, cfg)


def _flatten_targets(data: Dict) -> List[Tuple[str, str]]:
    drug = (data or {}).get("drug") or {}
    out: List[Tuple[str, str]] = []
    lt = drug.get("linkedTargets") or {}
    for r in lt.get("rows") or []:
        if r and r.get("id"):
            out.append((r["id"], r.get("approvedSymbol") or ""))
    moa = drug.get("mechanismsOfAction") or {}
    for r in moa.get("rows") or []:
        for t in (r or {}).get("targets") or []:
            if t and t.get("id"):
                out.append((t["id"], t.get("approvedSymbol") or ""))
    seen, uniq = set(), []
    for ens, sym in out:
        if ens in seen:
            continue
        seen.add(ens)
        uniq.append((ens, sym))
    return uniq


def _fetch_primary_target(chembl_id: str, cfg: Optional[Dict]) -> Optional[Tuple[str, str]]:
    # The Drug type has NO `linkedTargets` field in 26.06 (verified live); the only
    # drug->target link is `mechanismsOfAction`. Many small molecules have a null MoA
    # and legitimately resolve to no target (then the program is dropped by default).
    try:
        data = _graphql(QUERY_DRUG_MOA_TARGETS, {"id": chembl_id}, cfg)
    except (GraphQLError, RuntimeError) as e:  # incl. transport failures -> drop the drug
        log.debug("target query failed for %s: %s", chembl_id, e)
        return None
    targets = _flatten_targets(data)
    return targets[0] if targets else None


def _fetch_ctgov(nct: str, cfg: Optional[Dict]) -> Optional[Dict]:
    rc = _real_cfg(cfg)
    url = rc["ctgov_v2_url"].rstrip("/") + "/" + urllib.parse.quote(nct) + "?format=json"
    try:
        study = _http_json(url, cfg, method="GET")
    except Exception as e:  # noqa: BLE001 -- a missing study must not abort the run
        log.warning("ClinicalTrials.gov fetch failed for %s: %s", nct, e)
        return None
    return _parse_ctgov_study(study)


def _program_sponsor_tier(trials: Sequence[Dict], phase: int, info_time: int,
                          cfg: Optional[Dict]) -> int:
    """Max sponsor tier across the program's phase-`phase` trials KNOWN AS-OF info_time.

    Restricted to trials with ``start_year <= info_time`` so this (always-available,
    confounder) feature never encodes post-freeze sponsor information -- otherwise a
    later high-phase/approval trial's sponsor could leak into the deployable model.
    """
    def known(t):
        return (t.get("nct") and t.get("start_year") is not None
                and t["start_year"] <= info_time)
    ncts = [t["nct"] for t in trials if t.get("bin") == phase and known(t)]
    if not ncts:
        ncts = [t["nct"] for t in trials if known(t)]
    tiers = []
    for nct in dict.fromkeys(ncts):  # dedupe, preserve order
        info = _fetch_ctgov(nct, cfg)
        if info and info.get("sponsor_class"):
            tiers.append(_sponsor_tier(info["sponsor_class"]))
    return max(tiers) if tiers else 0


def _fetch_evidences(efo: str, ensembl_ids: Sequence[str], cfg: Optional[Dict]) -> List[Dict]:
    """All evidence rows for a (disease, targets) pair, paging past the 3000 cap."""
    rc = _real_cfg(cfg)
    size = min(int(rc["evidence_size"]), 3000)
    include_dates = True
    rows: List[Dict] = []
    cursor: Optional[str] = None
    pages = 0
    while True:
        pages += 1
        if pages > 10_000:
            log.warning("evidence pagination guard tripped for %s / %s", efo, ensembl_ids)
            break
        variables = {"efo": efo, "ens": list(ensembl_ids), "size": size, "cursor": cursor}
        try:
            data = _graphql(_evidences_query(include_dates), variables, cfg)
        except GraphQLError as e:
            if include_dates:
                log.warning("evidences query rejected date fields (%s); retrying core fields", e)
                include_dates = False
                continue
            log.warning("evidences query failed for %s / %s: %s", efo, ensembl_ids, e)
            break
        except RuntimeError as e:  # transport failure -> drop this pair, keep the run alive
            log.warning("evidences transport error for %s / %s: %s", efo, ensembl_ids, e)
            break
        disease = (data or {}).get("disease")
        if not disease:
            break
        ev = disease.get("evidences") or {}
        batch = ev.get("rows") or []
        rows.extend(batch)
        cursor = ev.get("cursor")
        count = ev.get("count") or 0
        if not cursor or not batch or len(rows) >= count:
            break
    return rows


def _pubmed_years(pmids: Sequence[str], cfg: Optional[Dict]) -> Dict[str, int]:
    """Map PMID -> publication year via NCBI esummary (batched, cached)."""
    rc = _real_cfg(cfg)
    out: Dict[str, int] = {}
    batch = int(rc.get("pubmed_batch", 200))
    base = rc["pubmed_esummary_url"]
    pmids = [str(p) for p in pmids if p]
    for i in range(0, len(pmids), batch):
        chunk = pmids[i:i + batch]
        params = {"db": "pubmed", "retmode": "json", "id": ",".join(chunk)}
        cache_url = base + "?" + urllib.parse.urlencode(params)  # query only, no identifiers
        for opt, name in (("ncbi_api_key", "api_key"), ("ncbi_tool", "tool"),
                          ("ncbi_email", "email")):
            if rc.get(opt):
                params[name] = rc[opt]
        url = base + "?" + urllib.parse.urlencode(params)
        try:
            obj = _http_json(url, cfg, method="GET", cache_url=cache_url)
        except Exception as e:  # noqa: BLE001
            log.warning("PubMed esummary batch failed: %s", e)
            continue
        result = (obj or {}).get("result") or {}
        for pid in chunk:
            rec = result.get(pid) or {}
            year = _year_from_date(rec.get("pubdate") or rec.get("epubdate"))
            if year:
                out[pid] = year
    return out


def _evidence_year(er: Dict, etype: str, policy: str, pmid_year: Dict[str, int]) -> Optional[int]:
    """Resolve the YEAR for one evidence row per the configured literature policy.

    The result is passed through ``_sane_year`` so an impossible parsed year (e.g. a
    stray 2119 from a malformed field) is dropped rather than poisoning censoring.
    """
    def pmid_min() -> Optional[int]:
        years = [pmid_year[str(p)] for p in (er.get("literature") or []) if str(p) in pmid_year]
        return min(years) if years else None

    def _raw() -> Optional[int]:
        if etype == "literature":
            if policy == "drop":
                return None
            if policy == "ot_publication_year":
                return _year_from_date(er.get("publicationYear"))
            # "pubmed_year" (default): earliest resolvable PubMed year; DROP (return None)
            # if no PMID resolves. This is the drop-if-undatable contract the LAP metric
            # relies on -- do NOT fall back to publicationYear here (that is the separate
            # "ot_publication_year" policy), or the dropped-count is silently corrupted.
            return pmid_min()

        # Non-literature: prefer explicit dates, then PMID-derived year.
        for field in ("publicationYear", "evidenceDate", "studyStartDate", "releaseDate"):
            year = _year_from_date(er.get(field))
            if year:
                return year
        return pmid_min()

    return _sane_year(_raw())


# ---------------------------------------------------------------------------
# Public connectors
# ---------------------------------------------------------------------------

def load_clinicaltrials(cfg: Optional[Dict] = None) -> pd.DataFrame:
    """Build program rows (see REQUIRED_PROGRAM_COLS) from Open Targets + CT.gov."""
    cfg = cfg or get_config()
    rc = _real_cfg(cfg)
    _configure_logging(cfg)
    _check_data_version(cfg)

    target_cache: Dict[str, Optional[Tuple[str, str]]] = {}
    rows: List[Dict] = []
    stats = {"diseases": 0, "drug_rows": 0, "programs": 0,
             "no_aact_trials": 0, "no_target": 0}

    for efo_in in _disease_ids(rc):
        resolved = _resolve_disease_id(efo_in, cfg)
        if not resolved:
            log.warning("disease %r did not resolve to a current OT id; skipping", efo_in)
            continue
        efo, dname = resolved
        if efo != efo_in:
            log.info("resolved disease %r -> %s (%s)", efo_in, efo, dname)
        data = _fetch_disease_candidates(efo, cfg)
        disease = data.get("disease")
        if not disease:
            log.warning("disease %s (%s) returned no data; skipping", efo, dname)
            continue
        stats["diseases"] += 1
        area = _primary_area(disease)
        candidates = (disease.get("drugAndClinicalCandidates") or {}).get("rows") or []
        if rc.get("max_drug_rows"):
            candidates = candidates[: int(rc["max_drug_rows"])]

        for row in candidates:
            drug = row.get("drug") or {}
            chembl = drug.get("id")
            if not chembl:
                continue
            stats["drug_rows"] += 1

            trials = _collect_aact_trials(row.get("clinicalReports") or [])
            if not trials:
                stats["no_aact_trials"] += 1
                continue

            if chembl not in target_cache:
                target_cache[chembl] = _fetch_primary_target(chembl, cfg)
            tinfo = target_cache[chembl]
            if tinfo is None:
                stats["no_target"] += 1
                if rc.get("drop_unmapped_targets", True):
                    continue
                # Unique per-drug sentinel: keeps each unmapped drug its own pseudo-
                # target so target-disjoint splitting does not merge them into one.
                ensembl = f"UNKNOWN__{chembl}"
            else:
                ensembl = tinfo[0]

            modality = _modality(drug.get("drugType"))
            entry_year, dated_trials = _phase_timeline(trials)
            for phase in (1, 2, 3):
                if phase not in entry_year:
                    continue
                info_time = entry_year[phase]
                # label = advanced to phase+1 AFTER the freeze point: a date-stamped
                # trial at a higher clinical phase that started strictly after
                # info_time. Advancement at or before info_time is NOT counted -- it
                # was already knowable at the freeze and would be label leakage.
                label = int(any(
                    b >= phase + 1 and y > info_time for (b, y) in dated_trials
                ))
                rows.append({
                    "program_id": f"{chembl}__{efo}__P{phase}",
                    "target_id": ensembl,
                    "indication": efo,
                    "therapeutic_area": area,
                    "sponsor_tier": _program_sponsor_tier(trials, phase, info_time, cfg),
                    "modality": modality,
                    "info_time": int(info_time),
                    "phase_from": int(phase),
                    "label": label,
                })
                stats["programs"] += 1

    programs = pd.DataFrame(rows, columns=REQUIRED_PROGRAM_COLS)
    programs = programs.drop_duplicates("program_id").reset_index(drop=True)
    log.info("programs built: %s", stats)
    if programs.empty:
        raise RuntimeError(
            "no programs constructed -- check cfg['real']['disease_ids'], network access, "
            "and (if a new Open Targets release shipped) the GraphQL field names."
        )
    return programs


def load_open_targets_evidence(
    programs: Optional[pd.DataFrame] = None, cfg: Optional[Dict] = None
) -> pd.DataFrame:
    """Build evidence rows (see REQUIRED_EVIDENCE_COLS) for each program's (disease, target).

    ``programs`` is required to key evidence to program_ids; if omitted it is
    built via ``load_clinicaltrials(cfg)``. Evidence describes a (disease, target)
    pair, so it is shared across every program sharing that pair.
    """
    cfg = cfg or get_config()
    rc = _real_cfg(cfg)
    _configure_logging(cfg)
    if programs is None:
        programs = load_clinicaltrials(cfg)

    # (indication EFO, target Ensembl) -> [program_id, ...]
    pairs: Dict[Tuple[str, str], List[str]] = {}
    for _, r in programs.iterrows():
        ensembl = r["target_id"]
        if not ensembl or str(ensembl).startswith("UNKNOWN"):
            continue  # no real Ensembl id -> no target-disease evidence to fetch
        pairs.setdefault((r["indication"], ensembl), []).append(r["program_id"])

    # Fetch raw evidence per pair; gather PMIDs for year resolution. Literature
    # co-occurrence evidence can be enormous (tens of thousands of rows per pair),
    # which also blows up the PubMed year-resolution cost, so it is optionally
    # capped per pair with a seeded random sample (see below).
    per_pair: Dict[Tuple[str, str], List[Dict]] = {}
    all_pmids = set()
    max_lit = rc.get("max_literature_per_pair")
    for (efo, ensembl) in pairs:
        erows = _fetch_evidences(efo, [ensembl], cfg)
        if max_lit:
            lit = [e for e in erows if e.get("datatypeId") == "literature"]
            other = [e for e in erows if e.get("datatypeId") != "literature"]
            if len(lit) > int(max_lit):
                # Seeded RANDOM sample (deterministic per pair), NOT top-by-score:
                # score-ranking biases the temporal distribution and would corrupt the
                # leakage/LAP measurement, which is precisely about publication timing.
                rng = random.Random(f"{efo}|{ensembl}")
                lit = rng.sample(lit, int(max_lit))
            erows = other + lit
        per_pair[(efo, ensembl)] = erows
        for er in erows:
            for pmid in er.get("literature") or []:
                all_pmids.add(str(pmid))

    policy = rc.get("literature_policy", "pubmed_year")
    need_pmids = all_pmids and (policy == "pubmed_year" or rc.get("use_pmid_year_fallback", True))
    pmid_year = _pubmed_years(sorted(all_pmids), cfg) if need_pmids else {}

    ev_rows: List[Tuple] = []
    stats = {"kept": 0, "dropped_undated": 0, "dropped_lit_undated": 0,
             "dropped_unknown_type": 0, "dropped_no_score": 0}
    for (efo, ensembl), erows in per_pair.items():
        pids = pairs[(efo, ensembl)]
        for er in erows:
            etype = DATATYPE_TO_EVIDENCE.get(er.get("datatypeId"))
            if etype is None:
                stats["dropped_unknown_type"] += 1
                continue
            score = er.get("score")
            if score is None:
                stats["dropped_no_score"] += 1
                continue
            year = _evidence_year(er, etype, policy, pmid_year)
            if year is None:
                stats["dropped_lit_undated" if etype == "literature" else "dropped_undated"] += 1
                continue
            for pid in pids:
                ev_rows.append((pid, etype, float(score), int(year)))
            stats["kept"] += 1

    evidence = pd.DataFrame(ev_rows, columns=REQUIRED_EVIDENCE_COLS)
    log.info("evidence built (policy=%s): %s", policy, stats)
    return evidence


def validate(programs: pd.DataFrame, evidence: pd.DataFrame) -> None:
    missing_p = set(REQUIRED_PROGRAM_COLS) - set(programs.columns)
    missing_e = set(REQUIRED_EVIDENCE_COLS) - set(evidence.columns)
    if missing_p:
        raise ValueError(f"programs is missing columns: {sorted(missing_p)}")
    if missing_e:
        raise ValueError(f"evidence is missing columns: {sorted(missing_e)}")
    if not programs["label"].isin([0, 1]).all():
        raise ValueError("programs.label must be 0/1")
    if not programs["phase_from"].isin([1, 2, 3]).all():
        raise ValueError("programs.phase_from must be in {1, 2, 3}")
    if programs["program_id"].duplicated().any():
        raise ValueError("programs.program_id must be unique")
    orphan = set(evidence["program_id"]) - set(programs["program_id"])
    if orphan:
        raise ValueError(f"{len(orphan)} evidence rows reference unknown program_ids")
    # Guard against impossible evidence years surviving the date parser (a bogus
    # future year would be "post-decision" for every program and silently corrupt
    # as-of-time censoring). _sane_year should have dropped these upstream.
    if len(evidence):
        hi = time.gmtime().tm_year + 1
        bad = ~evidence["evidence_date"].between(_MIN_EVIDENCE_YEAR, hi)
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} evidence rows have evidence_date outside "
                f"[{_MIN_EVIDENCE_YEAR}, {hi}] (parser latched onto a bad year)")


def build_real_dataset(cfg: Dict) -> Tuple[str, str]:
    """Assemble + validate real data and write it to cfg['data_dir']."""
    _configure_logging(cfg)
    programs = load_clinicaltrials(cfg)
    evidence = load_open_targets_evidence(programs, cfg)
    validate(programs, evidence)
    out_dir = cfg["data_dir"]
    os.makedirs(out_dir, exist_ok=True)
    p_path = os.path.join(out_dir, "programs.csv")
    e_path = os.path.join(out_dir, "evidence.csv")
    programs.to_csv(p_path, index=False)
    evidence.to_csv(e_path, index=False)
    log.info("wrote %d programs -> %s and %d evidence rows -> %s",
             len(programs), p_path, len(evidence), e_path)
    return p_path, e_path


if __name__ == "__main__":  # pragma: no cover -- convenience entry point
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    build_real_dataset(get_config(config_path))
