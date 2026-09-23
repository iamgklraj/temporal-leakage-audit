"""Offline tests for the real-data connectors.

Every network call is monkeypatched, so this runs with no internet, no auth and
no live API. It exercises the pure parsers plus the full program/evidence
construction against fixture payloads shaped like real Open Targets / CT.gov /
PubMed responses.

Run with either:  pytest tests/  OR  python tests/test_connectors.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal_leakage_audit.config import get_config  # noqa: E402
from temporal_leakage_audit.data import connectors as C  # noqa: E402


def _cfg():
    cfg = get_config()
    cfg["real"]["disease_ids"] = ["EFO_TEST"]
    cfg["real"]["check_version"] = False
    cfg["real"]["cache_dir"] = None          # no disk cache in tests
    cfg["real"]["sleep_between"] = 0
    return cfg


# --------------------------------------------------------------------------- #
# Pure parsers
# --------------------------------------------------------------------------- #

def test_extract_nct():
    assert C._extract_nct("https://clinicaltrials.gov/study/NCT04852770") == "NCT04852770"
    assert C._extract_nct("foo NCT00000001 bar") == "NCT00000001"
    assert C._extract_nct("no id here") is None
    assert C._extract_nct(None) is None
    print("OK: NCT extraction")


def test_year_from_date():
    assert C._year_from_date("2020-04-06") == 2020
    assert C._year_from_date("2019-11") == 2019
    assert C._year_from_date("2015") == 2015
    assert C._year_from_date(2015) == 2015
    assert C._year_from_date(None) is None
    assert C._year_from_date("no year") is None
    print("OK: year parsing")


def test_sponsor_tier_includes_ambig():
    assert C._sponsor_tier("INDUSTRY") == 2
    assert C._sponsor_tier("NIH") == 1
    assert C._sponsor_tier("NETWORK") == 1
    assert C._sponsor_tier("OTHER") == 0
    assert C._sponsor_tier("AMBIG") == 0            # the value the report flagged as missing
    assert C._sponsor_tier("SOMETHING_NEW") == 0    # default for unknown/future values
    assert C._sponsor_tier(None) == 0
    print("OK: sponsor-class tiers (incl. AMBIG)")


def test_modality_mapping():
    assert C._modality("Small molecule") == "small_molecule"
    assert C._modality("Antibody") == "biologic"
    assert C._modality("Protein") == "biologic"
    assert C._modality("Oligonucleotide") == "other"
    assert C._modality(None) == "other"
    print("OK: modality mapping")


def test_stage_maps_cover_full_vocabulary():
    vocab = [
        "WITHDRAWAL", "APPROVAL", "PHASE_4", "PREAPPROVAL", "PHASE_3",
        "PHASE_2_3", "PHASE_2", "PHASE_1_2", "PHASE_1", "EARLY_PHASE_1",
        "IND", "PRECLINICAL", "UNKNOWN",
    ]
    for s in vocab:
        assert s in C.STAGE_TO_NUM, f"{s} missing from STAGE_TO_NUM"
        assert s in C.STAGE_BIN, f"{s} missing from STAGE_BIN"
    # Sanity: a PHASE_1_2 trial sits at phase 1; APPROVAL/PHASE_4 sit at phase 4.
    assert C.STAGE_BIN["PHASE_1_2"] == 1
    assert C.STAGE_BIN["PHASE_2_3"] == 2
    assert C.STAGE_BIN["APPROVAL"] == 4 and C.STAGE_BIN["PHASE_4"] == 4
    print("OK: clinicalStage vocabulary fully mapped")


def test_parse_ctgov_study():
    study = {
        "protocolSection": {
            "statusModule": {
                "overallStatus": "TERMINATED",
                "startDateStruct": {"date": "2020-04-06"},
            },
            "designModule": {"phases": ["PHASE2", "PHASE3"]},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Acme Pharma", "class": "INDUSTRY"}
            },
        }
    }
    parsed = C._parse_ctgov_study(study)
    assert parsed["start_year"] == 2020
    assert parsed["status"] == "TERMINATED"
    assert parsed["phases"] == ["PHASE2", "PHASE3"]
    assert parsed["sponsor_class"] == "INDUSTRY"

    # Missing startDateStruct must not raise.
    empty = C._parse_ctgov_study({"protocolSection": {"statusModule": {}}})
    assert empty["start_year"] is None and empty["sponsor_class"] is None
    print("OK: CT.gov v2 study parsing (top-level protocolSection)")


def test_phase_timeline():
    trials = [
        {"bin": 1, "start_year": 2012},
        {"bin": 1, "start_year": 2010},   # earlier phase-1 start wins for entry_year
        {"bin": 2, "start_year": 2014},
        {"bin": 0, "start_year": 2009},   # IND/preclinical -> ignored
        {"bin": 3, "start_year": None},   # undated -> ignored
    ]
    entry_year, dated = C._phase_timeline(trials)
    assert entry_year == {1: 2010, 2: 2014}
    assert sorted(dated) == [(1, 2010), (1, 2012), (2, 2014)]  # only dated clinical trials
    print("OK: phase timeline (entry years + dated trials)")


# --------------------------------------------------------------------------- #
# Program construction (Open Targets QUERY A + CT.gov, mocked)
# --------------------------------------------------------------------------- #

_DISEASE_PAYLOAD = {
    "disease": {
        "id": "EFO_TEST",
        "name": "test disease",
        "therapeuticAreas": [{"id": "EFO_AREA", "name": "immune system disease"}],
        "drugAndClinicalCandidates": {
            "count": 2,
            "rows": [
                {   # dated Phase I(2010) -> II(2013) -> III(2016); industry-sponsored
                    "maxClinicalStage": "PHASE_3",
                    "drug": {"id": "CHEMBL1", "name": "DrugA",
                             "drugType": "Small molecule", "maximumClinicalStage": "PHASE_3"},
                    "clinicalReports": [
                        {"id": "r1", "source": "AACT", "type": "CLINICAL_TRIAL",
                         "url": "https://clinicaltrials.gov/study/NCT00000001",
                         "clinicalStage": "PHASE_1", "trialOverallStatus": "COMPLETED",
                         "trialStartDate": "2010-01-01"},
                        {"id": "r2", "source": "AACT", "type": "CLINICAL_TRIAL",
                         "url": "https://clinicaltrials.gov/study/NCT00000002",
                         "clinicalStage": "PHASE_2", "trialOverallStatus": "COMPLETED",
                         "trialStartDate": "2013-06-01"},
                        {"id": "r5", "source": "AACT", "type": "CLINICAL_TRIAL",
                         "url": "https://clinicaltrials.gov/study/NCT00000005",
                         "clinicalStage": "PHASE_3", "trialOverallStatus": "RECRUITING",
                         "trialStartDate": "2016-02-01"},
                        {"id": "r3", "source": "ChEMBL", "type": "DRUG_LABEL",
                         "url": "https://example.org/label", "clinicalStage": "PHASE_3"},
                    ],
                },
                {   # only a Phase I trial, never advanced -> label 0; drug has no target
                    "maxClinicalStage": "PHASE_1",
                    "drug": {"id": "CHEMBL2", "name": "DrugB",
                             "drugType": "Antibody", "maximumClinicalStage": "PHASE_1"},
                    "clinicalReports": [
                        {"id": "r4", "source": "AACT", "type": "CLINICAL_TRIAL",
                         "url": "https://clinicaltrials.gov/study/NCT00000003",
                         "clinicalStage": "PHASE_1", "trialOverallStatus": "TERMINATED",
                         "trialStartDate": "2016-03-01"},
                    ],
                },
            ],
        },
    }
}

_TARGETS = {"CHEMBL1": ("ENSG000001", "TP53"), "CHEMBL2": None}

_CTGOV = {
    "NCT00000001": {"sponsor_class": "INDUSTRY"},
    "NCT00000002": {"sponsor_class": "OTHER"},
    "NCT00000003": {"sponsor_class": "NIH"},
    "NCT00000005": {"sponsor_class": "NIH"},
}


def _patch_ct(monkeypatch):
    monkeypatch.setattr(C, "_check_data_version", lambda cfg: {})
    monkeypatch.setattr(C, "_resolve_disease_id", lambda v, cfg: (v, v))
    monkeypatch.setattr(C, "_fetch_disease_candidates", lambda efo, cfg: _DISEASE_PAYLOAD)
    monkeypatch.setattr(C, "_fetch_primary_target", lambda chembl, cfg: _TARGETS.get(chembl))
    monkeypatch.setattr(C, "_fetch_ctgov", lambda nct, cfg: _CTGOV.get(nct))


def test_load_clinicaltrials(monkeypatch):
    _patch_ct(monkeypatch)
    programs = C.load_clinicaltrials(_cfg())

    # CHEMBL2 has no Ensembl target and is dropped by default; CHEMBL1 yields P1/P2/P3.
    assert set(programs["program_id"]) == {
        "CHEMBL1__EFO_TEST__P1", "CHEMBL1__EFO_TEST__P2", "CHEMBL1__EFO_TEST__P3"}
    prog = programs.set_index("program_id")

    p1 = prog.loc["CHEMBL1__EFO_TEST__P1"]
    assert p1["target_id"] == "ENSG000001"
    assert p1["indication"] == "EFO_TEST"
    assert p1["therapeutic_area"] == "immune_system_disease"
    assert p1["modality"] == "small_molecule"
    assert int(p1["phase_from"]) == 1
    assert int(p1["info_time"]) == 2010          # earliest Phase I start
    assert int(p1["label"]) == 1                  # Phase II trial (2013) started after 2010
    assert int(p1["sponsor_tier"]) == 2           # NCT1 is INDUSTRY at phase 1

    p2 = prog.loc["CHEMBL1__EFO_TEST__P2"]
    assert int(p2["info_time"]) == 2013
    assert int(p2["label"]) == 1                  # Phase III trial (2016) started after 2013
    assert int(p2["sponsor_tier"]) == 0           # NCT2 (phase-2 trial) is OTHER

    p3 = prog.loc["CHEMBL1__EFO_TEST__P3"]
    assert int(p3["info_time"]) == 2016
    assert int(p3["label"]) == 0                  # no dated Phase IV/approval trial after 2016
    assert int(p3["sponsor_tier"]) == 1           # NCT5 (phase-3 trial) is NIH
    print("OK: program construction (phase_from / info_time / label / tier / target)")


def test_load_clinicaltrials_keeps_unmapped_when_configured(monkeypatch):
    _patch_ct(monkeypatch)
    cfg = _cfg()
    cfg["real"]["drop_unmapped_targets"] = False
    programs = C.load_clinicaltrials(cfg)
    # CHEMBL2 survives with a UNIQUE per-drug sentinel target (not a shared "UNKNOWN").
    b = programs.set_index("program_id").loc["CHEMBL2__EFO_TEST__P1"]
    assert b["target_id"] == "UNKNOWN__CHEMBL2"
    assert int(b["label"]) == 0
    print("OK: unmapped-target programs kept with unique sentinel when configured")


# Leakage regression: the label must NOT be set by (a) an undated global drug
# maxstage hint, or (b) higher-phase advancement that predates the freeze point.
_LEAKAGE_PAYLOAD = {
    "disease": {
        "id": "EFO_TEST",
        "name": "test disease",
        "therapeuticAreas": [{"id": "A", "name": "cancer"}],
        "drugAndClinicalCandidates": {"count": 2, "rows": [
            {   # only a Phase I trial in RA, but drug is APPROVED elsewhere (undated, global)
                "maxClinicalStage": "APPROVAL",
                "drug": {"id": "CHEMBLX", "drugType": "Small molecule",
                         "maximumClinicalStage": "APPROVAL"},
                "clinicalReports": [
                    {"id": "x1", "source": "AACT", "type": "CLINICAL_TRIAL",
                     "url": "https://clinicaltrials.gov/study/NCT01000001",
                     "clinicalStage": "PHASE_1", "trialStartDate": "2018-01-01"},
                ],
            },
            {   # a Phase III trial (2011) that PREDATES the Phase II entry (2013)
                "maxClinicalStage": "PHASE_3",
                "drug": {"id": "CHEMBLY", "drugType": "Small molecule",
                         "maximumClinicalStage": "PHASE_3"},
                "clinicalReports": [
                    {"id": "y1", "source": "AACT", "type": "CLINICAL_TRIAL",
                     "url": "https://clinicaltrials.gov/study/NCT01000002",
                     "clinicalStage": "PHASE_2", "trialStartDate": "2013-05-01"},
                    {"id": "y2", "source": "AACT", "type": "CLINICAL_TRIAL",
                     "url": "https://clinicaltrials.gov/study/NCT01000003",
                     "clinicalStage": "PHASE_3", "trialStartDate": "2011-05-01"},
                ],
            },
        ]},
    }
}


def test_label_no_leakage(monkeypatch):
    monkeypatch.setattr(C, "_check_data_version", lambda cfg: {})
    monkeypatch.setattr(C, "_resolve_disease_id", lambda v, cfg: (v, v))
    monkeypatch.setattr(C, "_fetch_disease_candidates", lambda efo, cfg: _LEAKAGE_PAYLOAD)
    monkeypatch.setattr(C, "_fetch_primary_target",
                        lambda chembl, cfg: (f"ENSG_{chembl}", chembl))
    monkeypatch.setattr(C, "_fetch_ctgov", lambda nct, cfg: {"sponsor_class": "OTHER"})
    prog = C.load_clinicaltrials(_cfg()).set_index("program_id")

    # (a) An undated 'Approval' (global, cross-indication) must NOT make P1 positive.
    assert int(prog.loc["CHEMBLX__EFO_TEST__P1"]["label"]) == 0
    # (b) A Phase III trial that started (2011) BEFORE the Phase II freeze (2013) must
    #     NOT make P2 positive -- advancement was not resolved after info_time.
    p2 = prog.loc["CHEMBLY__EFO_TEST__P2"]
    assert int(p2["info_time"]) == 2013 and int(p2["label"]) == 0
    print("OK: label is leakage-free (no undated hints, advancement must postdate freeze)")


# --------------------------------------------------------------------------- #
# Evidence construction (Open Targets QUERY B + PubMed, mocked)
# --------------------------------------------------------------------------- #

_EVIDENCE_ROWS = [
    {"datasourceId": "ot_genetics_portal", "datatypeId": "genetic_association",
     "score": 0.7, "publicationYear": 2011, "literature": ["111"]},
    {"datasourceId": "eva_somatic", "datatypeId": "somatic_mutation",
     "score": 0.4, "publicationYear": None, "literature": ["222"], "evidenceDate": "2009-05-01"},
    {"datasourceId": "europepmc", "datatypeId": "literature",
     "score": 0.3, "publicationYear": 2020, "literature": ["333", "444"]},   # datable via PubMed
    {"datasourceId": "europepmc", "datatypeId": "literature",
     "score": 0.9, "publicationYear": None, "literature": ["999"]},          # PMID has no year -> drop
    {"datasourceId": "chembl", "datatypeId": "known_drug",
     "score": 1.0, "publicationYear": None, "literature": []},               # no year anywhere -> drop
    {"datasourceId": "x", "datatypeId": "brand_new_datatype",
     "score": 0.5, "publicationYear": 2018, "literature": []},               # unknown type -> drop
]

_PMID_YEARS = {"111": 2011, "222": 2008, "333": 2018, "444": 2016}  # note "999" absent


def _patch_ev(monkeypatch):
    monkeypatch.setattr(C, "_fetch_evidences", lambda efo, ens, cfg: list(_EVIDENCE_ROWS))
    monkeypatch.setattr(C, "_pubmed_years", lambda pmids, cfg: dict(_PMID_YEARS))


def test_load_open_targets_evidence(monkeypatch):
    _patch_ct(monkeypatch)
    _patch_ev(monkeypatch)
    cfg = _cfg()
    programs = C.load_clinicaltrials(cfg)          # P1/P2/P3, same (EFO_TEST, ENSG000001)
    evidence = C.load_open_targets_evidence(programs, cfg)

    # Types kept: genetic, somatic, literature(datable), known_drug is dropped (no year).
    kept_types = set(evidence["evidence_type"])
    assert kept_types == {"genetic", "somatic", "literature"}

    # Each kept datum is fanned out to ALL programs sharing the (disease, target) pair.
    per_pid = evidence.groupby("program_id").size()
    assert set(per_pid.index) == {
        "CHEMBL1__EFO_TEST__P1", "CHEMBL1__EFO_TEST__P2", "CHEMBL1__EFO_TEST__P3"}
    assert (per_pid == 3).all(), per_pid.to_dict()

    ev1 = evidence[evidence["program_id"] == "CHEMBL1__EFO_TEST__P1"].set_index("evidence_type")
    assert int(ev1.loc["genetic", "evidence_date"]) == 2011
    # somatic: publicationYear null -> explicit evidenceDate (2009) is used before any
    # PMID-derived year (PMID 222 -> 2008), matching the report's fallback precedence.
    assert int(ev1.loc["somatic", "evidence_date"]) == 2009
    # literature: min PubMed year across PMIDs 333(2018),444(2016) -> 2016
    assert int(ev1.loc["literature", "evidence_date"]) == 2016
    print("OK: evidence construction (typing, dating, PubMed fallback, fan-out)")


def test_literature_policy_drop(monkeypatch):
    _patch_ct(monkeypatch)
    _patch_ev(monkeypatch)
    cfg = _cfg()
    cfg["real"]["literature_policy"] = "drop"
    programs = C.load_clinicaltrials(cfg)
    evidence = C.load_open_targets_evidence(programs, cfg)
    assert "literature" not in set(evidence["evidence_type"])
    print("OK: literature_policy='drop' removes all literature")


def test_literature_drop_if_undatable(monkeypatch):
    """Under pubmed_year, literature with an unresolvable PMID is DROPPED even if it
    has an OT publicationYear (no silent fallback -- keeps the LAP drop-count exact)."""
    _patch_ct(monkeypatch)
    # One literature datum whose PMID does not resolve, but publicationYear is set.
    undatable = [{"datasourceId": "europepmc", "datatypeId": "literature",
                  "score": 0.5, "publicationYear": 2019, "literature": ["777"]}]
    monkeypatch.setattr(C, "_fetch_evidences", lambda efo, ens, cfg: list(undatable))
    monkeypatch.setattr(C, "_pubmed_years", lambda pmids, cfg: {})   # nothing resolves
    programs = C.load_clinicaltrials(_cfg())

    cfg = _cfg()
    assert cfg["real"]["literature_policy"] == "pubmed_year"
    ev = C.load_open_targets_evidence(programs, cfg)
    assert len(ev) == 0, "undatable literature must be dropped, not dated by publicationYear"

    # The dedicated ot_publication_year policy DOES keep it (dated 2019).
    cfg2 = _cfg()
    cfg2["real"]["literature_policy"] = "ot_publication_year"
    ev2 = C.load_open_targets_evidence(programs, cfg2)
    assert set(ev2["evidence_date"]) == {2019}
    print("OK: pubmed_year drops undatable literature; ot_publication_year keeps it")


def test_evidence_validates_and_feeds_features(monkeypatch):
    _patch_ct(monkeypatch)
    _patch_ev(monkeypatch)
    cfg = _cfg()
    programs = C.load_clinicaltrials(cfg)
    evidence = C.load_open_targets_evidence(programs, cfg)

    # Contract validation must pass.
    C.validate(programs, evidence)

    # And the frames must actually flow through the real feature builder.
    from temporal_leakage_audit import features
    X_c, X_n, y, meta, groups = features.build(programs, evidence)
    assert len(X_c) == len(programs) == len(y)
    # Genetic support is present as-of-t for the datable genetic evidence.
    assert X_c["has_genetic_support"].sum() >= 1
    print("OK: real frames validate and build features")


# --------------------------------------------------------------------------- #
# Evidence pagination (cursor loop, mocked GraphQL)
# --------------------------------------------------------------------------- #

def test_fetch_evidences_cursor_pagination(monkeypatch):
    calls = {"n": 0}

    def fake_graphql(query, variables, cfg, use_cache=True):
        calls["n"] += 1
        if variables["cursor"] is None:
            return {"disease": {"evidences": {
                "count": 3, "cursor": "CUR",
                "rows": [{"datasourceId": "a", "datatypeId": "genetic_association",
                          "score": 0.1, "literature": []},
                         {"datasourceId": "b", "datatypeId": "genetic_association",
                          "score": 0.2, "literature": []}]}}}
        return {"disease": {"evidences": {
            "count": 3, "cursor": None,
            "rows": [{"datasourceId": "c", "datatypeId": "genetic_association",
                      "score": 0.3, "literature": []}]}}}

    monkeypatch.setattr(C, "_graphql", fake_graphql)
    rows = C._fetch_evidences("EFO_TEST", ["ENSG000001"], _cfg())
    assert len(rows) == 3
    assert calls["n"] == 2                      # two pages, then stopped on null cursor
    print("OK: evidence cursor pagination")


def test_fetch_evidences_date_field_fallback(monkeypatch):
    """If the live schema rejects the date fields, we retry with core fields only."""
    attempts = {"n": 0}

    def fake_graphql(query, variables, cfg, use_cache=True):
        attempts["n"] += 1
        if "evidenceDate" in query:
            raise C.GraphQLError("Cannot query field 'evidenceDate' on type 'Evidence'")
        return {"disease": {"evidences": {"count": 1, "cursor": None,
                "rows": [{"datasourceId": "a", "datatypeId": "genetic_association",
                          "score": 0.5, "literature": []}]}}}

    monkeypatch.setattr(C, "_graphql", fake_graphql)
    rows = C._fetch_evidences("EFO_TEST", ["ENSG000001"], _cfg())
    assert len(rows) == 1
    assert attempts["n"] == 2                    # full query failed, minimal query succeeded
    print("OK: evidence date-field fallback")


# --------------------------------------------------------------------------- #
# Transport robustness (findings 3/4/8)
# --------------------------------------------------------------------------- #

def test_graphql_does_not_cache_error_bodies(monkeypatch):
    captured = {}

    def fake_http_json(url, cfg, *, method="GET", body=None, use_cache=True, cache_ok=None):
        captured["cache_ok"] = cache_ok
        return {"data": {"ok": 1}}

    monkeypatch.setattr(C, "_http_json", fake_http_json)
    C._graphql("query { x }", {}, _cfg())
    ok = captured["cache_ok"]
    assert ok is not None
    assert ok({"data": {}}) is True                                  # good body -> cacheable
    assert ok({"data": None, "errors": [{"message": "boom"}]}) is False  # error body -> not cached
    print("OK: GraphQL error bodies are never cached")


def test_http_json_retries_malformed_json(monkeypatch):
    calls = {"n": 0}

    class _FakeResp:
        def read(self):
            return b"not json <<<"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return _FakeResp()

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    cfg = _cfg()
    cfg["real"]["max_retries"] = 3
    cfg["real"]["backoff_base"] = 0
    try:
        C._http_json("http://example.org", cfg, method="GET")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    except ValueError:
        assert False, "JSONDecodeError leaked instead of being retried/wrapped"
    assert calls["n"] == 3   # every attempt retried, not aborted on the first bad body
    print("OK: malformed JSON body is retried and surfaces as RuntimeError")


def test_fetch_primary_target_degrades_on_transport_error(monkeypatch):
    def boom(query, variables, cfg, use_cache=True):
        raise RuntimeError("HTTP 503 for ...")   # plain transport error, not GraphQLError

    monkeypatch.setattr(C, "_graphql", boom)
    assert C._fetch_primary_target("CHEMBLZ", _cfg()) is None
    print("OK: target fetch degrades to None on transport error (does not abort run)")


def test_pubmed_cache_key_ignores_identifiers(monkeypatch):
    """NCBI tool/email/api_key go in the request URL but not in the cache key."""
    seen = {}

    def fake_http(url, cfg, method="GET", cache_url=None, **kw):
        seen["url"], seen["cache_url"] = url, cache_url
        return {"result": {"1": {"pubdate": "2001 Jan"}}}

    monkeypatch.setattr(C, "_http_json", fake_http)
    cfg = _cfg()
    cfg["real"].update({"ncbi_tool": "t", "ncbi_email": "e@x.org", "ncbi_api_key": "k"})
    assert C._pubmed_years(["1"], cfg) == {"1": 2001}
    assert "tool=t" in seen["url"] and "api_key=k" in seen["url"]
    for token in ("tool=", "email=", "api_key="):
        assert token not in seen["cache_url"]
    print("OK: PubMed cache key excludes identification parameters")


# --------------------------------------------------------------------------- #
# Minimal monkeypatch shim so the file also runs without pytest installed
# --------------------------------------------------------------------------- #

class _MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)


if __name__ == "__main__":
    import inspect

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        if "monkeypatch" in inspect.signature(fn).parameters:
            mp = _MonkeyPatch()
            try:
                fn(mp)
            finally:
                mp.undo()
        else:
            fn()
    print(f"all {len(tests)} connector tests passed")
