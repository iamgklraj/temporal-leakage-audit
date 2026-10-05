"""Second patient-level demonstration of the leakage-response method: sepsis onset in the ICU.

Data: PhysioNet/Computing in Cardiology Challenge 2019, "Early Prediction of Sepsis from Clinical
Data" (physionet.org/content/challenge-2019/1.0.0; open access under the Creative Commons Attribution
4.0 International Public License stated on the project page; the LICENSE.txt bundled with the files
holds the ODC Open Database License text). One pipe-separated file per ICU stay with one row per hour
since ICU admission: 34 time-dependent variables (8 vital signs, 26 laboratory values; several values
within an hour are summarised by their median; NaN = nothing recorded in that hour), demographics
(Age, Gender, Unit1 = medical ICU, Unit2 = surgical ICU, HospAdmTime = hours between hospital and ICU
admission), ICULOS (hours since ICU admission) and SepsisLabel. Hospital system A (Beth Israel
Deaconess Medical Center; training_setA, 20,336 stays) and hospital system B (Emory University
Hospital; training_setB, 20,000 stays). The project page's link to a single archive no longer
resolves, so the files are fetched one by one from the open-access AWS mirror listed on the page
(s3://physionet-open/challenge-2019/1.0.0/) over 8 persistent connections and checked against the
MD5 ETags of the bucket listing (about 255 MB, 40,336 files, a few minutes).

Label (project page; Reyna et al., Crit Care Med 2020;48:210-217): t_sepsis is the earlier of
t_suspicion (IV antibiotics and blood cultures within 24 h or 72 h of each other, antibiotics given
for at least 72 consecutive hours) and t_SOFA (a two-point SOFA increase within 24 h), provided that
t_suspicion - 24 <= t_SOFA <= t_suspicion + 12. For septic patients SepsisLabel = 1 for
t >= t_sepsis - 6 (labels are shifted six hours ahead) and 0 before; for other patients it is always
0. Hence t_sepsis = the first ICULOS with SepsisLabel = 1, plus 6 (the challenge scoring code,
evaluate_sepsis_score.py, likewise sets t_sepsis = first positive row + 6 and scores predictions
up to t_sepsis + 3). Cohort rules (manuscript, Section 2.2): stays with fewer than 8 hourly windows
and patients with t_sepsis less than 4 h after ICU admission were not included; records were
truncated at ICU discharge and after 2 weeks. The documentation states no other truncation, but the
files show two more (descriptive counts under "data_checks" in the output): every septic record ends
at most 3 h after t_sepsis, the end of the scoring window (1,786 of 1,790 septic stays in A and
1,127 of 1,142 in B end 1-3 h after it; the rest end at or before it; at most 10 labelled rows per
stay), and non-septic records end by ICU hour 60 in all but 8 of 18,546 stays in A and 218 of 18,858
in B. When a record ends thus depends on the outcome, and measurement counts admitted after the
decision encode it (see "record_end_check" per design). Other facts: the files also hold hourly rows
without any vital sign or laboratory value (without them the row counts equal the manuscript's
739,663 and 684,508); 37% of records in A and 6.5% in B start after ICU hour 1; EtCO2 is never
recorded in A, so the learner cannot use it; Unit1 and Unit2 are complementary when recorded and
both missing in 47% (A) and 30% (B) of stays.

Design (pre-specified before any model was fitted and not changed afterwards). Units are ICU stays;
training = hospital system A, test = hospital system B (a fixed, external test set). Decision time
t_d = 24 h after ICU admission (primary) and t_d = 12 h (sensitivity analysis). Hours are aligned
by ICULOS: the row with ICULOS = k summarises hour k of the stay, and rows with ICULOS <= c count as
recorded by hour c (by the decision when c = t_d). Eligible at t_d: the stay has a row at
ICULOS >= t_d (still in the ICU at the decision) and is not septic at the decision, t_sepsis > t_d
(stays that are never septic are eligible). Outcome: y = 1 if t_d < t_sepsis <= t_d + 24 (sepsis
onset within the next 24 h), else 0. Features at a cut-off c = t_d + h: for each of the 34 variables,
the last, minimum, maximum and mean of its non-missing hourly values and the number of hours with a
value (a care-process signal) recorded by c, plus Age, Gender, HospAdmTime and the unit indicators
Unit1 and Unit2 as recorded (missing values left to the learner). ICULOS, record length, SepsisLabel
and anything derived from them are never features. Horizons h = 0, 6, 12, 24 h: h = 0 is the
deployable matrix and h = 24 the naive one; LAP = AUPRC(h = 24) - AUPRC(h = 0) on the same test
stays. Per-source placebo: measured values (last, min, max, mean) at h = 24 with counts at h = 0, and
counts at h = 24 with values at h = 0; per-variable placebo (exploratory): one variable's five
features at h = 24, the rest at h = 0. Decision level (naive versus deployable): precision among the
top 10% of test stays (as in decision_impact.py) and net benefit at risk thresholds 0.02, 0.05, 0.10,
0.15 and 0.20 (formula of decision_curve_icu.py). Learner, features and bootstrap mirror
audit_physionet2012.py: models.fit_gbdt (HistGradientBoosting, max_depth 3, lr 0.06, at most 300
iterations, l2 1.0, seed 0); patient-level paired percentile bootstrap, 1,000 resamples, seed 0.
Two technical notes: fit_gbdt keeps scikit-learn's early_stopping="auto", which is active here
because every training set exceeds 10,000 stays (10% stratified validation split, seed 0; the
iterations used are reported), unlike the 8,000-stay 2012 audit; and columns never observed in the
training rows (the EtCO2 values) are dropped before fitting, since the learner rejects all-missing
columns and could not split on them.

Usage:  python scripts/audit_physionet2019_sepsis.py [--out outputs/physionet2019_sepsis_audit.json]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import http.client  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import audit_physionet2012 as E  # noqa: E402  (paired bootstrap, percentile CI)
import decision_curve_icu as DC  # noqa: E402  (net benefit and treat-all reference)
from temporal_leakage_audit import metrics as M  # noqa: E402
from temporal_leakage_audit import models  # noqa: E402

S3_HOST = "physionet-open.s3.amazonaws.com"
S3_PREFIX = "challenge-2019/1.0.0/training/"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
N_CONN = 8
DATA_DIR = os.path.join(ROOT, "data", "physionet2019")
SETS = {"A": ("training_setA", 20336), "B": ("training_setB", 20000)}   # hospital system: (folder, stays)
SERIES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",                     # vital signs
          "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",  # laboratory
          "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium",
          "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
          "Fibrinogen", "Platelets"]
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]      # descriptors known at ICU admission
LABEL_LEAD = 6          # SepsisLabel = 1 from t_sepsis - 6 h onward
WINDOW = 24             # outcome window after the decision (hours)
N_BOOT, SEED = 1000, 0
DESIGNS = {"t24": (24, [0, 6, 12, 24]), "t12": (12, [0, 6, 12, 24])}   # (t_d hours, horizons h)
AGGS = ["last", "min", "max", "mean", "count"]
TOP = 0.10
THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20]

_local = threading.local()


def _list_bucket(folder):
    """(file name, MD5 ETag) of every .psv file in one training folder of the open S3 mirror."""
    items, token = [], None
    while True:
        q = {"list-type": "2", "prefix": f"{S3_PREFIX}{folder}/"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"https://{S3_HOST}/?{urllib.parse.urlencode(q)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.iter(S3_NS + "Contents"):
            key = c.findtext(S3_NS + "Key")
            if key.endswith(".psv"):
                items.append((os.path.basename(key), c.findtext(S3_NS + "ETag").strip('"')))
        token = root.findtext(S3_NS + "NextContinuationToken")
        if not token:
            return items


def _fetch(path):
    """GET one object over this thread's persistent HTTPS connection (5 attempts with back-off)."""
    err = None
    for attempt in range(5):
        try:
            if getattr(_local, "conn", None) is None:
                _local.conn = http.client.HTTPSConnection(S3_HOST, timeout=60)
            _local.conn.request("GET", path)
            r = _local.conn.getresponse()
            body = r.read()
            if r.status == 200:
                return body
            err = f"HTTP {r.status}"
        except (OSError, http.client.HTTPException) as e:
            err = repr(e)
            _local.conn = None
        time.sleep(2 ** attempt)
    raise IOError(f"could not download {path}: {err}")


def download():
    """Fetch the missing .psv files of both hospital systems (no single archive is published)."""
    for folder, n_expected in SETS.values():
        d = os.path.join(DATA_DIR, folder)
        os.makedirs(d, exist_ok=True)
        have = {f for f in os.listdir(d) if f.endswith(".psv")}
        if len(have) >= n_expected:
            continue
        todo = [(f, etag) for f, etag in _list_bucket(folder) if f not in have]
        print(f"downloading {len(todo)} files of {folder} from s3://physionet-open "
              f"({N_CONN} connections)", flush=True)

        def get(item, d=d, folder=folder):
            name, etag = item
            body = _fetch(f"/{S3_PREFIX}{folder}/{name}")
            if "-" not in etag and hashlib.md5(body, usedforsecurity=False).hexdigest() != etag:
                raise IOError(f"checksum mismatch for {folder}/{name}")
            with open(os.path.join(d, name + ".part"), "wb") as fh:
                fh.write(body)
            os.replace(os.path.join(d, name + ".part"), os.path.join(d, name))

        with ThreadPoolExecutor(N_CONN) as ex:
            for i, _ in enumerate(ex.map(get, todo), 1):
                if i % 5000 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
        n = sum(f.endswith(".psv") for f in os.listdir(d))
        assert n == n_expected, f"{folder}: {n} files, expected {n_expected}"


def load_hourly():
    """All hourly rows of both hospital systems, with the stay id ('patient') and 'set' (A or B)."""
    frames = []
    for s, (folder, _) in SETS.items():
        d = os.path.join(DATA_DIR, folder)
        header, lines = None, []
        for name in sorted(f for f in os.listdir(d) if f.endswith(".psv")):
            with open(os.path.join(d, name)) as fh:
                rows = fh.read().splitlines()
            header = header or rows[0]
            assert rows[0] == header, f"unexpected header in {folder}/{name}"
            lines.extend(f"{name[1:-4]}|{r}\n" for r in rows[1:] if r)
        df = pd.read_csv(io.StringIO("".join(lines)), sep="|", header=None,
                         names=["patient"] + header.split("|"))
        assert list(df.columns[1:]) == SERIES + STATIC + ["ICULOS", "SepsisLabel"], header
        df["set"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def stay_table(hourly):
    """One row per stay: hospital system, descriptors, first and last ICU hour and t_sepsis.

    t_sepsis = first ICULOS with SepsisLabel = 1, plus 6 (NaN for stays that are never septic).
    first_hour, last_hour and t_sepsis define eligibility and the outcome; they are never features.
    labelled_from_first_row flags septic stays whose onset may precede the record (counted only).
    """
    g = hourly.groupby("patient")
    st = g[["set"] + STATIC].first()
    st["first_hour"] = g["ICULOS"].min()
    st["last_hour"] = g["ICULOS"].max()
    st["t_sepsis"] = hourly[hourly["SepsisLabel"] == 1].groupby("patient")["ICULOS"].min() + LABEL_LEAD
    st["labelled_from_first_row"] = g["SepsisLabel"].first() == 1
    return st


def cohort(stays, t_d):
    """Eligibility at decision hour t_d and the outcome, sepsis onset in (t_d, t_d + 24].

    Returns (eligible, y, in_icu, septic_by_decision) as Series on the stay index.
    """
    in_icu = stays["last_hour"] >= t_d                   # a row at ICULOS >= t_d
    septic_by_decision = stays["t_sepsis"] <= t_d        # NaN (never septic) compares False
    eligible = in_icu & ~septic_by_decision
    y = ((stays["t_sepsis"] > t_d) & (stays["t_sepsis"] <= t_d + WINDOW)).astype(int)
    return eligible, y, in_icu, septic_by_decision


def long_table(hourly):
    """Long table (patient, hour, param, value) of the non-missing hourly values of the 34 series."""
    parts = []
    for v in SERIES:
        m = hourly[v].notna().values
        parts.append(pd.DataFrame({"patient": hourly["patient"].values[m],
                                   "hour": hourly["ICULOS"].values[m],
                                   "param": v, "value": hourly[v].values[m]}))
    return pd.concat(parts, ignore_index=True)


def features(long, stays, cutoff_hour, params):
    """Per-stay aggregates of the hourly values recorded no later than ICU hour cutoff_hour."""
    d = long[long["hour"] <= cutoff_hour].sort_values(["patient", "hour"])
    g = d.groupby(["patient", "param"])["value"]
    agg = pd.concat({"last": g.last(), "min": g.min(), "max": g.max(), "mean": g.mean(),
                     "count": g.size()}, axis=1)
    wide = agg.unstack("param")
    wide.columns = [f"{p}_{a}" for a, p in wide.columns]
    cols = [f"{p}_{a}" for p in params for a in AGGS]
    wide = wide.reindex(index=stays.index, columns=cols)
    wide[[c for c in cols if c.endswith("_count")]] = wide[[c for c in cols if c.endswith("_count")]].fillna(0)
    return stays[STATIC].join(wide)


def fit_predict(Xtr, ytr, Xte):
    """models.fit_gbdt / predict_gbdt (as in audit_physionet2012.fit_predict); returns (predictions, iterations).

    Columns never observed in the training rows are dropped first: EtCO2 is never recorded in hospital
    system A, a tree cannot split on such a column, and scikit-learn's HistGradientBoosting rejects
    all-missing columns, so dropping them leaves the fitted model unchanged.
    """
    keep = Xtr.columns[Xtr.notna().any().values]
    clf = models.fit_gbdt(Xtr[keep], ytr, seed=SEED)
    return models.predict_gbdt(clf, Xte[keep]), int(clf.n_iter_)


def ppv_top(y, p, frac=TOP):
    """Precision among the top frac of stays ranked by p (as in scripts/decision_impact.py)."""
    k = max(1, int(np.ceil(frac * len(y))))
    order = np.argsort(-p, kind="mergesort")      # stable: ties keep input order
    return float(y[order[:k]].mean())


def decision_level(y, p_naive, p_dep, n_boot=N_BOOT, seed=SEED):
    """Precision among the top 10% and net benefit, naive versus deployable, paired bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y)
    ppv = {"naive": [], "dep": []}
    nb = {pt: {"naive": [], "dep": [], "all": []} for pt in THRESHOLDS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        ppv["naive"].append(ppv_top(yb, p_naive[idx]))
        ppv["dep"].append(ppv_top(yb, p_dep[idx]))
        for pt in THRESHOLDS:
            nb[pt]["naive"].append(DC.net_benefit(yb, p_naive[idx], pt))
            nb[pt]["dep"].append(DC.net_benefit(yb, p_dep[idx], pt))
            nb[pt]["all"].append(DC.treat_all(yb, pt))
    a, b = np.array(ppv["naive"]), np.array(ppv["dep"])
    out = {"ppv_top10": {"ppv_top10_with_leakage": round(ppv_top(y, p_naive), 4), "ci_with_leakage": E.ci(a),
                         "ppv_top10_deployable": round(ppv_top(y, p_dep), 4), "ci_deployable": E.ci(b),
                         "difference": round(ppv_top(y, p_naive) - ppv_top(y, p_dep), 4),
                         "difference_ci": E.ci(a - b), "base_rate": round(float(y.mean()), 4),
                         "n_test": int(n), "n_top": int(np.ceil(TOP * n))},
           "net_benefit": []}
    for pt in THRESHOLDS:
        d = {k: np.array(v) for k, v in nb[pt].items()}
        nb_dep, nb_naive = DC.net_benefit(y, p_dep, pt), DC.net_benefit(y, p_naive, pt)
        out["net_benefit"].append({
            "threshold": pt,
            "net_benefit_deployable": round(nb_dep, 4), "ci_deployable": E.ci(d["dep"]),
            "net_benefit_naive": round(nb_naive, 4), "ci_naive": E.ci(d["naive"]),
            "net_benefit_treat_all": round(DC.treat_all(y, pt), 4),
            "overstatement": round(nb_naive - nb_dep, 4), "overstatement_ci": E.ci(d["naive"] - d["dep"]),
            "overstatement_per_100_patients": round(100 * (nb_naive - nb_dep), 2),
            "deployable_minus_treat_all": round(nb_dep - DC.treat_all(y, pt), 4),
            "deployable_minus_treat_all_ci": E.ci(d["dep"] - d["all"]),
            "share_flagged_deployable": round(float(np.mean(p_dep >= pt)), 4),
            "share_flagged_naive": round(float(np.mean(p_naive >= pt)), 4)})
    return out


def cohort_counts(stays, t_d):
    """Stays, sequential exclusions, eligible stays, events and base rate per hospital system."""
    eligible, y, in_icu, septic = cohort(stays, t_d)
    out = {}
    for s, name in (("A", "train"), ("B", "test")):
        m = stays["set"] == s
        out[name] = {"hospital_system": s, "n_stays": int(m.sum()),
                     "excluded_not_in_icu_at_decision": int((m & ~in_icu).sum()),
                     "excluded_septic_by_decision": int((m & in_icu & septic).sum()),
                     "septic_by_decision_any_length": int((m & septic).sum()),
                     "n_eligible": int((m & eligible).sum()),
                     "n_events": int((m & eligible & (y == 1)).sum()),
                     "base_rate": round(float(y[m & eligible].mean()), 4),
                     "eligible_with_no_row_by_decision": int((m & eligible & (stays["first_hour"] > t_d)).sum()),
                     "events_labelled_from_first_row":
                         int((m & eligible & (y == 1) & stays["labelled_from_first_row"]).sum())}
    return out


def record_end_check(st, y, t_d):
    """How often eligible test records end before the naive cut-off, by outcome (descriptive)."""
    after = np.minimum(st["last_hour"], t_d + WINDOW) - t_d      # hourly rows after the decision
    ends = st["last_hour"] < t_d + WINDOW
    out = {}
    for lab, name in ((1, "events"), (0, "non_events")):
        m = y == lab
        out[name] = {"n": int(m.sum()),
                     "share_record_ends_before_naive_cutoff": round(float(ends[m].mean()), 4),
                     "median_hours_recorded_after_decision": float(np.median(after[m])),
                     "mean_hours_recorded_after_decision": round(float(after[m].mean()), 2)}
    return out


def data_checks(hourly, stays):
    """Documented cohort rules and how records end, per hospital system (descriptive; no model)."""
    n_rows = hourly.groupby("patient").size()
    contiguous = stays["last_hour"] - stays["first_hour"] + 1 == n_rows
    septic = stays["t_sepsis"].notna()
    after_onset = stays["last_hour"] - stays["t_sepsis"]
    varies = (hourly.groupby("patient")[STATIC].nunique() > 1).any(axis=1)
    both_units = stays["Unit1"].notna() & stays["Unit2"].notna()

    def qs(v):
        v = np.asarray(v, dtype=float)
        return {k: round(float(np.quantile(v, p)), 1) for k, p in
                (("min", 0), ("p05", .05), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95), ("max", 1))}

    out = {}
    for s in SETS:
        m = stays["set"] == s
        ms = m & septic
        vals = hourly.loc[hourly["set"] == s, SERIES]
        out[s] = {"n_stays": int(m.sum()), "n_hourly_rows": int(n_rows[m].sum()),
                  "n_hourly_rows_with_a_vital_or_lab_value": int(vals.notna().any(axis=1).sum()),
                  "n_vital_and_lab_entries": int(vals.notna().sum().sum()),
                  "variables_never_recorded": [v for v in SERIES if vals[v].notna().sum() == 0],
                  "n_septic": int(ms.sum()), "septic_share": round(float(ms.sum() / m.sum()), 4),
                  "min_rows_per_stay": int(n_rows[m].min()),
                  "records_starting_after_icu_hour_1": int((stays["first_hour"][m] > 1).sum()),
                  "max_first_iculos": int(stays["first_hour"][m].max()),
                  "share_contiguous_hours": round(float(contiguous[m].mean()), 4),
                  "stays_with_varying_descriptors": int(varies[m].sum()),
                  "unit_missing_share": round(float(stays["Unit1"][m].isna().mean()), 4),
                  "share_unit1_plus_unit2_equal_1_when_both_recorded":
                      round(float((stays["Unit1"] + stays["Unit2"] == 1)[m & both_units].mean()), 4),
                  "record_hours_non_septic": qs(stays["last_hour"][m & ~septic]),
                  "non_septic_records_longer_than_60h": int((stays["last_hour"][m & ~septic] > 60).sum()),
                  "t_sepsis_hours": qs(stays["t_sepsis"][ms]),
                  "record_hours_after_t_sepsis": qs(after_onset[ms]),
                  "septic_records_ending_1_to_3h_after_t_sepsis": int(after_onset[ms].between(1, 3).sum()),
                  "septic_records_ending_more_than_3h_after_t_sepsis": int((after_onset[ms] > 3).sum()),
                  "septic_stays_labelled_from_first_row": int((ms & stays["labelled_from_first_row"]).sum())}
    return out


def run_design(long, stays, params, t_d, horizons):
    eligible, y_all, _, _ = cohort(stays, t_d)
    st = stays[eligible]
    y = y_all[eligible].values
    is_tr = (st["set"] == "A").values
    Xs = {h: features(long, st, t_d + h, params) for h in horizons}
    ytr, yte = y[is_tr], y[~is_tr]
    preds, n_iter = {}, {}
    for h in horizons:
        preds[h], n_iter[h] = fit_predict(Xs[h][is_tr], ytr, Xs[h][~is_tr])
    h0, hN = horizons[0], horizons[-1]
    # per-source placebo: admit one feature group at the naive cut-off, keep the rest deployable
    groups = {"measurement_counts": [c for c in Xs[h0].columns if c.endswith("_count")],
              "measured_values": [c for c in Xs[h0].columns
                                  if any(c.endswith("_" + a) for a in ("last", "min", "max", "mean"))]}
    for gname, cols in groups.items():
        Xg = Xs[h0].copy()
        Xg[cols] = Xs[hN][cols]
        preds[f"placebo:{gname}"], _ = fit_predict(Xg[is_tr], ytr, Xg[~is_tr])
    # per-variable placebo (exploratory): one variable's features at the naive cut-off
    for p in params:
        cols = [f"{p}_{a}" for a in AGGS]
        Xg = Xs[h0].copy()
        Xg[cols] = Xs[hN][cols]
        preds[f"var:{p}"], _ = fit_predict(Xg[is_tr], ytr, Xg[~is_tr])
    boot = E.paired_boot(yte, preds)
    base = float(yte.mean())
    curve = []
    for h in horizons:
        curve.append({"h_hours": h, "data_up_to_hours": t_d + h,
                      "auprc": round(M.auprc(yte, preds[h]), 4), "auprc_ci": E.ci(boot[h]["auprc"]),
                      "auroc": round(M.auroc(yte, preds[h]), 4), "auroc_ci": E.ci(boot[h]["auroc"]),
                      "lift": round(M.auprc(yte, preds[h]) - base, 4)})

    def contrast(k):
        return {"auprc": round(M.auprc(yte, preds[k]) - M.auprc(yte, preds[h0]), 4),
                "auprc_ci": E.ci(boot[k]["auprc"] - boot[h0]["auprc"]),
                "auprc_p_gt_0": round(float(np.mean(boot[k]["auprc"] - boot[h0]["auprc"] > 0)), 3),
                "auroc": round(M.auroc(yte, preds[k]) - M.auroc(yte, preds[h0]), 4),
                "auroc_ci": E.ci(boot[k]["auroc"] - boot[h0]["auroc"])}

    return {"decision_time_hours": t_d, "horizons_hours": horizons,
            "outcome_window_hours": [t_d, t_d + WINDOW],
            "cohort": cohort_counts(stays, t_d),
            "n_train": int(is_tr.sum()), "n_test": int((~is_tr).sum()),
            "n_train_events": int(ytr.sum()), "n_test_events": int(yte.sum()),
            "train_base_rate": round(float(ytr.mean()), 4), "test_base_rate": round(base, 4),
            "n_features": int(Xs[h0].shape[1]),
            "features_never_observed_in_training_dropped": {
                h: [c for c in Xs[h].columns if Xs[h][is_tr][c].isna().all()] for h in horizons},
            "boosting_iterations_used": n_iter,
            "curve": curve, "LAP": contrast(hN),
            "per_source_placebo": {g: contrast(f"placebo:{g}") for g in groups},
            "per_variable_placebo_exploratory": dict(sorted(
                ((p, contrast(f"var:{p}")) for p in params), key=lambda kv: -kv[1]["auprc"])),
            "decision_level_impact": decision_level(yte, preds[hN], preds[h0]),
            "record_end_check": record_end_check(st[~is_tr], yte, t_d)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/physionet2019_sepsis_audit.json")
    args = ap.parse_args()
    t0 = time.time()
    download()
    hourly = load_hourly()
    stays = stay_table(hourly)
    max_cutoff = max(t_d + max(hs) for t_d, hs in DESIGNS.values())
    long = long_table(hourly[hourly["ICULOS"] <= max_cutoff])
    checks = data_checks(hourly, stays)
    del hourly
    params = SERIES
    print(f"stays={len(stays)} (A {checks['A']['n_stays']}, B {checks['B']['n_stays']}) septic "
          f"A {checks['A']['n_septic']} B {checks['B']['n_septic']}; values up to hour {max_cutoff}: "
          f"{len(long)}", flush=True)
    out = {"dataset": "PhysioNet/CinC Challenge 2019, Early Prediction of Sepsis from Clinical Data "
                      "(physionet.org/content/challenge-2019/1.0.0; CC BY 4.0)",
           "script": "scripts/audit_physionet2019_sepsis.py",
           "outcome": "Sepsis-3 onset within 24 h after the decision: t_d < t_sepsis <= t_d + 24, "
                      "t_sepsis = first ICULOS with SepsisLabel = 1, plus 6",
           "split": "train = hospital system A (training_setA); test = hospital system B (training_setB), "
                    "external and fixed",
           "design": {"pre_specified": "fixed before any model was fitted; not changed after seeing results",
                      "unit": "ICU stay",
                      "decision_times_hours": {"primary": 24, "sensitivity": 12},
                      "hour_alignment": "the row with ICULOS = k summarises hour k after ICU admission; rows "
                                        "with ICULOS <= c are recorded by hour c (by the decision when c = t_d)",
                      "eligibility": "a row at ICULOS >= t_d (still in the ICU at the decision) and "
                                     "t_sepsis > t_d (never-septic stays eligible); exclusions counted in "
                                     "this order",
                      "outcome": "y = 1 if t_d < t_sepsis <= t_d + 24, else 0",
                      "horizons_hours": "cut-off t_d + h, h in {0, 6, 12, 24}; h = 0 deployable, "
                                        "h = 24 naive",
                      "LAP": "AUPRC(h = 24) - AUPRC(h = 0) on the same test stays",
                      "per_source_placebo": "measured values (last, min, max, mean) at h = 24 with counts at "
                                            "h = 0, and counts at h = 24 with values at h = 0",
                      "per_variable_placebo": "exploratory: one variable's five features at h = 24",
                      "decision_level": f"precision among the top {int(TOP * 100)}% of test stays; net benefit "
                                        f"at thresholds {THRESHOLDS} (TP/n - FP/n * pt/(1-pt))"},
           "label_documentation": {
               "sepsislabel": "1 for t >= t_sepsis - 6 in septic patients, 0 otherwise (shifted 6 h ahead)",
               "t_sepsis": "min(t_suspicion, t_SOFA) if t_suspicion - 24 <= t_SOFA <= t_suspicion + 12 "
                           "(Sepsis-3)",
               "cohort_rules": "stays with fewer than 8 hourly windows and patients with t_sepsis < 4 h "
                               "after ICU admission not included; records truncated at ICU discharge and "
                               "after 2 weeks (Reyna et al., Crit Care Med 2020, Section 2.2)",
               "hourly_bins": "several values within an hour summarised by their median",
               "observed_in_files_not_in_documentation":
                   "every septic record ends at most 3 h after t_sepsis (the end of the challenge scoring "
                   "window); non-septic records rarely extend past ICU hour 60 (counts under data_checks); "
                   "measurement counts admitted after the decision therefore partly encode the outcome"},
           "n_stays_A": checks["A"]["n_stays"], "n_stays_B": checks["B"]["n_stays"],
           "n_variables": len(params), "variables": params,
           "features": "per variable: last, min, max, mean and number of hours with a value recorded by the "
                       "cut-off; Age, Gender, Unit1, Unit2 and HospAdmTime as recorded (missing left to the "
                       "learner); ICULOS, record length and SepsisLabel are not features",
           "learner": "models.fit_gbdt (HistGradientBoosting, max_depth 3, lr 0.06, 300 iter, l2 1.0, seed 0); "
                      "scikit-learn's early_stopping='auto' is active because n_train > 10,000 (iterations used "
                      "reported per design); columns never observed in training are dropped before fitting",
           "bootstrap": f"patient-level paired percentile bootstrap, {N_BOOT} resamples, seed {SEED}",
           "data_checks": checks}
    for key, (t_d, horizons) in DESIGNS.items():
        out[key] = r = run_design(long, stays, params, t_d, horizons)
        print(f"{key}: n_train {r['n_train']} ({r['n_train_events']} events) n_test {r['n_test']} "
              f"({r['n_test_events']} events, base rate {r['test_base_rate']}); deployable AUPRC "
              f"{r['curve'][0]['auprc']} naive {r['curve'][-1]['auprc']} LAP {r['LAP']['auprc']:+.4f} "
              f"{r['LAP']['auprc_ci']}", flush=True)
    out["runtime_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
