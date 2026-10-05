"""Numerical sanity checks for Appendix A ("Formal properties") of the manuscript.

Part 1  Proposition 1. Population AP of a fixed binormal ROC curve, computed by numerical
        integration of AP(pi) = int_0^1 pi r / {pi r + (1 - pi) R^-(r)} dr, increases strictly with
        the prevalence pi and agrees with large-sample average precision; AUROC does not move; an
        uninformative score has AP = pi; and the AP difference between two scores whose ROC
        curves cross changes sign with pi.
Part 2  Proposition 4. Toy logistic model with a deployable feature X0 and a post-decision feature
        X1. Large-sample AP and AUROC of the Bayes deployable score (a strictly increasing
        function of P(Y=1|X0)), the naive score, and the naive score with X1 replaced by an
        independent draw from its marginal (population joint permutation). LAP <= PI is checked
        for several dependence structures and naive models, then three ways the bound can fail
        are shown: a suboptimal deployable learner, small test sets with row permutation, and the
        step convention of scikit-learn's average precision for a tied deployable score. A fitted
        gradient-boosted version is included.
Part 3  Proposition 5. Exact duplicate post-decision features: permutation importance of the two
        copies in a fitted gradient-boosted model versus drop-and-refit.

Run:  python scripts/check_propositions.py
All randomness is seeded; runtime is about a minute.
"""
import numpy as np
from scipy import integrate
from scipy.special import expit
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

N_LARGE = 2_000_000
rng = np.random.default_rng(20260930)


# ----------------------------------------------------------------------------- helpers
def ap_step(y, s):
    """Average precision with scikit-learn's convention: precision is evaluated once per
    distinct score value, after the whole tie group (checked against sklearn below)."""
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    tp, fp = np.cumsum(y_sorted), np.cumsum(1 - y_sorted)
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s) - 1]
    tp_g, fp_g = tp[last], fp[last]
    return float(np.sum(np.diff(np.r_[0, tp_g]) * tp_g / (tp_g + fp_g)) / tp[-1])


def ap_tie_broken(y, s, gen):
    """AP after breaking ties uniformly at random (order by s, then by an independent uniform)."""
    order = np.lexsort((gen.random(len(s)), s))
    r = np.empty(len(s))
    r[order] = np.arange(len(s))
    return ap_step(y, r)


def pop_ap(pi, rinv):
    """Population AP of Proposition 1 / equation (1): int_0^1 pi r / (pi r + (1-pi) R^-(r)) dr."""
    f = lambda r: pi * r / (pi * r + (1 - pi) * rinv(r)) if r > 0 else 1.0
    val, _ = integrate.quad(f, 0, 1, limit=400, points=[1e-6, 1e-4, 1e-2, 0.3])
    return val


def draw_labels(n, pi, gen):
    n1 = int(round(pi * n))
    y = np.zeros(n, dtype=int)
    y[:n1] = 1
    return y


print("=" * 88)
print("Check of the step-AP helper against scikit-learn (tied and untied scores)")
yy = rng.integers(0, 2, 5000)
for s in (rng.normal(size=5000), rng.integers(0, 7, 5000).astype(float)):
    print(f"  ap_step = {ap_step(yy, s):.12f}   sklearn = {average_precision_score(yy, s):.12f}")

# ============================================================================= Part 1
print("=" * 88)
print("PART 1  Proposition 1: AP of a fixed ROC curve increases with prevalence")
a_bin = 1.0  # binormal ROC R(alpha) = Phi(a + Phi^-1(alpha)); AUROC = Phi(a / sqrt 2)
rinv_bin = lambda r: norm.cdf(norm.ppf(r) - a_bin)
print(f"Binormal ROC, a = {a_bin}, b = 1; population AUROC = {norm.cdf(a_bin / np.sqrt(2)):.4f}")
print(f"{'pi':>6} {'AP (integral)':>14} {'AP (n=2e6)':>11} {'AUROC (n=2e6)':>14} {'AP uninform.':>13}")
pis = [0.01, 0.05, 0.10, 0.173, 0.20, 0.32, 0.43, 0.50, 0.70, 0.90]
aps = []
for pi in pis:
    ap_int = pop_ap(pi, rinv_bin)
    aps.append(ap_int)
    y = draw_labels(N_LARGE, pi, rng)
    s = rng.normal(size=N_LARGE) + a_bin * y  # positives N(a,1), negatives N(0,1)
    ap_unif = pop_ap(pi, lambda r: r)
    print(f"{pi:6.3f} {ap_int:14.4f} {ap_step(y, s):11.4f} {roc_auc_score(y, s):14.4f} {ap_unif:13.4f}")
print(f"AP strictly increasing in pi: {bool(np.all(np.diff(aps) > 0))}; "
      f"uninformative score has AP = pi (last column)")

print("\nLAP between two scores whose ROC curves cross can change sign with pi")
c_top = 0.3  # score A: a fraction c of positives ranked above all negatives, the rest at random
rinv_A = lambda r: max(0.0, (r - c_top) / (1 - c_top))
a_B = 2.0    # score B: binormal a = 2, b = 1
rinv_B = lambda r: norm.cdf(norm.ppf(r) - a_B)
print(f"A: R_A(alpha) = {c_top} + {1 - c_top}*alpha (AUROC {c_top + (1 - c_top) / 2:.3f});  "
      f"B: binormal a = {a_B} (AUROC {norm.cdf(a_B / np.sqrt(2)):.3f})")
grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
diff = np.array([pop_ap(p, rinv_A) - pop_ap(p, rinv_B) for p in grid])
cross = grid[np.nonzero(np.diff(np.sign(diff)))[0]]
for p in (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.90):
    print(f"  pi = {p:4.2f}: AP_A = {pop_ap(p, rinv_A):.4f}  AP_B = {pop_ap(p, rinv_B):.4f}  "
          f"AP_A - AP_B = {pop_ap(p, rinv_A) - pop_ap(p, rinv_B):+.4f}")
print(f"  sign of AP_A - AP_B changes between pi = {cross[0]:.2f} and {cross[0] + 0.01:.2f}")
for p in (0.02, 0.50):
    y = draw_labels(N_LARGE, p, rng)
    top = rng.random(N_LARGE) < c_top
    sA = np.where((y == 1) & top, 2.0 + rng.random(N_LARGE), rng.random(N_LARGE))
    sB = rng.normal(size=N_LARGE) + a_B * y
    print(f"  large-sample check, pi = {p:.2f}: AP_A = {ap_step(y, sA):.4f}  AP_B = {ap_step(y, sB):.4f}")

# ============================================================================= Part 2
print("=" * 88)
print("PART 2  Proposition 4: joint permutation importance bounds LAP from above")
print("Model: X0 ~ N(0,1); X1 = rho X0 + sqrt(1-rho^2) Z; Y ~ Bernoulli(expit(b0 + b1 X0 + b2 X1)).")
print("P(Y=1|X0) = E[expit(b0 + (b1 + b2 rho) X0 + b2 sqrt(1-rho^2) Z)] is strictly increasing in X0")
print("when b1 + b2 rho > 0, so X0 itself is a Bayes deployable score (same ROC curve).")

gh_x, gh_w = np.polynomial.hermite.hermgauss(80)


def eta0(x0, b0, b1, b2, rho):
    """P(Y=1 | X0 = x0) by Gauss-Hermite quadrature (used only to confirm monotonicity)."""
    z = np.sqrt(2) * gh_x
    return (expit(b0 + (b1 + b2 * rho) * x0[:, None] + b2 * np.sqrt(1 - rho ** 2) * z[None, :])
            @ gh_w) / np.sqrt(np.pi)


def simulate(n, b0, b1, b2, rho, gen):
    x0 = gen.normal(size=n)
    x1 = rho * x0 + np.sqrt(1 - rho ** 2) * gen.normal(size=n)
    y = (gen.random(n) < expit(b0 + b1 * x0 + b2 * x1)).astype(int)
    x1_perm = gen.normal(size=n)  # independent draw from the marginal N(0,1) of X1
    return x0, x1, x1_perm, y


settings = [
    # label, b0, b1, b2, rho, naive model f(x0, x1)
    ("X1 independent of X0", -1.6, 1.0, 1.5, 0.0, lambda a, b: a + 1.5 * b),
    ("X1 correlated with X0 (rho=0.6)", -1.6, 1.0, 1.5, 0.6, lambda a, b: a + 1.5 * b),
    ("X1 negatively correlated (rho=-0.6)", -1.6, 1.0, 1.5, -0.6, lambda a, b: a + 1.5 * b),
    ("strong leak, CTO-like (b2=4)", -2.5, 0.3, 4.0, 0.3, lambda a, b: 0.3 * a + 4.0 * b),
    ("arbitrary naive model f=x1+0.2x0^3", -1.6, 1.0, 1.5, 0.6, lambda a, b: b + 0.2 * a ** 3),
    ("no leak: b2=0 (X1 pure noise)", -1.6, 1.0, 0.0, 0.0, lambda a, b: a + 0.0 * b),
]
hdr = f"{'setting':38} {'pi':>5} {'M(S0)':>7} {'M(Sinf)':>7} {'M(Sprm)':>7} {'LAP':>7} {'PI':>7} {'PI-LAP':>7}"
for metric in ("AP", "AUROC"):
    print(f"\n[{metric}] population approximated with n = {N_LARGE:,} test units")
    print(hdr)
    for k, (lab, b0, b1, b2, rho, f) in enumerate(settings):
        gen = np.random.default_rng(100 + k)  # same data for the AP and AUROC tables
        grid_x = np.linspace(-4, 4, 201)
        assert np.all(np.diff(eta0(grid_x, b0, b1, b2, rho)) >= 0), "P(Y=1|X0) not monotone"
        x0, x1, x1p, y = simulate(N_LARGE, b0, b1, b2, rho, gen)
        m = ap_step if metric == "AP" else roc_auc_score
        s0, sinf, sprm = x0, f(x0, x1), f(x0, x1p)
        v0, vinf, vprm = m(y, s0), m(y, sinf), m(y, sprm)
        lap, pi_ = vinf - v0, vinf - vprm
        print(f"{lab:38} {y.mean():5.3f} {v0:7.4f} {vinf:7.4f} {vprm:7.4f} {lap:7.4f} {pi_:7.4f} "
              f"{pi_ - lap:+7.4f}")
print("PI - LAP = M(S0) - M(Sprm) >= 0 in every setting; it is > 0 even when X1 is independent of X0.")

print("\nFailure 1: a suboptimal deployable learner (deployable score X0 + 2 N(0,1) noise)")
gen = np.random.default_rng(1)
x0, x1, x1p, y = simulate(N_LARGE, -1.6, 1.0, 1.5, 0.0, gen)
s0_bad = x0 + 2.0 * gen.normal(size=N_LARGE)
sinf, sprm = x0 + 1.5 * x1, x0 + 1.5 * x1p
for nm, m in (("AP", ap_step), ("AUROC", roc_auc_score)):
    lap, pi_ = m(y, sinf) - m(y, s0_bad), m(y, sinf) - m(y, sprm)
    print(f"  {nm:5}: LAP = {lap:.4f}  PI = {pi_:.4f}  PI - LAP = {pi_ - lap:+.4f}  (bound fails)")

print("\nFailure 2: finite test sets with row permutation (as in compare_methods.py, 20 permutations)")
b0, b1, b2, rho = -1.6, 1.0, 0.35, 0.0   # weak leak, so the population gap is small
gen = np.random.default_rng(2)
x0, x1, x1p, y = simulate(N_LARGE, b0, b1, b2, rho, gen)
gap_pop = ap_step(y, x0) - ap_step(y, x0 + (b2 / b1) * x1p)
print(f"  population (n=2e6): LAP = {ap_step(y, x0 + (b2 / b1) * x1) - ap_step(y, x0):.4f}, "
      f"PI - LAP = {gap_pop:+.4f}")
for n_test in (200, 1000, 5000):
    fails, gaps = 0, []
    reps = 1000
    for _ in range(reps):
        x0, x1, _, y = simulate(n_test, b0, b1, b2, rho, gen)
        if y.sum() == 0:
            continue
        sinf = x0 + (b2 / b1) * x1
        m0, minf = ap_step(y, x0), ap_step(y, sinf)
        mprm = np.mean([ap_step(y, x0 + (b2 / b1) * x1[gen.permutation(n_test)]) for _ in range(20)])
        gaps.append(m0 - mprm)
        fails += (minf - m0) > (minf - mprm)
    print(f"  n_test = {n_test:5d}: mean(PI - LAP) = {np.mean(gaps):+.4f}; "
          f"empirical LAP > empirical PI in {100 * fails / reps:.1f}% of {reps} test sets")

print("\nFailure 3: ties and scikit-learn's step convention")
print("  X0 in {0,1,2}; X1 is pure noise independent of (Y, X0); the naive model uses X1 only to")
print("  break the ties of the Bayes deployable score: f(x0, x1) = eta0(x0) + 1e-6 x1.")
p_lvl = np.array([0.2, 0.3, 0.5])      # P(X0 = 2, 1, 0), listed from the highest risk down
eta_lvl = np.array([0.6, 0.3, 0.1])    # P(Y=1 | X0 = level)
pi_t = float(p_lvl @ eta_lvl)
A = np.cumsum(p_lvl * eta_lvl)         # cumulative P(Y=1, X0 in top levels)
B = np.cumsum(p_lvl)                   # cumulative P(X0 in top levels)
w = p_lvl * eta_lvl / pi_t             # P(X0 = level | Y = 1)
step_exact = float(w @ (A / B))
A0, B0 = np.r_[0, A[:-1]], np.r_[0, B[:-1]]
seg = [integrate.quad(lambda t, k=k: (A0[k] + t * p_lvl[k] * eta_lvl[k]) / (B0[k] + t * p_lvl[k]),
                      0, 1)[0] for k in range(3)]
tie_exact = float(w @ np.array(seg))
print(f"  exact population values: base rate = {pi_t:.3f}; step AP of eta0 = {step_exact:.4f}; "
      f"tie-broken AP of eta0 = {tie_exact:.4f}")
gen = np.random.default_rng(3)
lvl = gen.choice(3, size=N_LARGE, p=p_lvl)
y = (gen.random(N_LARGE) < eta_lvl[lvl]).astype(int)
s0 = eta_lvl[lvl]
x1, x1p = gen.normal(size=N_LARGE), gen.normal(size=N_LARGE)
sinf, sprm = s0 + 1e-6 * x1, s0 + 1e-6 * x1p
st0, stinf, stprm = ap_step(y, s0), ap_step(y, sinf), ap_step(y, sprm)
tb0 = ap_tie_broken(y, s0, gen)
print(f"  n = 2e6, step convention:  LAP = {stinf - st0:+.4f}   PI = {stinf - stprm:+.4f}   "
      f"(LAP > PI: bound fails)")
print(f"  n = 2e6, ties broken at random: LAP = {stinf - tb0:+.4f}   PI = {stinf - stprm:+.4f}")
print(f"  AUROC (trapezoidal, ties count 1/2): LAP = {roc_auc_score(y, sinf) - roc_auc_score(y, s0):+.4f}")

print("\nFitted learners (gradient-boosted trees as in the paper), rho = 0.6 setting")
gen = np.random.default_rng(4)
b0, b1, b2, rho = -1.6, 1.0, 1.5, 0.6
x0tr, x1tr, _, ytr = simulate(20_000, b0, b1, b2, rho, gen)
x0te, x1te, _, yte = simulate(200_000, b0, b1, b2, rho, gen)
hgb = dict(max_depth=3, learning_rate=0.06, max_iter=300, l2_regularization=1.0, random_state=0)
dep = HistGradientBoostingClassifier(**hgb).fit(x0tr[:, None], ytr)
nav = HistGradientBoostingClassifier(**hgb).fit(np.c_[x0tr, x1tr], ytr)
p0 = dep.predict_proba(x0te[:, None])[:, 1]
pinf = nav.predict_proba(np.c_[x0te, x1te])[:, 1]
pprm = [nav.predict_proba(np.c_[x0te, x1te[gen.permutation(len(yte))]])[:, 1] for _ in range(5)]
print(f"  distinct deployable scores: {len(np.unique(p0))} on {len(yte):,} test units")
for nm, m in (("AP step", ap_step), ("AP tie-broken", lambda yy_, ss: ap_tie_broken(yy_, ss, gen)),
              ("AUROC", roc_auc_score)):
    v0, vinf = m(yte, p0), m(yte, pinf)
    vprm = float(np.mean([m(yte, q) for q in pprm]))
    print(f"  {nm:14}: M(p0) = {v0:.4f}  M(pinf) = {vinf:.4f}  M(perm) = {vprm:.4f}  "
          f"LAP = {vinf - v0:.4f}  PI = {vinf - vprm:.4f}  PI - LAP = {v0 - vprm:+.4f}")
print(f"  Bayes deployable score on the same test set: AP = {ap_step(yte, x0te):.4f}, "
      f"AUROC = {roc_auc_score(yte, x0te):.4f}")

# ============================================================================= Part 3
print("=" * 88)
print("PART 3  Proposition 5: redundant post-decision features")
print("X0 deployable; Xa post-decision; Xb = Xa exactly (a redundant copy); fitted gradient boosting.")
gen = np.random.default_rng(5)


def sim3(n):
    x0 = gen.normal(size=n)
    xa = 0.3 * x0 + np.sqrt(1 - 0.09) * gen.normal(size=n)
    y = (gen.random(n) < expit(-1.6 + 0.8 * x0 + 2.0 * xa)).astype(int)
    return np.c_[x0, xa, xa.copy()], y


Xtr, ytr = sim3(20_000)
Xte, yte = sim3(200_000)
full = HistGradientBoostingClassifier(**hgb).fit(Xtr, ytr)
p_full = full.predict_proba(Xte)[:, 1]
base = ap_step(yte, p_full)


def perm_drop(cols, B=10):
    out = []
    for _ in range(B):
        Xp = Xte.copy()
        Xp[:, cols] = Xte[gen.permutation(len(yte))][:, cols]
        out.append(base - ap_step(yte, full.predict_proba(Xp)[:, 1]))
    return float(np.mean(out))


p_dep = HistGradientBoostingClassifier(**hgb).fit(Xtr[:, [0]], ytr).predict_proba(Xte[:, [0]])[:, 1]
p_no_a = HistGradientBoostingClassifier(**hgb).fit(Xtr[:, [0, 2]], ytr).predict_proba(Xte[:, [0, 2]])[:, 1]
lap_total = base - ap_step(yte, p_dep)
lap_no_a = ap_step(yte, p_no_a) - ap_step(yte, p_dep)
print(f"  naive AP = {base:.4f}; deployable AP = {ap_step(yte, p_dep):.4f}; LAP = {lap_total:.4f}")
print(f"  permutation importance: Xa alone = {perm_drop([1]):.4f}; Xb alone = {perm_drop([2]):.4f}; "
      f"Xa and Xb jointly = {perm_drop([1, 2]):.4f}")
print(f"  drop Xa and refit: AP = {ap_step(yte, p_no_a):.4f}; LAP = {lap_no_a:.4f} "
      f"({100 * lap_no_a / lap_total:.1f}% of the total); drop-and-refit importance of Xa = "
      f"{base - ap_step(yte, p_no_a):+.4f}")
print("=" * 88)
