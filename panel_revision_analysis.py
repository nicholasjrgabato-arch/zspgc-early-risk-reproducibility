"""Panel-revision analyses for the ZSPGC early academic-deficiency study.
Reuses the original run_ml_experiment.py settings (CatBoost depth 4, 150 it., lr 0.05, Balanced; StratifiedGroupKFold k=5)."""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, recall_score, f1_score, precision_score
from catboost import CatBoostClassifier
import shap
warnings.filterwarnings("ignore")

import os
BASE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(BASE, "cleaned_dataset") + os.sep
df = pd.read_csv(D + "student_risk_ml_dataset.csv")
tr = df[df.split_type == "train_eval"].reset_index(drop=True)
te = df[df.split_type == "prospective_test"].reset_index(drop=True)
y = tr.at_risk.astype(int).values
g = tr.student_pid.values

acad = ['program_is_bsis', 'year_level', 'enrolled_units', 'course_count', 'has_lab_courses', 'lab_courses_count', 'is_underload', 'is_overload']
demo = ['age', 'is_female']
geo = ['distance_km', 'locality_is_campus_area', 'locality_is_distant']
socio = ['is_4ps_beneficiary', 'income_bracket', 'has_philsys_id', 'is_first_gen_college', 'is_dependent', 'has_disability', 'is_indigenous']
long_ = ['is_returning_student', 'prior_term_gpa', 'has_prior_gpa_recorded', 'prior_deficiencies', 'prior_at_risk']
F25 = ['program_is_bsis', 'year_level', 'age', 'is_female'] + geo + socio + ['enrolled_units', 'course_count', 'has_lab_courses', 'lab_courses_count', 'is_underload', 'is_overload'] + long_
F20 = [c for c in F25 if c not in long_]
CFG = {"M1": acad, "M2": acad + demo, "M3": acad + demo + geo, "M4": acad + socio, "M5": F20}

def cb(seed=42):
    return CatBoostClassifier(iterations=150, depth=4, learning_rate=0.05, auto_class_weights='Balanced', verbose=False, random_seed=seed, thread_count=4)

def metrics(yv, p):
    pr = (p >= 0.5).astype(int)
    return dict(roc=roc_auc_score(yv, p), pr=average_precision_score(yv, p), brier=brier_score_loss(yv, p),
                rec=recall_score(yv, pr, zero_division=0), f1=f1_score(yv, pr, zero_division=0), prec=precision_score(yv, pr, zero_division=0))

def cv_run(cols, split_seed, model_seed=42, Xdf=tr, yy=y, gg=g, frac=1.0, sub_seed=0):
    sk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    oof = np.zeros(len(yy)); folds = []
    for k, (a, b) in enumerate(sk.split(Xdf[cols], yy, groups=gg)):
        if frac < 1.0:
            rng = np.random.RandomState(sub_seed * 100 + k)
            pos = a[yy[a] == 1]; neg = a[yy[a] == 0]
            a = np.concatenate([rng.choice(pos, max(5, int(len(pos) * frac)), replace=False), rng.choice(neg, max(10, int(len(neg) * frac)), replace=False)])
        m = cb(model_seed); m.fit(Xdf[cols].iloc[a], yy[a])
        p = m.predict_proba(Xdf[cols].iloc[b])[:, 1]; oof[b] = p
        folds.append(metrics(yy[b], p))
    return oof, folds

R = {}

# ---- 0. Reproduce original M1/M5 (seed 42) and check 25 vs 20 features ----
oof25, f25 = cv_run(F25, 42); oof20, f20 = cv_run(F20, 42); oofM1, fM1 = cv_run(acad, 42)
mean = lambda fl, k: float(np.mean([f[k] for f in fl])); sd = lambda fl, k: float(np.std([f[k] for f in fl]))
R["reproduce"] = {name: {k: [mean(fl, k), sd(fl, k)] for k in ["roc", "pr", "brier", "rec", "f1", "prec"]} for name, fl in [("M5_25feat", f25), ("M5_20feat", f20), ("M1", fM1)]}
R["max_abs_diff_25_vs_20_oof"] = float(np.max(np.abs(oof25 - oof20)))
m25 = cb(); m25.fit(tr[F25], y); m20 = cb(); m20.fit(tr[F20], y)
p25 = m25.predict_proba(te[F25])[:, 1]; p20 = m20.predict_proba(te[F20])[:, 1]
R["prospective_25_vs_20_maxdiff"] = float(np.max(np.abs(p25 - p20)))
R["long_feature_importance_in_25"] = {c: float(v) for c, v in zip(F25, m25.get_feature_importance()) if c in long_}
tiers = pd.cut(p20, [-1, 0.35, 0.60, 2], right=False, labels=["Low", "Moderate", "High"]).value_counts().to_dict()
R["prospective_tiers_20feat"] = {str(k): int(v) for k, v in tiers.items()}
print("reproduce done", R["reproduce"]["M5_20feat"]["pr"], R["max_abs_diff_25_vs_20_oof"], R["prospective_25_vs_20_maxdiff"])

# ---- 1. Repeated grouped CV (20 x 5) with fold-aligned paired differences, M1..M5 ----
REP = 20
rep = {n: [] for n in CFG}; rep_oof = {n: [] for n in CFG}
for s in range(REP):
    for n, cols in CFG.items():
        o, fl = cv_run(cols, split_seed=1000 + s)
        rep[n].append(fl); rep_oof[n].append(o)
    print("rep", s)
def summary(n, k):
    vals = np.array([[f[k] for f in fl] for fl in rep[n]])   # REP x 5
    return dict(mean=float(vals.mean()), sd_fold=float(vals.std()), ci_low=float(np.percentile(vals.mean(1), 2.5)), ci_high=float(np.percentile(vals.mean(1), 97.5)))
R["repeated"] = {n: {k: summary(n, k) for k in ["roc", "pr", "brier", "rec", "f1", "prec"]} for n in CFG}

def nb_test(d, n_train, n_test):  # Nadeau & Bengio corrected resampled t-test; d = all fold differences (REP*5)
    J = len(d); m = d.mean(); v = d.var(ddof=1)
    t = m / np.sqrt((1 / J + n_test / n_train) * v); p = 2 * stats.t.sf(abs(t), J - 1)
    half = stats.t.ppf(0.975, J - 1) * np.sqrt((1 / J + n_test / n_train) * v)
    return dict(mean_diff=float(m), ci_low=float(m - half), ci_high=float(m + half), t=float(t), p=float(p))
paired = {}
for n in ["M2", "M3", "M4", "M5"]:
    paired[n] = {}
    for k in ["roc", "pr", "brier", "rec", "f1"]:
        d = np.array([[f[k] for f in fl] for fl in rep[n]]) - np.array([[f[k] for f in fl] for fl in rep["M1"]])
        d = d.ravel()
        res = nb_test(d, 304, 76); res["share_folds_better"] = float(np.mean(d < 0) if k == "brier" else np.mean(d > 0))
        paired[n][k] = res
R["paired_vs_M1"] = paired

# ---- 2. Pooled-OOF bootstrap CI on the difference (seed-42 split) + DeLong ----
B = 2000; rng = np.random.RandomState(7); bd = {"pr": [], "roc": [], "brier": []}
for _ in range(B):
    i = rng.randint(0, len(y), len(y))
    if y[i].sum() < 2: continue
    bd["pr"].append(average_precision_score(y[i], oof20[i]) - average_precision_score(y[i], oofM1[i]))
    bd["roc"].append(roc_auc_score(y[i], oof20[i]) - roc_auc_score(y[i], oofM1[i]))
    bd["brier"].append(brier_score_loss(y[i], oof20[i]) - brier_score_loss(y[i], oofM1[i]))
R["bootstrap_diff_M5_minus_M1"] = {k: dict(point=float(np.mean(v)), ci_low=float(np.percentile(v, 2.5)), ci_high=float(np.percentile(v, 97.5)), p_le0=float(np.mean(np.array(v) <= 0))) for k, v in bd.items()}

def delong(y_true, p1, p2):
    pos = y_true == 1; neg = ~pos; m, n = pos.sum(), neg.sum()
    def comps(p):
        X, Y = p[pos], p[neg]
        psi = (X[:, None] > Y[None, :]).astype(float) + 0.5 * (X[:, None] == Y[None, :])
        return psi.mean(1), psi.mean(0), psi.mean()
    v10a, v01a, a1 = comps(p1); v10b, v01b, a2 = comps(p2)
    s10 = np.cov(np.vstack([v10a, v10b])); s01 = np.cov(np.vstack([v01a, v01b]))
    S = s10 / m + s01 / n; var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var); return dict(auc1=float(a1), auc2=float(a2), z=float(z), p=float(2 * stats.norm.sf(abs(z))))
R["delong_M5_vs_M1_pooled_oof"] = delong(y, oof20, oofM1)

# ---- 3. Utility at fixed advising capacity (repeated OOF), paired ----
cap = {}
for frac in [0.10, 0.15, 0.20, 0.25, 0.30]:
    k = int(round(frac * len(y))); rows = {}
    for n in ["M1", "M5"]:
        hits = []
        for o in rep_oof[n]:
            top = np.argsort(-o, kind="stable")[:k]; hits.append(y[top].sum())
        rows[n] = np.array(hits)
    d = rows["M5"] - rows["M1"]
    cap[f"{int(frac*100)}%"] = dict(k=k, M1_recall=float(rows["M1"].mean() / y.sum()), M5_recall=float(rows["M5"].mean() / y.sum()),
                                    M1_prec=float(rows["M1"].mean() / k), M5_prec=float(rows["M5"].mean() / k),
                                    diff_hits_mean=float(d.mean()), diff_hits_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], share_reps_M5_better=float(np.mean(d > 0)), share_ties=float(np.mean(d == 0)))
R["capacity"] = cap
# cost-sensitive: missed at-risk student costs 5x an unnecessary counseling session (assumption stated in report)
cost = {}
for n in ["M1", "M5"]:
    best = []
    for o in rep_oof[n]:
        c = [ (5 * ((o < t) & (y == 1)).sum() + 1 * ((o >= t) & (y == 0)).sum()) / len(y) for t in np.linspace(0.05, 0.95, 91)]
        best.append(min(c))
    cost[n] = dict(mean=float(np.mean(best)), ci=[float(np.percentile(best, 2.5)), float(np.percentile(best, 97.5))])
cost["at_0.5"] = {n: float(np.mean([(5 * ((o < .5) & (y == 1)).sum() + ((o >= .5) & (y == 0)).sum()) / len(y) for o in rep_oof[n]])) for n in ["M1", "M5"]}
R["cost_5to1"] = cost
print("capacity done")

# ---- 4. Learning curve (M5, 20 features) ----
lc = {}
for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
    prs, rocs = [], []
    for s in range(10):
        _, fl = cv_run(F20, split_seed=2000 + s, frac=frac, sub_seed=s)
        prs.append(np.mean([f["pr"] for f in fl])); rocs.append(np.mean([f["roc"] for f in fl]))
    lc[str(frac)] = dict(train_n=int(round(304 * frac)), pr_mean=float(np.mean(prs)), pr_ci=[float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5))],
                         roc_mean=float(np.mean(rocs)), roc_ci=[float(np.percentile(rocs, 2.5)), float(np.percentile(rocs, 97.5))])
    print("lc", frac)
R["learning_curve_M5"] = lc

# ---- 5. Correlated features, grouped SHAP, grouped permutation importance ----
corr = tr[F20].corr(method="spearman")
pairs = []
for i, a in enumerate(F20):
    for b in F20[i + 1:]:
        v = corr.loc[a, b]
        if abs(v) >= 0.4: pairs.append((a, b, float(v)))
R["corr_pairs_abs_ge_0.4"] = sorted(pairs, key=lambda t: -abs(t[2]))
sv = shap.TreeExplainer(m20)(tr[F20]).values
groups = {"Academic load (Registrar)": acad, "Demographic": demo, "Commuting mobility": geo, "Socioeconomic/civil (TES)": socio}
gs = {k: float(np.abs(sv[:, [F20.index(c) for c in cols]].sum(1)).mean()) for k, cols in groups.items()}
R["grouped_shap_mean_abs"] = gs
# grouped permutation importance on held-out folds (10 repeats of the seed-42 split)
sk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42); perm = {k: [] for k in groups}; perm_single = {c: [] for c in ["has_philsys_id", "year_level", "program_is_bsis", "distance_km"]}
for a, b in sk.split(tr[F20], y, groups=g):
    m = cb(); m.fit(tr[F20].iloc[a], y[a]); Xv = tr[F20].iloc[b].copy(); base = average_precision_score(y[b], m.predict_proba(Xv)[:, 1])
    for r in range(10):
        rr = np.random.RandomState(r)
        for k, cols in groups.items():
            Xp = Xv.copy(); idx = rr.permutation(len(Xp)); Xp[cols] = Xv[cols].values[idx]
            perm[k].append(base - average_precision_score(y[b], m.predict_proba(Xp)[:, 1]))
        for c in perm_single:
            Xp = Xv.copy(); Xp[c] = rr.permutation(Xv[c].values)
            perm_single[c].append(base - average_precision_score(y[b], m.predict_proba(Xp)[:, 1]))
R["grouped_permutation_drop_prauc"] = {k: dict(mean=float(np.mean(v)), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]) for k, v in perm.items()}
R["single_permutation_drop_prauc"] = {k: dict(mean=float(np.mean(v)), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]) for k, v in perm_single.items()}
# PhilSys association within program-year strata (descriptive, not causal)
st = tr.groupby(["program_is_bsis", "year_level", "has_philsys_id"]).at_risk.agg(["mean", "count"]).reset_index()
R["philsys_strata"] = st.to_dict(orient="records")
# Bootstrap CI for headline M5 metrics (pooled OOF, seed-42)
bm = {"pr": [], "roc": [], "rec": []}
rng = np.random.RandomState(11)
for _ in range(B):
    i = rng.randint(0, len(y), len(y))
    bm["pr"].append(average_precision_score(y[i], oof20[i])); bm["roc"].append(roc_auc_score(y[i], oof20[i])); bm["rec"].append(recall_score(y[i], (oof20[i] >= .5).astype(int)))
R["bootstrap_M5_pooled"] = {k: dict(point=float(np.mean(v)), ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]) for k, v in bm.items()}
json.dump(R, open(os.path.join(BASE, "panel_revision_results.json"), "w"), indent=1)
pd.DataFrame({"student_pid": te.student_pid, "risk_score_20feat": p20}).to_csv(os.path.join(D, "ml_results", "prospective_scores_20feat.csv"), index=False)
print("ALL DONE")
