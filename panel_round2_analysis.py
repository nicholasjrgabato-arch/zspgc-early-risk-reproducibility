"""Round-2 panel analyses: calibration, model-selection objective, tuning (nested CV), group-vs-stratified CV,
feature interactions, threshold derivation, fold assignments. Same data and model settings as run_ml_experiment.py."""
import os, json, warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
import shap

BASE = os.path.dirname(os.path.abspath(__file__))
D = os.environ.get("DATA_DIR", os.path.join(BASE, "cleaned_dataset"))
OUT = os.environ.get("OUT_DIR", BASE)
df = pd.read_csv(os.path.join(D, "student_risk_ml_dataset.csv"))
tr = df[df.split_type == "train_eval"].reset_index(drop=True); te = df[df.split_type == "prospective_test"].reset_index(drop=True)
y = tr.at_risk.astype(int).values; g = tr.student_pid.values
acad = ['program_is_bsis', 'year_level', 'enrolled_units', 'course_count', 'has_lab_courses', 'lab_courses_count', 'is_underload', 'is_overload']
demo = ['age', 'is_female']; geo = ['distance_km', 'locality_is_campus_area', 'locality_is_distant']
socio = ['is_4ps_beneficiary', 'income_bracket', 'has_philsys_id', 'is_first_gen_college', 'is_dependent', 'has_disability', 'is_indigenous']
F20 = ['program_is_bsis', 'year_level', 'age', 'is_female'] + geo + socio + ['enrolled_units', 'course_count', 'has_lab_courses', 'lab_courses_count', 'is_underload', 'is_overload']
GROUP = {**{c: "A" for c in acad}, **{c: "B" for c in demo}, **{c: "C" for c in geo}, **{c: "D" for c in socio}}
SPW = (y == 0).sum() / (y == 1).sum()
R = {"n": int(len(y)), "pos": int(y.sum()), "unique_students": int(pd.Series(g).nunique())}

def models(seed=42):
    return {"Logistic Regression": Pipeline([("s", StandardScaler()), ("c", LogisticRegression(class_weight="balanced", C=0.5, max_iter=1000, random_state=seed))]),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=4, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=4),
            "XGBoost": xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=SPW, eval_metric="logloss", random_state=seed, n_jobs=4),
            "LightGBM": lgb.LGBMClassifier(n_estimators=150, max_depth=4, num_leaves=15, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=SPW, random_state=seed, verbose=-1, n_jobs=4),
            "CatBoost": CatBoostClassifier(iterations=150, depth=4, learning_rate=0.05, auto_class_weights="Balanced", verbose=False, random_seed=seed, thread_count=4)}

def splits(seed, grouped=True):
    sk = StratifiedGroupKFold(5, shuffle=True, random_state=seed) if grouped else StratifiedKFold(5, shuffle=True, random_state=seed)
    return list(sk.split(tr[F20], y, groups=g) if grouped else sk.split(tr[F20], y))

def oof_pred(make, cols, sp):
    o = np.zeros(len(y))
    for a, b in sp:
        m = make(); m.fit(tr[cols].iloc[a], y[a]); o[b] = m.predict_proba(tr[cols].iloc[b])[:, 1]
    return o

def prec_at(o, frac=0.20):
    k = int(round(frac * len(y))); top = np.argsort(-o, kind="stable")[:k]; return y[top].mean()

def calib(o):
    eps = 1e-6; lp = np.log(np.clip(o, eps, 1 - eps) / (1 - np.clip(o, eps, 1 - eps)))
    sl = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y); slope = float(sl.coef_[0][0])
    # calibration-in-the-large: intercept with slope fixed at 1 (offset); solve by 1-D Newton
    a = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(a + lp))); a -= (p - y).sum() / (p * (1 - p)).sum()
    bins = np.quantile(o, np.linspace(0, 1, 11)); idx = np.clip(np.digitize(o, bins[1:-1]), 0, 9)
    ece = sum(abs(o[idx == k].mean() - y[idx == k].mean()) * (idx == k).mean() for k in range(10) if (idx == k).any())
    return dict(slope=slope, intercept=float(a), ece=float(ece), mean_pred=float(o.mean()), prevalence=float(y.mean()), brier=float(brier_score_loss(y, o)))

# ---- 1. Fold assignments + group vs stratified CV ----
fold_rows = []
for s in [42] + list(range(1000, 1020)):
    for k, (a, b) in enumerate(splits(s)):
        for i in b: fold_rows.append((s, g[i], k))
pd.DataFrame(fold_rows, columns=["split_seed", "student_pid", "test_fold"]).to_csv(os.path.join(OUT, "fold_assignments.csv"), index=False)
cbm = lambda: models()["CatBoost"]
gs, ss = [], []
for s in range(1000, 1010):
    og = oof_pred(cbm, F20, splits(s, True)); os_ = oof_pred(cbm, F20, splits(s, False))
    gs.append(average_precision_score(y, og)); ss.append(average_precision_score(y, os_))
R["group_vs_stratified"] = dict(grouped_pr_mean=float(np.mean(gs)), strat_pr_mean=float(np.mean(ss)), grouped_sd=float(np.std(gs)), strat_sd=float(np.std(ss)),
                                returning_students_in_cohort2=int(te.is_returning_student.sum()),
                                students_in_both_cohorts=int(len(set(tr.student_pid) & set(te.student_pid))))
print("group done", R["group_vs_stratified"])

# ---- 2. Calibration + model-selection objective (reference split + 10 repeats) ----
ref = splits(42); cal = {}; sel = {n: {"pr": [], "p20": [], "brier": [], "roc": [], "slope": []} for n in models()}
oof_ref = {}
for n in models():
    o = oof_pred(lambda n=n: models()[n], F20, ref); oof_ref[n] = o; cal[n] = calib(o)
for s in range(1000, 1010):
    sp = splits(s)
    for n in models():
        o = oof_pred(lambda n=n: models()[n], F20, sp)
        sel[n]["pr"].append(average_precision_score(y, o)); sel[n]["p20"].append(prec_at(o)); sel[n]["brier"].append(brier_score_loss(y, o))
        sel[n]["roc"].append(roc_auc_score(y, o)); sel[n]["slope"].append(calib(o)["slope"])
    print("sel rep", s)
R["calibration_ref"] = cal
R["selection_repeated"] = {n: {k: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()} for n, d in sel.items()}
# recalibration of CatBoost inside CV (Platt / isotonic fitted on inner OOF of the training fold)
def recal_oof(method):
    o = np.zeros(len(y))
    for a, b in ref:
        inner = np.zeros(len(a)); sk = StratifiedKFold(5, shuffle=True, random_state=7)
        for ia, ib in sk.split(tr[F20].iloc[a], y[a]):
            m = cbm(); m.fit(tr[F20].iloc[a[ia]], y[a[ia]]); inner[ib] = m.predict_proba(tr[F20].iloc[a[ib]])[:, 1]
        m = cbm(); m.fit(tr[F20].iloc[a], y[a]); raw = m.predict_proba(tr[F20].iloc[b])[:, 1]
        if method == "platt":
            lr = LogisticRegression(C=1e6).fit(np.log(inner / (1 - inner)).reshape(-1, 1), y[a]); o[b] = lr.predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]
        else:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(inner, y[a]); o[b] = iso.predict(raw)
    return o
for meth in ["platt", "isotonic"]:
    o = recal_oof(meth); c = calib(o); c.update(pr=float(average_precision_score(y, o)), roc=float(roc_auc_score(y, o))); R[f"recal_{meth}"] = c
    oof_ref["CatBoost+" + meth] = o
c = cal["CatBoost"]; c.update(pr=float(average_precision_score(y, oof_ref["CatBoost"])), roc=float(roc_auc_score(y, oof_ref["CatBoost"])))
pd.DataFrame({"student_pid": g, "y": y, **{k: v for k, v in oof_ref.items()}}).to_csv(os.path.join(OUT, "oof_predictions_reference_split.csv"), index=False)
print("calibration done")

# ---- 3. Nested CV for CatBoost hyperparameters ----
grid = [dict(depth=d, learning_rate=lr, iterations=it) for d in (3, 4, 6) for lr in (0.03, 0.05, 0.1) for it in (150, 300)]
nest = {"nested_pr": [], "fixed_pr": [], "chosen": []}
for s in range(1000, 1005):
    for a, b in splits(s):
        sk = StratifiedKFold(3, shuffle=True, random_state=s)
        best, bestv = None, -1
        for prm in grid:
            sc = []
            for ia, ib in sk.split(tr[F20].iloc[a], y[a]):
                m = CatBoostClassifier(**prm, auto_class_weights="Balanced", verbose=False, random_seed=42, thread_count=4)
                m.fit(tr[F20].iloc[a[ia]], y[a[ia]]); sc.append(average_precision_score(y[a[ib]], m.predict_proba(tr[F20].iloc[a[ib]])[:, 1]))
            if np.mean(sc) > bestv: best, bestv = prm, np.mean(sc)
        m = CatBoostClassifier(**best, auto_class_weights="Balanced", verbose=False, random_seed=42, thread_count=4); m.fit(tr[F20].iloc[a], y[a])
        nest["nested_pr"].append(average_precision_score(y[b], m.predict_proba(tr[F20].iloc[b])[:, 1]))
        m = cbm(); m.fit(tr[F20].iloc[a], y[a]); nest["fixed_pr"].append(average_precision_score(y[b], m.predict_proba(tr[F20].iloc[b])[:, 1]))
        nest["chosen"].append(f"d{best['depth']}_lr{best['learning_rate']}_it{best['iterations']}")
    print("nested rep", s)
d = np.array(nest["nested_pr"]) - np.array(nest["fixed_pr"])
R["nested_cv"] = dict(nested_mean=float(np.mean(nest["nested_pr"])), fixed_mean=float(np.mean(nest["fixed_pr"])), diff_mean=float(d.mean()),
                      diff_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], chosen_counts=pd.Series(nest["chosen"]).value_counts().to_dict(), grid_size=len(grid), outer_folds=len(d))
print("nested done", R["nested_cv"])

# ---- 4. Interactions: SHAP interaction values + controlled ablation ----
full = cbm(); full.fit(tr[F20], y)
iv = shap.TreeExplainer(full).shap_interaction_values(tr[F20]); iv = np.array(iv)
if iv.ndim == 4: iv = iv[1]
mabs = np.abs(iv).mean(0); n = len(F20)
main = float(np.trace(mabs)); off = float(mabs.sum() - np.trace(mabs))
within = sum(mabs[i, j] for i in range(n) for j in range(n) if i != j and GROUP[F20[i]] == GROUP[F20[j]])
cross = off - within
pairs = sorted([(F20[i], F20[j], float(2 * mabs[i, j])) for i in range(n) for j in range(i + 1, n)], key=lambda t: -t[2])[:8]
R["shap_interactions"] = dict(main_effect_share=main / (main + off), interaction_share=off / (main + off), cross_domain_share_of_interaction=cross / off if off else 0, top_pairs=pairs)
# controlled ablation with XGBoost interaction constraints (same settings), 10 repeats
def xgbm(**kw):
    p = dict(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=SPW, eval_metric="logloss", random_state=42, n_jobs=4); p.update(kw)
    return xgb.XGBClassifier(**p)
groups_idx = [[F20.index(c) for c in F20 if GROUP[c] == G] for G in "ABCD"]
variants = {"Unconstrained (depth 4)": lambda: xgbm(),
            "Within-source interactions only": lambda: xgbm(interaction_constraints=str(groups_idx)),
            "Additive, no interactions (depth 1)": lambda: xgbm(max_depth=1, n_estimators=300),
            "CatBoost depth 4": cbm,
            "CatBoost depth 1 (additive)": lambda: CatBoostClassifier(iterations=300, depth=1, learning_rate=0.05, auto_class_weights="Balanced", verbose=False, random_seed=42, thread_count=4)}
abl = {k: [] for k in variants}
for s in range(1000, 1010):
    sp = splits(s)
    for k, mk in variants.items(): abl[k].append(average_precision_score(y, oof_pred(mk, F20, sp)))
    print("int rep", s)
base_u = np.array(abl["Unconstrained (depth 4)"]); base_c = np.array(abl["CatBoost depth 4"])
R["interaction_ablation"] = {k: dict(pr_mean=float(np.mean(v)), diff_vs_ref=float(np.mean(np.array(v) - (base_c if k.startswith("CatBoost") else base_u))),
                                     diff_ci=[float(np.percentile(np.array(v) - (base_c if k.startswith("CatBoost") else base_u), 2.5)), float(np.percentile(np.array(v) - (base_c if k.startswith("CatBoost") else base_u), 97.5))]) for k, v in abl.items()}
print("interactions done")

# ---- 5. Thresholds from Cohort-1 OOF only ----
o = oof_ref["CatBoost"]
def share_above(t, s=o): return float((s >= t).mean())
qs = {}
for frac in (0.10, 0.20, 0.30, 0.40, 0.50):
    cut = float(np.sort(o)[::-1][int(round(frac * len(o))) - 1]); qs[f"{int(frac*100)}%"] = cut
rep_cuts = {f: [] for f in ("30%", "50%")}
for s in range(1000, 1020):
    oo = oof_pred(cbm, F20, splits(s))
    for f, fr in (("30%", .30), ("50%", .50)): rep_cuts[f].append(float(np.sort(oo)[::-1][int(round(fr * len(oo))) - 1]))
tiers = {}
for lo, hi, name in [(0.60, 9, "High (>=0.60)"), (0.35, 0.60, "Moderate (0.35-0.60)"), (-1, 0.35, "Low (<0.35)")]:
    m = (o >= lo) & (o < hi); tiers[name] = dict(share=float(m.mean()), n=int(m.sum()), observed_rate=float(y[m].mean()) if m.any() else None, captured=int(y[m].sum()))
full_te = full.predict_proba(te[F20])[:, 1]
R["thresholds"] = dict(oof_cutoffs_by_budget=qs, rep_cut_30=[float(np.mean(rep_cuts["30%"])), float(min(rep_cuts["30%"])), float(max(rep_cuts["30%"]))],
                       rep_cut_50=[float(np.mean(rep_cuts["50%"])), float(min(rep_cuts["50%"])), float(max(rep_cuts["50%"]))],
                       share_ge_060_cohort1=share_above(0.60), share_ge_035_cohort1=share_above(0.35), cohort1_tiers=tiers,
                       cohort2_share_ge_060=float((full_te >= .60).mean()), cohort2_share_ge_035=float((full_te >= .35).mean()))
json.dump(R, open(os.path.join(OUT, "round2_results.json"), "w"), indent=1, default=str)
print("ALL DONE")
