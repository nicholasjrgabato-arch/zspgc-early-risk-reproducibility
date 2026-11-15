# Requires scikit-learn >= 1.9 to reproduce the manuscript fold assignment (StratifiedGroupKFold, seed 42).
"""Additional analyses for manuscript revision v4 (cross-cohort J10, extended nested CV incl. depth 1-2,
subgroup bootstrap CIs, Year-4 sensitivity, default-threshold counts, RF Brier check)."""
import json, os, sys, warnings, itertools
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from scipy.stats import spearmanr
import shap
warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(BASE, "cleaned_dataset") + os.sep
df = pd.read_csv(D + "student_risk_ml_dataset.csv")
tr = df[df.split_type == "train_eval"].reset_index(drop=True); te = df[df.split_type == "prospective_test"].reset_index(drop=True)
y = tr.at_risk.astype(int).values; g = tr.student_pid.values
acad = ['program_is_bsis','year_level','enrolled_units','course_count','has_lab_courses','lab_courses_count','is_underload','is_overload']
geo = ['distance_km','locality_is_campus_area','locality_is_distant']
socio = ['is_4ps_beneficiary','income_bracket','has_philsys_id','is_first_gen_college','is_dependent','has_disability','is_indigenous']
F20 = ['program_is_bsis','year_level','age','is_female'] + geo + socio + ['enrolled_units','course_count','has_lab_courses','lab_courses_count','is_underload','is_overload']
OUT = os.environ.get("OUT", os.path.join(BASE, "revision_v4_results.json"))
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
def cb(depth=4, it=150, lr=0.05, seed=42):
    return CatBoostClassifier(iterations=it, depth=depth, learning_rate=lr, auto_class_weights='Balanced', verbose=False, random_seed=seed, thread_count=4)
def oof_run(cols, seed, mk=lambda: cb(), X=tr, yy=y, gg=g):
    o = np.zeros(len(yy)); fl = []
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X[cols], yy, groups=gg):
        m = mk(); m.fit(X[cols].iloc[a], yy[a]); p = m.predict_proba(X[cols].iloc[b])[:, 1]; o[b] = p
        fl.append(dict(pr=average_precision_score(yy[b], p), roc=roc_auc_score(yy[b], p), brier=brier_score_loss(yy[b], p)))
    return o, fl
step = sys.argv[1]
if step == "quick":
    # A. cross-cohort stability incl. J10 (same model as run_ml_experiment 6D)
    m = cb(); m.fit(tr[F20], y); ex = shap.TreeExplainer(m)
    vtr = np.abs(ex(tr[F20]).values).mean(0); vte = np.abs(ex(te[F20]).values).mean(0)
    top = lambda v, k: set(np.argsort(v)[::-1][:k])
    R["cross_cohort"] = dict(rho=float(spearmanr(vtr, vte)[0]), J5=len(top(vtr,5)&top(vte,5))/len(top(vtr,5)|top(vte,5)),
                             J10=len(top(vtr,10)&top(vte,10))/len(top(vtr,10)|top(vte,10)),
                             top10_train=[F20[i] for i in np.argsort(vtr)[::-1][:10]], top10_test=[F20[i] for i in np.argsort(vte)[::-1][:10]])
    # B. reference OOF (seed 42): default-threshold counts and subgroup bootstrap
    o, fl = oof_run(F20, 42)
    flag = o >= 0.5
    R["ref_default"] = dict(flagged=int(flag.sum()), tp=int((flag & (y==1)).sum()), fp=int((flag & (y==0)).sum()),
                            pooled_recall=float((flag & (y==1)).sum()/y.sum()), pooled_prec=float((flag & (y==1)).sum()/flag.sum()),
                            fold_mean_pr=float(np.mean([f["pr"] for f in fl])))
    subs = {"Female": tr.is_female==1, "Male": tr.is_female==0, "4Ps": tr.is_4ps_beneficiary==1, "Non-4Ps": tr.is_4ps_beneficiary==0,
            "Distant": tr.locality_is_distant==1, "Local": tr.locality_is_distant==0, "FirstGen": tr.is_first_gen_college==1, "ContGen": tr.is_first_gen_college==0}
    rng = np.random.RandomState(2026); B = 2000
    def sm(mask, idx):
        yy, pp = y[idx][mask[idx]], o[idx][mask[idx]]
        rec = ((pp>=.5)&(yy==1)).sum()/max(1,(yy==1).sum()); fpr = ((pp>=.5)&(yy==0)).sum()/max(1,(yy==0).sum())
        auc = roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan
        return rec, fpr, auc
    full = np.arange(len(y)); boots = {k: [] for k in subs}; diffs = {"sex":[], "4ps":[], "loc":[], "gen":[]}
    for _ in range(B):
        i = rng.randint(0, len(y), len(y)); r = {k: sm(np.asarray(v), i) for k, v in subs.items()}
        for k in subs: boots[k].append(r[k])
        diffs["sex"].append(r["Female"][0]-r["Male"][0]); diffs["4ps"].append(r["4Ps"][0]-r["Non-4Ps"][0])
        diffs["loc"].append(r["Distant"][0]-r["Local"][0]); diffs["gen"].append(r["FirstGen"][0]-r["ContGen"][0])
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    R["subgroups_tmp"]=1
    R["subgroups"] = {k: dict(n=int(np.asarray(v).sum()), pos=int(y[np.asarray(v)].sum()), point=[float(x) for x in sm(np.asarray(v), full)],
                              rec_ci=ci([b[0] for b in boots[k]]), fpr_ci=ci([b[1] for b in boots[k]]), auc_ci=ci([b[2] for b in boots[k]])) for k, v in subs.items()}
    R["recall_gap_ci"] = {k: dict(point=float(np.mean(v)), ci=ci(v)) for k, v in diffs.items()}
    print(json.dumps({k: R[k] for k in ["cross_cohort","ref_default","recall_gap_ci"]}, indent=1))
    save()
if step == "y4":
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    # C. Year-4 sensitivity: performance on Years 1-3 only (models trained on all) and retrained without Year 4, 10 partitions
    y4 = (tr.year_level == 4).values
    R["year4"] = dict(n_year4=int(y4.sum()), pos_year4=int(y[y4].sum()))
    res = {"M1_all_eval123": [], "M5_all_eval123": [], "M1_retrain123": [], "M5_retrain123": [], "M1_all": [], "M5_all": []}
    t123 = tr[~y4].reset_index(drop=True); y123 = y[~y4]; g123 = g[~y4]
    for s in range(10):
        for nm, cols in [("M1", acad), ("M5", F20)]:
            oo, _ = oof_run(cols, 1000+s)
            res[f"{nm}_all"].append((average_precision_score(y, oo), roc_auc_score(y, oo)))
            res[f"{nm}_all_eval123"].append((average_precision_score(y123, oo[~y4]), roc_auc_score(y123, oo[~y4])))
            o2, _ = oof_run(cols, 1000+s, X=t123, yy=y123, gg=g123)
            res[f"{nm}_retrain123"].append((average_precision_score(y123, o2), roc_auc_score(y123, o2)))
    R["year4"]["pooled"] = {k: dict(pr=float(np.mean([a for a,_ in v])), pr_ci=ci([a for a,_ in v]), roc=float(np.mean([b for _,b in v]))) for k, v in res.items()}
    save(); print(json.dumps(R["year4"], indent=1))
if step == "rf":
    # D. RF Brier check, reference partition (Table II setting)
    rf = lambda: RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=4, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=4)
    orf, frf = oof_run(F20, 42, mk=rf)
    R["rf_brier_ref"] = dict(fold_mean=float(np.mean([f["brier"] for f in frf])), pooled=float(brier_score_loss(y, orf)))
    save(); print(json.dumps(R["rf_brier_ref"]))
if step == "nested":
    grid = list(itertools.product([1,2,3,4,6],[0.03,0.05,0.1],[150,300]))
    rows = []
    for r in range(5):
        for k, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=3000+r).split(tr[F20], y, groups=g)):
            Xa, ya, ga = tr[F20].iloc[a].reset_index(drop=True), y[a], g[a]
            best, bs = None, -1
            for (d, lr, it) in grid:
                sc = []
                for ia, ib in StratifiedGroupKFold(3, shuffle=True, random_state=7).split(Xa, ya, groups=ga):
                    m = cb(d, it, lr); m.fit(Xa.iloc[ia], ya[ia]); sc.append(average_precision_score(ya[ib], m.predict_proba(Xa.iloc[ib])[:,1]))
                if np.mean(sc) > bs: bs, best = np.mean(sc), (d, lr, it)
            out = {}
            for nm, mk in [("tuned", lambda: cb(*[best[0], best[2], best[1]])), ("fixed_d4", lambda: cb(4)), ("fixed_d1", lambda: cb(1))]:
                m = mk(); m.fit(Xa, ya); out[nm] = average_precision_score(y[b], m.predict_proba(tr[F20].iloc[b])[:,1])
            rows.append(dict(rep=r, fold=k, depth=best[0], lr=best[1], it=best[2], **out)); print(rows[-1], flush=True)
    dd = pd.DataFrame(rows); bt = lambda d: [float(np.percentile(d,2.5)), float(np.percentile(d,97.5))]
    R["nested_extended"] = dict(tuned=float(dd.tuned.mean()), fixed_d4=float(dd.fixed_d4.mean()), fixed_d1=float(dd.fixed_d1.mean()),
        tuned_minus_d4=float((dd.tuned-dd.fixed_d4).mean()), tuned_minus_d4_ci=bt(dd.tuned-dd.fixed_d4),
        d1_minus_d4=float((dd.fixed_d1-dd.fixed_d4).mean()), d1_minus_d4_ci=bt(dd.fixed_d1-dd.fixed_d4),
        depth_counts=dd.depth.value_counts().to_dict(), folds=rows)
    R["nested_extended"]["depth_counts"] = {str(k): int(v) for k, v in R["nested_extended"]["depth_counts"].items()}
    save(); print("NESTED DONE", json.dumps({k: v for k, v in R["nested_extended"].items() if k != "folds"}))
if step == "depth1":
    # depth-1 vs depth-4 CatBoost on 10 repeated partitions: pooled PR-AUC, ROC, Brier, precision@20%
    out = {"d4": [], "d1": []}
    for s in range(10):
        for nm, d in [("d4", 4), ("d1", 1)]:
            o, _ = oof_run(F20, 1000+s, mk=lambda d=d: cb(d)); top = np.argsort(-o, kind="stable")[:76]
            out[nm].append((average_precision_score(y, o), roc_auc_score(y, o), brier_score_loss(y, o), y[top].mean()))
    R["depth1_vs_depth4"] = {nm: dict(zip(["pr","roc","brier","prec20"], [float(x) for x in np.mean(v, 0)])) for nm, v in out.items()}
    R["depth1_vs_depth4"]["pr_diff_ci"] = [float(np.percentile(np.array(out["d1"])[:,0]-np.array(out["d4"])[:,0], q)) for q in (2.5, 97.5)]
    save(); print(json.dumps(R["depth1_vs_depth4"], indent=1))
