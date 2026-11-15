import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, precision_score, accuracy_score, confusion_matrix, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR = os.path.join(BASE_DIR, 'cleaned_dataset')
OUTPUT_DIR = os.path.join(CLEAN_DIR, 'ml_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load Data
ml_data_path = os.path.join(CLEAN_DIR, 'student_risk_ml_dataset.csv')
prospective_path = os.path.join(CLEAN_DIR, 'student_risk_prospective_test.csv')

df_ml = pd.read_csv(ml_data_path)
df_prospective = pd.read_csv(prospective_path)

train_df = df_ml[df_ml['split_type'] == 'train_eval'].copy()
test_df = df_ml[df_ml['split_type'] == 'prospective_test'].copy()

print(f"Historical Train/Eval: {len(train_df)} rows")
print(f"Prospective Test: {len(test_df)} rows")

# Feature groups for Multi-Source Ablation & Full Pipeline
acad_cols = [
    'program_is_bsis', 'year_level', 'enrolled_units', 'course_count',
    'has_lab_courses', 'lab_courses_count', 'is_underload', 'is_overload'
]
demo_cols = ['age', 'is_female']
geo_cols = ['distance_km', 'locality_is_campus_area', 'locality_is_distant']
socio_cols = [
    'is_4ps_beneficiary', 'income_bracket', 'has_philsys_id',
    'is_first_gen_college', 'is_dependent', 'has_disability', 'is_indigenous'
]
long_cols = [
    'is_returning_student', 'prior_term_gpa', 'has_prior_gpa_recorded',
    'prior_deficiencies', 'prior_at_risk'
]

# Full 25-feature vector (matches existing schema)
feature_cols = (
    ['program_is_bsis', 'year_level', 'age', 'is_female'] +
    geo_cols + socio_cols +
    ['enrolled_units', 'course_count', 'has_lab_courses', 'lab_courses_count', 'is_underload', 'is_overload'] +
    long_cols
)

X_train = train_df[feature_cols].copy()
y_train = train_df['at_risk'].astype(int).values
groups_train = train_df['student_pid'].values

X_test = test_df[feature_cols].copy()

neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())
scale_pos_weight = neg_count / pos_count
print(f"Class Balance (Historical n=380): {pos_count} At-Risk ({pos_count/len(y_train)*100:.2f}%), {neg_count} Satisfactory ({neg_count/len(y_train)*100:.2f}%)")
print(f"Computed scale_pos_weight = {scale_pos_weight:.2f}")

# 2. Define Model Architectures
def get_models(scale_pos_weight, seed=RANDOM_STATE):
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                class_weight='balanced',
                C=0.5,
                max_iter=1000,
                random_state=seed
            ))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=seed,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            verbose=-1,
            n_jobs=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=150,
            depth=4,
            learning_rate=0.05,
            auto_class_weights='Balanced',
            verbose=False,
            random_seed=seed
        )
    }

# 3. StratifiedGroupKFold Cross-Validation (k=5)
n_splits = 5
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

models = get_models(scale_pos_weight)
results = {}
oof_predictions = {name: np.zeros(len(y_train)) for name in models}
oof_probabilities = {name: np.zeros(len(y_train)) for name in models}

print("\n" + "="*70)
print(f"STRATIFIED GROUP {n_splits}-FOLD CROSS-VALIDATION BENCHMARK")
print("="*70)

for name, model in models.items():
    fold_recalls, fold_f1s, fold_aucs, fold_praucs, fold_briers = [], [], [], [], []
    fold_precisions, fold_accuracies, fold_specificities = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups=groups_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
        
        model.fit(X_tr, y_tr)
        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        oof_probabilities[name][val_idx] = y_proba
        oof_predictions[name][val_idx] = y_pred
        
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        prec = precision_score(y_val, y_pred, zero_division=0)
        acc = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        prauc = average_precision_score(y_val, y_proba)
        brier = brier_score_loss(y_val, y_proba)
        
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        fold_aucs.append(auc)
        fold_praucs.append(prauc)
        fold_briers.append(brier)
        fold_precisions.append(prec)
        fold_accuracies.append(acc)
        fold_specificities.append(spec)
        
    results[name] = {
        'Recall': (np.mean(fold_recalls), np.std(fold_recalls)),
        'F1-Score': (np.mean(fold_f1s), np.std(fold_f1s)),
        'ROC-AUC': (np.mean(fold_aucs), np.std(fold_aucs)),
        'PR-AUC': (np.mean(fold_praucs), np.std(fold_praucs)),
        'Brier': (np.mean(fold_briers), np.std(fold_briers)),
        'Precision': (np.mean(fold_precisions), np.std(fold_precisions)),
        'Accuracy': (np.mean(fold_accuracies), np.std(fold_accuracies)),
        'Specificity': (np.mean(fold_specificities), np.std(fold_specificities))
    }
    
    print(f"\nModel: {name}")
    print(f"  Recall:      {np.mean(fold_recalls):.4f} +/- {np.std(fold_recalls):.4f}")
    print(f"  F1-Score:    {np.mean(fold_f1s):.4f} +/- {np.std(fold_f1s):.4f}")
    print(f"  ROC-AUC:     {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print(f"  PR-AUC:      {np.mean(fold_praucs):.4f} +/- {np.std(fold_praucs):.4f}")
    print(f"  Brier Score: {np.mean(fold_briers):.4f} +/- {np.std(fold_briers):.4f}")
    print(f"  Precision:   {np.mean(fold_precisions):.4f} | Accuracy: {np.mean(fold_accuracies):.4f} | Specificity: {np.mean(fold_specificities):.4f}")

# 4. Save Metrics Summary Table
metrics_table = []
for name, m in results.items():
    metrics_table.append({
        'Model': name,
        'Recall (Mean)': round(m['Recall'][0], 4),
        'Recall (Std)': round(m['Recall'][1], 4),
        'F1-Score (Mean)': round(m['F1-Score'][0], 4),
        'F1-Score (Std)': round(m['F1-Score'][1], 4),
        'ROC-AUC (Mean)': round(m['ROC-AUC'][0], 4),
        'ROC-AUC (Std)': round(m['ROC-AUC'][1], 4),
        'PR-AUC (Mean)': round(m['PR-AUC'][0], 4),
        'PR-AUC (Std)': round(m['PR-AUC'][1], 4),
        'Brier Score (Mean)': round(m['Brier'][0], 4),
        'Precision (Mean)': round(m['Precision'][0], 4),
        'Accuracy (Mean)': round(m['Accuracy'][0], 4),
        'Specificity (Mean)': round(m['Specificity'][0], 4)
    })
metrics_df = pd.DataFrame(metrics_table)
metrics_csv_path = os.path.join(OUTPUT_DIR, 'model_benchmark_results.csv')
metrics_df.to_csv(metrics_csv_path, index=False)

best_model_name = 'CatBoost'

# 5. Multi-Source Feature-Group Ablation Study (RQ1)
print("\n" + "="*70)
print("MULTI-SOURCE FEATURE-GROUP ABLATION STUDY (CATBOOST 5-FOLD CV)")
print("="*70)

ablation_configs = [
    ('M1: Academic-Only Baseline (Registrar)', acad_cols),
    ('M2: Academic + Demographic', acad_cols + demo_cols),
    ('M3: Academic + Demo + Geographic Mobility', acad_cols + demo_cols + geo_cols),
    ('M4: Academic + Socioeconomic/TES Only', acad_cols + socio_cols),
    ('M5: Full Multi-Source Integration (Proposed)', feature_cols)
]

ablation_rows = []
for label, cols in ablation_configs:
    X_ab = train_df[cols].copy()
    f_rec, f_f1, f_auc, f_prauc, f_brier, f_prec, f_acc = [], [], [], [], [], [], []
    for tr_idx, val_idx in sgkf.split(X_ab, y_train, groups=groups_train):
        cb = CatBoostClassifier(iterations=150, depth=4, learning_rate=0.05, auto_class_weights='Balanced', verbose=False, random_seed=RANDOM_STATE)
        cb.fit(X_ab.iloc[tr_idx], y_train[tr_idx])
        p = cb.predict_proba(X_ab.iloc[val_idx])[:, 1]
        pred = (p >= 0.5).astype(int)
        f_rec.append(recall_score(y_train[val_idx], pred, zero_division=0))
        f_f1.append(f1_score(y_train[val_idx], pred, zero_division=0))
        f_auc.append(roc_auc_score(y_train[val_idx], p))
        f_prauc.append(average_precision_score(y_train[val_idx], p))
        f_brier.append(brier_score_loss(y_train[val_idx], p))
        f_prec.append(precision_score(y_train[val_idx], pred, zero_division=0))
        f_acc.append(accuracy_score(y_train[val_idx], pred))
    row_dict = {
        'Configuration': label,
        'Features': len(cols),
        'Recall': f"{np.mean(f_rec):.4f} ± {np.std(f_rec):.4f}",
        'F1-Score': f"{np.mean(f_f1):.4f} ± {np.std(f_f1):.4f}",
        'ROC-AUC': f"{np.mean(f_auc):.4f} ± {np.std(f_auc):.4f}",
        'PR-AUC': f"{np.mean(f_prauc):.4f} ± {np.std(f_prauc):.4f}",
        'Brier': f"{np.mean(f_brier):.4f}",
        'Precision': f"{np.mean(f_prec):.4f}",
        'Accuracy': f"{np.mean(f_acc):.4f}"
    }
    ablation_rows.append(row_dict)
    print(f"{label:45s} | Feats={len(cols):2d} | Recall={row_dict['Recall']} | F1={row_dict['F1-Score']} | ROC-AUC={row_dict['ROC-AUC']} | PR-AUC={row_dict['PR-AUC']} | Brier={row_dict['Brier']}")

pd.DataFrame(ablation_rows).to_csv(os.path.join(OUTPUT_DIR, 'ablation_study_results.csv'), index=False)

# 6. SHAP Explanation Stability Analysis (RQ3)
print("\n" + "="*70)
print("SHAP EXPLANATION STABILITY ANALYSIS (FOLDS, SEEDS, MODELS, COHORTS)")
print("="*70)

active_cols = [c for c in feature_cols if c not in long_cols] # 20 active features in Cohort 1

# 6A. Intra-model Fold Stability (CatBoost across 5 CV folds)
fold_shap_importances = []
for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups=groups_train)):
    cb = CatBoostClassifier(iterations=150, depth=4, learning_rate=0.05, auto_class_weights='Balanced', verbose=False, random_seed=RANDOM_STATE)
    cb.fit(X_train.iloc[tr_idx], y_train[tr_idx])
    exp = shap.TreeExplainer(cb)
    sv = exp(X_train)
    mean_abs = np.abs(sv.values).mean(axis=0)
    # restrict to 20 active features to avoid tie distortion on 0-variance cols
    active_indices = [feature_cols.index(c) for c in active_cols]
    fold_shap_importances.append(mean_abs[active_indices])

fold_spearmans, fold_jaccards5, fold_jaccards10 = [], [], []
for i in range(len(fold_shap_importances)):
    for j in range(i+1, len(fold_shap_importances)):
        r, _ = spearmanr(fold_shap_importances[i], fold_shap_importances[j])
        fold_spearmans.append(r)
        top5_i = set(np.argsort(fold_shap_importances[i])[::-1][:5])
        top5_j = set(np.argsort(fold_shap_importances[j])[::-1][:5])
        fold_jaccards5.append(len(top5_i & top5_j) / len(top5_i | top5_j))
        top10_i = set(np.argsort(fold_shap_importances[i])[::-1][:10])
        top10_j = set(np.argsort(fold_shap_importances[j])[::-1][:10])
        fold_jaccards10.append(len(top10_i & top10_j) / len(top10_i | top10_j))

print(f"CatBoost 5-Fold Stability: Spearman rho = {np.mean(fold_spearmans):.4f} +/- {np.std(fold_spearmans):.4f} | Top-5 Jaccard = {np.mean(fold_jaccards5):.4f} | Top-10 Jaccard = {np.mean(fold_jaccards10):.4f}")

# 6B. Seed Stability (CatBoost across 5 random seeds)
seeds = [42, 100, 2026, 777, 999]
seed_shap_importances = []
active_indices = [feature_cols.index(c) for c in active_cols]
for s in seeds:
    cb = CatBoostClassifier(iterations=150, depth=4, learning_rate=0.05, auto_class_weights='Balanced', verbose=False, random_seed=s)
    cb.fit(X_train, y_train)
    exp = shap.TreeExplainer(cb)
    sv = exp(X_train)
    seed_shap_importances.append(np.abs(sv.values).mean(axis=0)[active_indices])

seed_spearmans, seed_jaccards5 = [], []
for i in range(len(seed_shap_importances)):
    for j in range(i+1, len(seed_shap_importances)):
        r, _ = spearmanr(seed_shap_importances[i], seed_shap_importances[j])
        seed_spearmans.append(r)
        t5_i = set(np.argsort(seed_shap_importances[i])[::-1][:5])
        t5_j = set(np.argsort(seed_shap_importances[j])[::-1][:5])
        seed_jaccards5.append(len(t5_i & t5_j) / len(t5_i | t5_j))

print(f"CatBoost 5-Seed Stability: Spearman rho = {np.mean(seed_spearmans):.4f} +/- {np.std(seed_spearmans):.4f} | Top-5 Jaccard = {np.mean(seed_jaccards5):.4f}")

# 6C. Cross-Model Attribution Consensus (CatBoost vs XGBoost vs LightGBM vs Random Forest)
tree_models = get_models(scale_pos_weight)
model_shap_vecs = {}
for mname in ['CatBoost', 'XGBoost', 'LightGBM', 'Random Forest']:
    m = tree_models[mname]
    m.fit(X_train, y_train)
    exp = shap.TreeExplainer(m)
    sv = exp(X_train)
    vals = sv.values[:, :, 1] if (len(sv.values.shape) == 3 and sv.values.shape[2] == 2) else sv.values
    model_shap_vecs[mname] = np.abs(vals).mean(axis=0)[active_indices]

for mname in ['XGBoost', 'LightGBM', 'Random Forest']:
    r, _ = spearmanr(model_shap_vecs['CatBoost'], model_shap_vecs[mname])
    t5_cb = set(np.argsort(model_shap_vecs['CatBoost'])[::-1][:5])
    t5_m = set(np.argsort(model_shap_vecs[mname])[::-1][:5])
    j5 = len(t5_cb & t5_m) / len(t5_cb | t5_m)
    t10_cb = set(np.argsort(model_shap_vecs['CatBoost'])[::-1][:10])
    t10_m = set(np.argsort(model_shap_vecs[mname])[::-1][:10])
    j10 = len(t10_cb & t10_m) / len(t10_cb | t10_m)
    print(f"Cross-Model Consensus (CatBoost vs {mname:13s}): Spearman rho = {r:.4f} | Top-5 Jaccard = {j5:.4f} | Top-10 Jaccard = {j10:.4f}")

# 6D. Cross-Cohort Explanation Stability (Historical Train n=380 vs Prospective Test n=356)
final_cb = CatBoostClassifier(iterations=150, depth=4, learning_rate=0.05, auto_class_weights='Balanced', verbose=False, random_seed=RANDOM_STATE)
final_cb.fit(X_train, y_train)
exp_cb = shap.TreeExplainer(final_cb)
sv_train = exp_cb(X_train)
sv_test = exp_cb(X_test)
vec_tr = np.abs(sv_train.values).mean(axis=0)[active_indices]
vec_te = np.abs(sv_test.values).mean(axis=0)[active_indices]
r_cohort, _ = spearmanr(vec_tr, vec_te)
t5_tr = set(np.argsort(vec_tr)[::-1][:5])
t5_te = set(np.argsort(vec_te)[::-1][:5])
j5_cohort = len(t5_tr & t5_te) / len(t5_tr | t5_te)
print(f"Cross-Cohort Stability (2025-2026 vs 2026-2027): Spearman rho = {r_cohort:.4f} | Top-5 Jaccard = {j5_cohort:.4f}")

# 7. Alert-Capacity & Budget Calibration Analysis (RQ4)
print("\n" + "="*70)
print("ADVISING ALERT-CAPACITY BUDGET ANALYSIS (OUT-OF-FOLD CATBOOST)")
print("="*70)
cb_oof_probs = oof_probabilities['CatBoost']
sorted_idx = np.argsort(cb_oof_probs)[::-1]
for pct in [10, 15, 20, 25, 30]:
    k = int(round(len(y_train) * (pct / 100.0)))
    alert_pred = np.zeros(len(y_train), dtype=int)
    alert_pred[sorted_idx[:k]] = 1
    rec_k = recall_score(y_train, alert_pred)
    prec_k = precision_score(y_train, alert_pred)
    f1_k = f1_score(y_train, alert_pred)
    tp_k = int(y_train[sorted_idx[:k]].sum())
    cutoff_p = cb_oof_probs[sorted_idx[k-1]]
    print(f"Alert Budget {pct:2d}% (k={k:3d} students, P>={cutoff_p:.3f}): Captures {tp_k:2d}/{pos_count} At-Risk | Recall={rec_k:.4f} ({rec_k*100:.1f}%) | Precision={prec_k:.4f} | F1={f1_k:.4f}")

# 8. Subgroup Fairness & Parity Evaluation (Out-of-Fold CatBoost)
print("\n" + "="*70)
print("SUBGROUP FAIRNESS & PARITY ANALYSIS (OUT-OF-FOLD CATBOOST)")
print("="*70)
cb_oof_preds = oof_predictions['CatBoost']
subgroups = [
    ('Sex: Female', train_df['is_female'] == 1),
    ('Sex: Male', train_df['is_female'] == 0),
    ('Socioeconomic: 4Ps Indigent', train_df['is_4ps_beneficiary'] == 1),
    ('Socioeconomic: Non-4Ps', train_df['is_4ps_beneficiary'] == 0),
    ('Locality: Distant (>25 km)', train_df['locality_is_distant'] == 1),
    ('Locality: Local/Nearby (<=25 km)', train_df['locality_is_distant'] == 0),
    ('First-Gen College Student', train_df['is_first_gen_college'] == 1),
    ('Continuing-Gen Student', train_df['is_first_gen_college'] == 0)
]
for sg_name, mask in subgroups:
    idx = np.where(mask.values)[0]
    y_sg = y_train[idx]
    pr_sg = cb_oof_preds[idx]
    pb_sg = cb_oof_probs[idx]
    tn, fp, fn, tp = confusion_matrix(y_sg, pr_sg, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    auc_sg = roc_auc_score(y_sg, pb_sg) if len(np.unique(y_sg)) > 1 else 0
    print(f"{sg_name:34s} (n={len(idx):3d}, pos={int(y_sg.sum()):2d}) | Recall(TPR)={tpr:.4f} | FPR={fpr:.4f} | ROC-AUC={auc_sg:.4f}")

# 9. Visualizations
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Fig 1: Benchmark & ROC Curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
metrics_plot_df = metrics_df.melt(
    id_vars=['Model'], 
    value_vars=['Recall (Mean)', 'F1-Score (Mean)', 'ROC-AUC (Mean)', 'PR-AUC (Mean)', 'Accuracy (Mean)'],
    var_name='Metric', value_name='Score'
)
metrics_plot_df['Metric'] = metrics_plot_df['Metric'].str.replace(' (Mean)', '', regex=False)

sns.barplot(data=metrics_plot_df, x='Model', y='Score', hue='Metric', palette='viridis', ax=axes[0])
axes[0].set_title('Cross-Validated Performance Benchmark (StratifiedGroupKFold, k=5)', fontsize=13, fontweight='bold', pad=12)
axes[0].set_ylim(0, 1.05)
axes[0].set_ylabel('Score (0.0 - 1.0)', fontsize=11)
axes[0].set_xlabel('Classification Algorithm', fontsize=11)
axes[0].legend(title='Metric', loc='lower right', frameon=True)
axes[0].tick_params(axis='x', rotation=15)

for name in models:
    fpr, tpr, _ = roc_curve(y_train, oof_probabilities[name])
    auc_val = results[name]['ROC-AUC'][0]
    axes[1].plot(fpr, tpr, lw=2.2, label=f"{name} (AUC = {auc_val:.3f})")

axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Chance Baseline (AUC = 0.500)')
axes[1].set_title('Out-of-Fold Cross-Validated ROC Curves', fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
axes[1].set_ylabel('True Positive Rate (Recall)', fontsize=11)
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].legend(loc='lower right', frameon=True)

plt.tight_layout()
benchmark_fig_path = os.path.join(OUTPUT_DIR, 'benchmark_comparison_roc.png')
plt.savefig(benchmark_fig_path, dpi=300, bbox_inches='tight')
plt.close()

# SHAP Plots on Final CatBoost Model
shap_values_pos = sv_train

# Print exact top 15 SHAP features for verification
mean_abs_all = np.abs(shap_values_pos.values).mean(axis=0)
print("\nExact Top 15 Mean |SHAP| Features:")
for rank_i, f_idx in enumerate(np.argsort(mean_abs_all)[::-1][:15], 1):
    print(f"  {rank_i:2d}. {feature_cols[f_idx]:25s}: {mean_abs_all[f_idx]:.4f}")

# Fig 2: SHAP Global Bar Plot
fig, ax = plt.subplots(figsize=(10, 7))
shap.plots.bar(shap_values_pos, max_display=15, show=False)
plt.title('Top 15 Predictive Risk Factors (Mean |SHAP| Importance)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
shap_bar_path = os.path.join(OUTPUT_DIR, 'shap_feature_importance_bar.png')
plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
plt.close()

# Fig 3: SHAP Beeswarm Plot
fig, ax = plt.subplots(figsize=(11, 7.5))
shap.plots.beeswarm(shap_values_pos, max_display=15, show=False)
plt.title('SHAP Global Risk Factor Attributions: CatBoost Model', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
shap_beeswarm_path = os.path.join(OUTPUT_DIR, 'shap_summary_beeswarm.png')
plt.savefig(shap_beeswarm_path, dpi=300, bbox_inches='tight')
plt.close()

# Fig 4: Student-Level Waterfall Plot (Unclipped with bbox_inches='tight')
train_probs = final_cb.predict_proba(X_train)[:, 1]
high_risk_indices = np.where((y_train == 1) & (train_probs > 0.75))[0]
case_idx = int(high_risk_indices[0])
case_pid = train_df.iloc[case_idx]['student_pid']
case_prob = train_probs[case_idx]

print(f"\nWaterfall Student Case: idx={case_idx}, PID={case_pid}, P={case_prob:.4f}, base_value={shap_values_pos.base_values[case_idx]:.4f}, fx={shap_values_pos.base_values[case_idx] + shap_values_pos.values[case_idx].sum():.4f}")
for f_idx in np.argsort(np.abs(shap_values_pos.values[case_idx]))[::-1][:10]:
    print(f"  {feature_cols[f_idx]:25s} (val={X_train.iloc[case_idx, f_idx]}): SHAP={shap_values_pos.values[case_idx][f_idx]:+.4f}")

fig, ax = plt.subplots(figsize=(10, 6.5))
shap.plots.waterfall(shap_values_pos[case_idx], max_display=12, show=False)
plt.title(f'Local SHAP Attribution: {case_pid} (P = {case_prob:.1%})', fontsize=12, fontweight='bold', pad=14)
plt.tight_layout()
shap_waterfall_path = os.path.join(OUTPUT_DIR, 'shap_waterfall_student_case.png')
plt.savefig(shap_waterfall_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
plt.close()

# Fig 5: Prospective Cohort Risk Distribution
prospective_probs = final_cb.predict_proba(X_test)[:, 1]
prospective_preds = (prospective_probs >= 0.5).astype(int)

def get_risk_tier(p):
    if p >= 0.60:
        return 'High Risk (Urgent Early Intervention)'
    elif p >= 0.35:
        return 'Moderate Risk (Adviser Monitoring)'
    else:
        return 'Low Risk (Satisfactory Standing)'

risk_tiers = [get_risk_tier(p) for p in prospective_probs]
tier_counts = pd.Series(risk_tiers).value_counts()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(prospective_probs, bins=25, kde=True, color='#2b5c8f', ax=axes[0])
axes[0].axvline(0.50, color='red', linestyle='--', lw=1.8, label='Decision Cutoff (0.50)')
axes[0].axvline(0.60, color='darkred', linestyle=':', lw=1.8, label='High Risk Threshold (0.60)')
axes[0].axvline(0.35, color='orange', linestyle=':', lw=1.8, label='Moderate Risk Threshold (0.35)')
axes[0].set_title('Predicted Risk Probability Distribution (2026-2027 Cohort)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted Probability P(At-Risk)', fontsize=11)
axes[0].set_ylabel('Number of Students', fontsize=11)
axes[0].legend(frameon=True)

tier_order = ['Low Risk (Satisfactory Standing)', 'Moderate Risk (Adviser Monitoring)', 'High Risk (Urgent Early Intervention)']
tier_df = pd.DataFrame({'Intervention Tier': tier_order, 'Count': [tier_counts.get(t, 0) for t in tier_order]})
palette_colors = {'Low Risk (Satisfactory Standing)': '#388e3c', 'Moderate Risk (Adviser Monitoring)': '#f57c00', 'High Risk (Urgent Early Intervention)': '#d32f2f'}

sns.barplot(data=tier_df, x='Intervention Tier', y='Count', hue='Intervention Tier', palette=palette_colors, legend=False, ax=axes[1])
axes[1].set_title('Early Intervention Distribution for Current Students', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Intervention Category', fontsize=11)
axes[1].set_ylabel('Student Count', fontsize=11)
axes[1].tick_params(axis='x', rotation=15)
for p in axes[1].patches:
    h = p.get_height()
    if h > 0:
        axes[1].annotate(f"{int(h)} ({h/len(risk_tiers)*100:.1f}%)",
                         (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='center', xytext=(0, 6), textcoords='offset points', fontweight='bold')

plt.tight_layout()
prospective_fig_path = os.path.join(OUTPUT_DIR, 'prospective_cohort_risk_distribution.png')
plt.savefig(prospective_fig_path, dpi=300, bbox_inches='tight')
plt.close()

print("\nALL EXPERIMENTS AND FIGURES COMPLETED SUCCESSFULLY!")
