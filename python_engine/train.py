import os
import sys

# Ensure UTF-8 output encoding on Windows
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, precision_score, accuracy_score, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CLEAN_DIR = os.path.join(ROOT_DIR, 'cleaned_dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'program_is_bsis', 'year_level', 'age', 'is_female',
    'distance_km', 'locality_is_campus_area', 'locality_is_distant',
    'is_4ps_beneficiary', 'income_bracket', 'has_philsys_id',
    'is_first_gen_college', 'is_dependent', 'has_disability', 'is_indigenous',
    'enrolled_units', 'course_count', 'has_lab_courses', 'lab_courses_count', 'is_underload', 'is_overload',
    'is_returning_student', 'prior_term_gpa', 'has_prior_gpa_recorded',
    'prior_deficiencies', 'prior_at_risk'
]

def load_data():
    dataset_path = os.path.join(CLEAN_DIR, 'student_risk_ml_dataset.csv')
    df = pd.read_csv(dataset_path)
    train_df = df[df['split_type'] == 'train_eval'].copy()
    
    # Fill any null prior_term_gpa with mean or flag
    train_df['prior_term_gpa'] = train_df['prior_term_gpa'].fillna(2.20)
    train_df['has_prior_gpa_recorded'] = train_df['has_prior_gpa_recorded'].fillna(0)
    
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df['at_risk'].astype(int).values
    groups = train_df['student_pid'].values
    return X_train, y_train, groups, train_df

def get_model_instance(name, params, scale_pos_weight, seed=42):
    iters = int(params.get('iterations', 150))
    depth = int(params.get('depth', 4))
    lr = float(params.get('learning_rate', 0.05))

    if name == 'CatBoost':
        return CatBoostClassifier(
            iterations=iters,
            depth=depth,
            learning_rate=lr,
            auto_class_weights='Balanced',
            verbose=False,
            random_seed=seed
        )
    elif name == 'XGBoost':
        return xgb.XGBClassifier(
            n_estimators=iters,
            max_depth=depth,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=seed,
            n_jobs=-1
        )
    elif name == 'LightGBM':
        return lgb.LGBMClassifier(
            n_estimators=iters,
            max_depth=depth,
            num_leaves=15,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            verbose=-1,
            n_jobs=-1
        )
    elif name == 'Random Forest':
        return RandomForestClassifier(
            n_estimators=iters,
            max_depth=depth + 2,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1
        )
    elif name == 'Logistic Regression':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                class_weight='balanced',
                C=0.5,
                max_iter=1000,
                random_state=seed
            ))
        ])
    else:
        raise ValueError(f"Unknown model: {name}")

def train_and_eval(model_choice='CatBoost', params=None, n_splits=5):
    if params is None:
        params = {}
    
    print(f"Loading institutional dataset from {CLEAN_DIR}...", flush=True)
    X, y, groups, full_df = load_data()
    
    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    scale_pos_weight = float(neg_count / pos_count)
    print(f"Data Loaded: {len(y)} observations ({pos_count} At-Risk [{pos_count/len(y)*100:.1f}%], {neg_count} Satisfactory). Scale pos weight: {scale_pos_weight:.2f}", flush=True)

    target_models = [model_choice] if model_choice != 'ALL' else ['CatBoost', 'XGBoost', 'LightGBM', 'Random Forest', 'Logistic Regression']
    all_results = {}

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for mname in target_models:
        print(f"\n--- Training {mname} with 5-Fold Stratified Group K-Fold (Grouping on student_pid) ---", flush=True)
        recalls, f1s, aucs, praucs, briers, precs, accs, specs = [], [], [], [], [], [], [], []
        
        for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            clf = get_model_instance(mname, params, scale_pos_weight)
            clf.fit(X_tr, y_tr)
            
            y_proba = clf.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            prec = precision_score(y_val, y_pred, zero_division=0)
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)
            prauc = average_precision_score(y_val, y_proba)
            brier = brier_score_loss(y_val, y_proba)
            
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0

            recalls.append(rec)
            f1s.append(f1)
            aucs.append(auc)
            praucs.append(prauc)
            briers.append(brier)
            precs.append(prec)
            accs.append(acc)
            specs.append(spec)
            print(f"  Fold {fold}/{n_splits} -> Recall: {rec*100:.2f}%, F1: {f1:.4f}, ROC-AUC: {auc:.4f}", flush=True)

        m_res = {
            'model_name': mname,
            'recall_mean': float(np.mean(recalls)),
            'recall_std': float(np.std(recalls)),
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s)),
            'roc_auc_mean': float(np.mean(aucs)),
            'roc_auc_std': float(np.std(aucs)),
            'pr_auc_mean': float(np.mean(praucs)),
            'pr_auc_std': float(np.std(praucs)),
            'brier_mean': float(np.mean(briers)),
            'precision_mean': float(np.mean(precs)),
            'accuracy_mean': float(np.mean(accs)),
            'specificity_mean': float(np.mean(specs))
        }
        all_results[mname] = m_res
        print(f"  => {mname} Summary: Recall={m_res['recall_mean']*100:.2f}% +/- {m_res['recall_std']*100:.2f}%, F1={m_res['f1_mean']:.4f}, ROC-AUC={m_res['roc_auc_mean']:.4f}", flush=True)

    # Train final full model on entire historical dataset
    primary_model_name = 'CatBoost' if 'CatBoost' in target_models else target_models[0]
    print(f"\nTraining final {primary_model_name} on all historical training data...", flush=True)
    final_model = get_model_instance(primary_model_name, params, scale_pos_weight)
    final_model.fit(X, y)

    # Save model artifact
    cbm_path = os.path.join(MODELS_DIR, 'catboost_model.cbm')
    joblib_path = os.path.join(MODELS_DIR, f"{primary_model_name.lower().replace(' ', '_')}.joblib")
    
    if primary_model_name == 'CatBoost':
        final_model.save_model(cbm_path)
    joblib.dump(final_model, joblib_path)
    print(f"Model successfully saved to {MODELS_DIR}", flush=True)

    # Compute Feature Importances (TreeSHAP)
    try:
        explainer = shap.TreeExplainer(final_model)
        sv = explainer(X)
        vals = sv.values[:, :, 1] if (len(sv.values.shape) == 3 and sv.values.shape[2] == 2) else sv.values
        mean_shap = np.abs(vals).mean(axis=0)
        shap_ranked = []
        for idx in np.argsort(mean_shap)[::-1]:
            shap_ranked.append({
                'feature': FEATURE_COLS[idx],
                'mean_abs_shap': float(mean_shap[idx])
            })
    except Exception as e:
        # Fallback to feature_importances_ if shap fails
        fi = getattr(final_model, 'feature_importances_', np.ones(len(FEATURE_COLS)))
        shap_ranked = [{'feature': FEATURE_COLS[i], 'mean_abs_shap': float(fi[i])} for i in np.argsort(fi)[::-1]]

    meta = {
        'status': 'SUCCESS',
        'primary_model': primary_model_name,
        'results': all_results,
        'top_shap_features': shap_ranked[:15],
        'feature_cols': FEATURE_COLS,
        'params': params,
        'n_train': len(y),
        'pos_count': pos_count,
        'neg_count': neg_count
    }

    meta_path = os.path.join(MODELS_DIR, 'model_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print("\n[TRAINING_COMPLETE_JSON]")
    print(json.dumps(meta))
    return meta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CatBoost', help='Model name or ALL')
    parser.add_argument('--iterations', type=int, default=150)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    args = parser.parse_args()

    train_and_eval(
        model_choice=args.model,
        params={
            'iterations': args.iterations,
            'depth': args.depth,
            'learning_rate': args.learning_rate
        }
    )

