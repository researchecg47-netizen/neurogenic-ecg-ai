"""
model.py
--------
ML model: Random Forest + XGBoost soft-voting ensemble.
Supports:
  - Initial training
  - Incremental training (upload new data → retrain or partial fit)
  - Model persistence (save/load versioned .pkl files)
  - SHAP explainability
  - Lead ablation study (1-lead, 3-lead, 12-lead)
  - Full evaluation suite
"""

import os
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — using Random Forest only. Install with: pip install xgboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not installed — explainability disabled. Install with: pip install shap")

import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent / 'models'
MODEL_DIR.mkdir(exist_ok=True)

LEAD_GROUPS = {
    '12lead': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
    '3lead': ['I', 'II', 'V2'],
    '1lead': ['II'],
}


# ---------------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame, lead_set: str = '12lead') -> List[str]:
    """Return model feature columns (exclude label, id cols)."""
    exclude = {'label', 'subject_id', 'study_id', 'ecg_time'}
    all_feats = [c for c in df.columns if c not in exclude]

    if lead_set == '12lead':
        return all_feats

    # Filter to only include ST/T features for selected leads
    leads = LEAD_GROUPS[lead_set]
    other_feats = [c for c in all_feats if not any(
        c.startswith(f'{l}_') for l in LEAD_GROUPS['12lead'])]
    lead_feats = [c for c in all_feats if any(c.startswith(f'{l}_') for l in leads)]
    return other_feats + lead_feats


def prepare_X_y(df: pd.DataFrame, lead_set: str = '12lead') -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and labels."""
    feat_cols = get_feature_cols(df, lead_set)
    X = df[feat_cols].copy()
    y = df['label'].copy()
    return X, y


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_rf() -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            max_features='sqrt',
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ))
    ])


def build_xgb(scale_pos_weight: float = 1.0) -> Pipeline:
    if not HAS_XGB:
        raise ImportError("xgboost not installed")
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        ))
    ])


def build_ensemble(scale_pos_weight: float = 1.0):
    """Soft-voting ensemble — uses RF+XGB if xgboost available, RF-only otherwise."""
    rf = build_rf()
    if HAS_XGB:
        xgb_model = build_xgb(scale_pos_weight)
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb_model)],
            voting='soft',
            n_jobs=-1,
        )
        return ensemble
    else:
        print("Note: using Random Forest only (install xgboost for full ensemble)")
        return rf


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame,
          lead_set: str = '12lead',
          cv_folds: int = 5,
          model_name: str = 'ensemble') -> Dict:
    """
    Train the ensemble model with cross-validation.
    Returns a result dict with the fitted model, metrics, and feature names.
    """
    X, y = prepare_X_y(df, lead_set)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    spw = n_neg / max(n_pos, 1)

    print(f"\n{'='*50}")
    print(f"Training: {model_name} | lead_set: {lead_set}")
    print(f"Dataset: {len(df)} samples | {n_pos} neurogenic | {n_neg} ischemic")
    print(f"Features: {X.shape[1]}")
    print(f"Scale pos weight: {spw:.2f}")

    model = build_ensemble(scale_pos_weight=spw)

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"\n{cv_folds}-fold CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    # Fit on full training data
    model.fit(X, y)
    print("Model fitted on full dataset.")

    result = {
        'model': model,
        'feature_names': list(X.columns),
        'lead_set': lead_set,
        'cv_auc_mean': float(cv_aucs.mean()),
        'cv_auc_std': float(cv_aucs.std()),
        'n_samples': len(df),
        'n_neurogenic': n_pos,
        'n_ischemic': n_neg,
        'trained_at': datetime.now().isoformat(),
    }
    return result


# ---------------------------------------------------------------------------
# Evaluation (test set — touch once only)
# ---------------------------------------------------------------------------

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series,
             feature_names: Optional[List[str]] = None) -> Dict:
    """Full evaluation on held-out test set."""
    if feature_names:
        X_test = X_test[feature_names]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=['Ischemic', 'Neurogenic'],
                                    output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    f1_neuro = report['Neurogenic']['f1-score']
    f1_isch = report['Ischemic']['f1-score']

    metrics = {
        'auc_roc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1_neurogenic': f1_neuro,
        'f1_ischemic': f1_isch,
        'n_test': len(y_test),
        'confusion_matrix': cm.tolist(),
        'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
    }

    print(f"\n{'='*50}")
    print("TEST SET EVALUATION (touched once)")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}  (neurogenic recall)")
    print(f"Specificity: {specificity:.4f}  (ischemic recall)")
    print(f"PPV:         {ppv:.4f}")
    print(f"NPV:         {npv:.4f}")
    print(f"F1 Neuro:    {f1_neuro:.4f}")
    print(f"F1 Isch:     {f1_isch:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def compute_shap(model, X: pd.DataFrame,
                 feature_names: Optional[List[str]] = None,
                 max_samples: int = 500) -> Dict:
    """
    Compute SHAP values using the RF component of the ensemble.
    Returns summary dict with top features and raw shap values.
    """
    if not HAS_SHAP:
        print("SHAP not installed — using feature importances instead.")
        return compute_feature_importance_fallback(model, X, feature_names)

    if feature_names:
        X = X[feature_names]

    # Get RF pipeline — handle both ensemble and standalone RF
    if hasattr(model, 'named_estimators_'):
        rf_pipe = model.named_estimators_['rf']
    else:
        rf_pipe = model

    rf_clf = rf_pipe.named_steps['clf']
    X_transformed = rf_pipe.named_steps['imputer'].transform(X)
    X_transformed = rf_pipe.named_steps['scaler'].transform(X_transformed)
    X_df = pd.DataFrame(X_transformed, columns=X.columns)

    if len(X_df) > max_samples:
        X_df = X_df.sample(max_samples, random_state=42)

    explainer = shap.TreeExplainer(rf_clf)
    shap_values = explainer.shap_values(X_df)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=X_df.columns)
    top_features = feature_importance.nlargest(15)

    print(f"\nTop 10 SHAP features (neurogenic classification):")
    for feat, val in top_features.head(10).items():
        print(f"  {feat:<35} {val:.4f}")

    return {
        'shap_values': sv,
        'feature_names': list(X_df.columns),
        'top_features': top_features.to_dict(),
        'mean_abs_shap': mean_abs_shap.tolist(),
    }


def compute_feature_importance_fallback(model, X: pd.DataFrame,
                                         feature_names: Optional[List[str]] = None) -> Dict:
    """Use built-in RF feature importances when SHAP is unavailable."""
    if feature_names:
        X = X[feature_names]

    if hasattr(model, 'named_estimators_'):
        rf_pipe = model.named_estimators_['rf']
    else:
        rf_pipe = model

    importances = rf_pipe.named_steps['clf'].feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).nlargest(15)

    print(f"\nTop 10 features by RF importance (install shap for SHAP values):")
    for feat, val in feat_imp.head(10).items():
        print(f"  {feat:<35} {val:.4f}")

    return {
        'shap_values': None,
        'feature_names': list(X.columns),
        'top_features': feat_imp.to_dict(),
        'mean_abs_shap': importances.tolist(),
    }


# ---------------------------------------------------------------------------
# Lead ablation study
# ---------------------------------------------------------------------------

def lead_ablation_study(df: pd.DataFrame, cv_folds: int = 5) -> pd.DataFrame:
    """
    Compare model performance across 1-lead, 3-lead, 12-lead configurations.
    Key for wearable viability analysis.
    """
    results = []
    for lead_set in ['1lead', '3lead', '12lead']:
        X, y = prepare_X_y(df, lead_set)
        n_pos = int(y.sum())
        spw = (len(y) - n_pos) / max(n_pos, 1)
        model = build_ensemble(scale_pos_weight=spw)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        aucs = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        results.append({
            'lead_config': lead_set,
            'n_features': X.shape[1],
            'cv_auc_mean': aucs.mean(),
            'cv_auc_std': aucs.std(),
        })
        print(f"{lead_set}: AUC = {aucs.mean():.4f} ± {aucs.std():.4f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Model persistence — versioned saves
# ---------------------------------------------------------------------------

def save_model(result: Dict, name: str = 'neurogenic_ecg') -> Path:
    """
    Save model + metadata. Versioned by timestamp + data hash.
    Always keeps previous versions — you can roll back.
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_hash = hashlib.md5(
        f"{result['n_samples']}_{result['trained_at']}".encode()
    ).hexdigest()[:6]
    filename = MODEL_DIR / f"{name}_v{ts}_{data_hash}.pkl"

    save_obj = {
        'model': result['model'],
        'feature_names': result['feature_names'],
        'lead_set': result['lead_set'],
        'metadata': {
            'trained_at': result['trained_at'],
            'n_samples': result['n_samples'],
            'n_neurogenic': result['n_neurogenic'],
            'n_ischemic': result['n_ischemic'],
            'cv_auc_mean': result['cv_auc_mean'],
            'cv_auc_std': result['cv_auc_std'],
        }
    }

    with open(filename, 'wb') as f:
        pickle.dump(save_obj, f)

    # Update "latest" symlink
    latest = MODEL_DIR / f"{name}_latest.pkl"
    if latest.exists():
        latest.unlink()
    with open(latest, 'wb') as f:
        pickle.dump(save_obj, f)

    print(f"\nModel saved: {filename.name}")
    print(f"Latest pointer updated: {latest.name}")
    return filename


def load_model(path: Optional[str] = None, name: str = 'neurogenic_ecg') -> Dict:
    """
    Load model. If path=None, loads the latest version.
    """
    if path is None:
        path = MODEL_DIR / f"{name}_latest.pkl"
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print(f"Loaded model trained at {obj['metadata']['trained_at']}")
    print(f"  Samples: {obj['metadata']['n_samples']} | "
          f"CV AUC: {obj['metadata']['cv_auc_mean']:.4f}")
    return obj


def list_saved_models(name: str = 'neurogenic_ecg') -> pd.DataFrame:
    """List all saved model versions with metadata."""
    files = sorted(MODEL_DIR.glob(f"{name}_v*.pkl"))
    rows = []
    for f in files:
        try:
            with open(f, 'rb') as fh:
                obj = pickle.load(fh)
            rows.append({
                'filename': f.name,
                'trained_at': obj['metadata']['trained_at'],
                'n_samples': obj['metadata']['n_samples'],
                'cv_auc': obj['metadata']['cv_auc_mean'],
                'lead_set': obj['lead_set'],
            })
        except Exception:
            pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Incremental training (upload new data → merge → retrain)
# ---------------------------------------------------------------------------

def incremental_train(new_df: pd.DataFrame,
                       existing_data_path: Optional[str] = None,
                       save_combined: bool = True) -> Dict:
    """
    Add new labeled data and retrain from scratch on the combined dataset.
    This is the correct approach for tree-based models (no partial_fit).

    Steps:
    1. Load existing training data (if any)
    2. Merge with new data, deduplicating on subject_id+study_id if present
    3. Retrain full model
    4. Save new version

    Args:
        new_df: New labeled dataframe with 'label' column
        existing_data_path: Path to existing combined_data.csv
        save_combined: Whether to save the merged dataset for future use
    """
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    combined_path = data_dir / 'combined_training_data.csv'

    if existing_data_path:
        combined_path = Path(existing_data_path)

    if combined_path.exists():
        existing = pd.read_csv(combined_path)
        print(f"Loaded existing data: {len(existing)} rows")

        # Deduplicate if IDs are present
        if 'study_id' in new_df.columns and 'study_id' in existing.columns:
            new_ids = set(new_df['study_id'].astype(str))
            existing = existing[~existing['study_id'].astype(str).isin(new_ids)]
            print(f"After dedup: {len(existing)} existing + {len(new_df)} new rows")

        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()
        print(f"No existing data found. Training on {len(new_df)} new rows.")

    print(f"Combined dataset: {len(combined)} rows")

    if save_combined:
        combined.to_csv(combined_path, index=False)
        print(f"Combined data saved: {combined_path}")

    # Retrain
    result = train(combined, lead_set='12lead')
    save_model(result)
    return result


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model_obj: Dict, X: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference on new ECG features.
    Returns dataframe with prediction, confidence, and top contributing features.
    """
    feature_names = model_obj['feature_names']
    model = model_obj['model']

    # Align columns
    X_aligned = X.reindex(columns=feature_names, fill_value=np.nan)
    probs = model.predict_proba(X_aligned)
    preds = model.predict(X_aligned)

    results = pd.DataFrame({
        'prediction': ['Neurogenic' if p == 1 else 'Ischemic' for p in preds],
        'confidence': probs.max(axis=1),
        'prob_neurogenic': probs[:, 1],
        'prob_ischemic': probs[:, 0],
    })
    return results
