"""
run_shap.py
-----------
Run SHAP analysis on the current saved model.
Produces top feature importance rankings for the paper Discussion section.

Usage:
  python3 run_shap.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model import load_model, compute_shap, compute_feature_importance_fallback

def run_shap_analysis(data_path: str = 'data/combined_smote.csv',
                      max_samples: int = 1000,
                      output_dir: str = 'outputs'):

    Path(output_dir).mkdir(exist_ok=True)

    print("Loading model...")
    model_obj = load_model()
    model = model_obj['model']
    feature_names = model_obj['feature_names']

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    drop_cols = [c for c in df.columns if df[c].dtype == object and c != 'label']
    df = df.drop(columns=drop_cols)

    # Sample for speed
    neurogenic = df[df['label'] == 1].sample(min(max_samples//2, len(df[df['label']==1])), random_state=42)
    ischemic = df[df['label'] == 0].sample(min(max_samples//2, len(df[df['label']==0])), random_state=42)
    sample = pd.concat([neurogenic, ischemic]).sample(frac=1, random_state=42)

    X = sample.reindex(columns=feature_names, fill_value=np.nan)
    y = sample['label']

    print(f"Running SHAP on {len(X)} samples ({(y==1).sum()} neurogenic, {(y==0).sum()} ischemic)...")

    result = compute_shap(model, X, feature_names, max_samples=max_samples)

    # Save top features to CSV
    top_df = pd.DataFrame(list(result['top_features'].items()),
                          columns=['feature', 'mean_abs_shap'])
    top_df = top_df.sort_values('mean_abs_shap', ascending=False)
    out_path = f"{output_dir}/shap_top_features.csv"
    top_df.to_csv(out_path, index=False)
    print(f"\nTop 15 features saved to: {out_path}")

    print(f"\nFull top 15 SHAP feature ranking:")
    print(f"{'Rank':<6} {'Feature':<35} {'Mean |SHAP|':<12} {'Clinical meaning'}")
    print("-" * 80)

    clinical_map = {
        'qtc_ms': 'QTc interval — key neurogenic marker',
        'qtc_prolonged': 'QTc >450ms flag',
        'qtc_markedly_prolonged': 'QTc >500ms flag — severe prolongation',
        'heart_rate': 'Heart rate — autonomic instability',
        'rr_ms': 'RR interval — beat-to-beat variation',
        'rmssd': 'HRV — parasympathetic activity',
        'sdnn': 'HRV — overall autonomic variability',
        'hr_std': 'Heart rate variability',
    }

    for rank, (_, row) in enumerate(top_df.head(15).iterrows(), 1):
        feat = row['feature']
        val = row['mean_abs_shap']
        lead = feat.split('_')[0] if '_' in feat else ''
        feat_type = '_'.join(feat.split('_')[1:]) if '_' in feat else feat
        clinical = clinical_map.get(feat, f"{feat_type} in lead {lead}" if lead else feat)
        print(f"{rank:<6} {feat:<35} {val:<12.4f} {clinical}")

    return result

if __name__ == '__main__':
    run_shap_analysis()
