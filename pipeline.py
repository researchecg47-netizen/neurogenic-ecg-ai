"""
pipeline.py
-----------
Main orchestrator. Run this file directly.

Usage:
  python pipeline.py --mode train_synthetic     # Train on synthetic data (works now)
  python pipeline.py --mode train_csv --file data/my_features.csv
  python pipeline.py --mode upload --file data/new_batch.csv  # Incremental
  python pipeline.py --mode evaluate --file data/test.csv
  python pipeline.py --mode predict --file data/new_ecgs.csv
  python pipeline.py --mode ablation
  python pipeline.py --mode list_models
"""

import argparse
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Local modules
from features import generate_synthetic_dataset, features_from_machine_measurements
from preprocessing import load_machine_measurements, load_feature_csv, label_dataset
from model import (train, evaluate, compute_shap, lead_ablation_study,
                   save_model, load_model, list_saved_models,
                   incremental_train, predict, prepare_X_y)


def mode_train_synthetic():
    """Train on synthetic data. Works immediately, no real data needed."""
    print("Generating synthetic ECG feature dataset...")
    df = generate_synthetic_dataset(n_neurogenic=300, n_ischemic=300)
    print(f"Synthetic dataset: {len(df)} records, {df.shape[1]-1} features")
    print(f"Class balance: {df['label'].value_counts().to_dict()}")

    train_df, test_df = train_test_split(df, test_size=0.2,
                                          stratify=df['label'], random_state=42)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # Train
    result = train(train_df, lead_set='12lead')

    # Evaluate
    X_test, y_test = prepare_X_y(test_df, '12lead')
    metrics = evaluate(result['model'], X_test, y_test, result['feature_names'])
    result['test_metrics'] = metrics

    # SHAP
    print("\nComputing SHAP values...")
    shap_result = compute_shap(result['model'], X_test, result['feature_names'])
    result['shap'] = shap_result

    # Save
    save_model(result)

    # Save test data for future evaluation
    test_df.to_csv(Path(__file__).parent / 'data' / 'test_set.csv', index=False)
    print("\nTest set saved to data/test_set.csv (do not retrain on this)")

    # Summary JSON
    summary = {
        'mode': 'synthetic',
        'n_train': len(train_df),
        'n_test': len(test_df),
        'cv_auc': result['cv_auc_mean'],
        'test_auc': metrics['auc_roc'],
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'top_5_features': list(shap_result['top_features'].keys())[:5],
    }
    out_path = Path(__file__).parent / 'outputs' / 'training_summary.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {out_path}")
    return summary


def mode_train_csv(filepath: str):
    """Train on a real feature CSV with a 'label' column."""
    print(f"Loading: {filepath}")
    df = load_feature_csv(filepath)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    if 'label' not in df.columns:
        print("ERROR: CSV must have a 'label' column (0=ischemic, 1=neurogenic)")
        sys.exit(1)

    train_df, test_df = train_test_split(df, test_size=0.2,
                                          stratify=df['label'], random_state=42)
    result = train(train_df, lead_set='12lead')
    X_test, y_test = prepare_X_y(test_df)
    metrics = evaluate(result['model'], X_test, y_test, result['feature_names'])
    shap_result = compute_shap(result['model'], X_test, result['feature_names'])
    result['test_metrics'] = metrics
    result['shap'] = shap_result
    save_model(result)
    return metrics


def mode_upload(filepath: str):
    """
    Upload new labeled data and incrementally retrain.
    The new data is merged with all previously uploaded data.
    """
    print(f"Loading new data: {filepath}")

    # Auto-detect file type
    if filepath.endswith('.csv'):
        try:
            df = load_feature_csv(filepath)
            print(f"Loaded as feature CSV: {len(df)} rows")
        except ValueError:
            # Might be machine_measurements.csv — try labeling it
            print("No 'label' column found. Trying to label from report text...")
            raw = pd.read_csv(filepath, low_memory=False)
            labeled = label_dataset(raw)
            df = features_from_machine_measurements(labeled)
            print(f"Labeled and extracted features: {len(df)} rows")
    else:
        print("Only CSV files supported for upload currently.")
        sys.exit(1)

    result = incremental_train(df)
    print(f"\nIncremental training complete.")
    print(f"New model CV AUC: {result['cv_auc_mean']:.4f}")
    return result


def mode_evaluate(filepath: str):
    """Evaluate the current latest model on a test CSV."""
    print(f"Loading test data: {filepath}")
    df = load_feature_csv(filepath)
    model_obj = load_model()
    X, y = prepare_X_y(df)
    metrics = evaluate(model_obj['model'], X, y, model_obj['feature_names'])
    return metrics


def mode_predict(filepath: str):
    """Run inference on new ECG features (no labels required)."""
    print(f"Loading data for inference: {filepath}")
    df = pd.read_csv(filepath)
    model_obj = load_model()
    results = predict(model_obj, df)
    out_path = Path(__file__).parent / 'outputs' / 'predictions.csv'
    results.to_csv(out_path, index=False)
    print(f"\nPredictions saved: {out_path}")
    print(results.head(10).to_string())
    return results


def mode_ablation():
    """Run lead ablation study on synthetic data (or combined_training_data.csv if present)."""
    data_path = Path(__file__).parent / 'data' / 'combined_training_data.csv'
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"Using combined training data: {len(df)} rows")
    else:
        print("No real data found. Using synthetic data for ablation.")
        df = generate_synthetic_dataset(n_neurogenic=300, n_ischemic=300)

    print("\nRunning lead ablation study...")
    ablation_results = lead_ablation_study(df)
    print("\nResults:")
    print(ablation_results.to_string(index=False))
    out_path = Path(__file__).parent / 'outputs' / 'ablation_results.csv'
    ablation_results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return ablation_results


def mode_list_models():
    """List all saved model versions."""
    models = list_saved_models()
    if models.empty:
        print("No saved models found.")
    else:
        print("\nSaved models:")
        print(models.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neurogenic ECG Classifier Pipeline')
    parser.add_argument('--mode', required=True,
                        choices=['train_synthetic', 'train_csv', 'upload',
                                 'evaluate', 'predict', 'ablation', 'list_models'])
    parser.add_argument('--file', type=str, default=None,
                        help='Path to CSV file (for train_csv, upload, evaluate, predict)')

    args = parser.parse_args()

    if args.mode == 'train_synthetic':
        mode_train_synthetic()
    elif args.mode == 'train_csv':
        if not args.file:
            print("--file required for train_csv mode")
            sys.exit(1)
        mode_train_csv(args.file)
    elif args.mode == 'upload':
        if not args.file:
            print("--file required for upload mode")
            sys.exit(1)
        mode_upload(args.file)
    elif args.mode == 'evaluate':
        if not args.file:
            print("--file required for evaluate mode")
            sys.exit(1)
        mode_evaluate(args.file)
    elif args.mode == 'predict':
        if not args.file:
            print("--file required for predict mode")
            sys.exit(1)
        mode_predict(args.file)
    elif args.mode == 'ablation':
        mode_ablation()
    elif args.mode == 'list_models':
        mode_list_models()
