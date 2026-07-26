"""
mimic_demo_processor.py
-----------------------
Processes real MIMIC-IV-ECG Demo waveforms through the full pipeline.
Takes raw .dat/.hea files, extracts features, classifies each record,
and produces annotated ECG plots + a feature CSV.

Usage:
  python3 mimic_demo_processor.py
  python3 mimic_demo_processor.py --demo_path /custom/path --plot_n 10
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import preprocess_lead, detect_r_peaks, segment_beats, median_beat
from features import compute_st_features, compute_t_wave_features, compute_hrv_features, compute_qtc

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def find_all_records(demo_path):
    records = []
    base = Path(demo_path)
    for dat_file in sorted(base.rglob('*.dat')):
        record_path = str(dat_file).replace('.dat', '')
        records.append(record_path)
    print(f"Found {len(records)} WFDB records in {demo_path}")
    return records


def load_wfdb_record(record_path):
    try:
        import wfdb
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal
        if signals is None or np.all(np.isnan(signals)):
            return None, None, None
        signals = signals.T
        fs = float(record.fs)
        sig_names = record.sig_name
        return signals, fs, sig_names
    except Exception:
        return None, None, None


def extract_features(record_path):
    signals, fs, sig_names = load_wfdb_record(record_path)
    if signals is None:
        return None

    features = {'record_path': record_path}

    ii_idx = 1 if signals.shape[0] > 1 else 0
    for j, name in enumerate(sig_names):
        if name.upper().replace(' ', '') == 'II':
            ii_idx = j
            break

    lead_ii = preprocess_lead(signals[ii_idx], fs)
    r_peaks = detect_r_peaks(lead_ii, fs)
    if len(r_peaks) < 2:
        return None

    rr_intervals = np.diff(r_peaks) / fs * 1000
    features['rr_ms'] = float(np.mean(rr_intervals))
    features['heart_rate'] = float(60000 / features['rr_ms'])
    features['hr_std'] = float(np.std(60000 / rr_intervals))

    if len(r_peaks) >= 3:
        sd = np.diff(rr_intervals)
        features['rmssd'] = float(np.sqrt(np.mean(sd ** 2)))
        features['sdnn'] = float(np.std(rr_intervals))
        features['pnn50'] = float(np.mean(np.abs(sd) > 50) * 100)
    else:
        features['rmssd'] = np.nan
        features['sdnn'] = np.nan
        features['pnn50'] = np.nan

    for i, lead in enumerate(LEAD_NAMES):
        matched_idx = None
        for j, name in enumerate(sig_names):
            if name.upper().replace(' ', '') == lead.upper().replace(' ', ''):
                matched_idx = j
                break
        if matched_idx is None or matched_idx >= signals.shape[0]:
            for col in ['st_j0','st_j20','st_j60','st_slope','t_peak','t_peak_idx','t_area','t_asymmetry']:
                features[f'{lead}_{col}'] = np.nan
            continue

        sig = preprocess_lead(signals[matched_idx], fs)
        beats = segment_beats(sig, r_peaks, fs)
        if len(beats) == 0:
            for col in ['st_j0','st_j20','st_j60','st_slope','t_peak','t_peak_idx','t_area','t_asymmetry']:
                features[f'{lead}_{col}'] = np.nan
            continue

        med = median_beat(beats)
        r_center = len(med) // 2
        st = compute_st_features(med, r_center, fs)
        for k, v in st.items():
            features[f'{lead}_{k}'] = v
        qrs_end = r_center + int(0.06 * fs)
        t_end = min(r_center + int(0.32 * fs), len(med))
        tw = compute_t_wave_features(med, qrs_end, t_end)
        for k, v in tw.items():
            features[f'{lead}_{k}'] = v

    qt_approx = features['rr_ms'] * 0.42
    qtc = compute_qtc(qt_approx, features['rr_ms'])
    features['qtc_ms'] = qtc
    features['qtc_prolonged'] = int(qtc > 450) if not np.isnan(qtc) else 0
    features['qtc_markedly_prolonged'] = int(qtc > 500) if not np.isnan(qtc) else 0
    features['qrs_dur_ms'] = np.nan
    features['wide_qrs'] = 0

    return features


def process_all(demo_path, max_records=None, plot_first_n=5, output_dir='outputs'):
    Path(output_dir).mkdir(exist_ok=True)
    plot_dir = Path(output_dir) / 'mimic_demo_plots'
    plot_dir.mkdir(exist_ok=True)

    records = find_all_records(demo_path)
    if max_records:
        records = records[:max_records]

    model_obj = None
    try:
        from model import load_model
        model_obj = load_model()
        print("Model loaded — will classify each record")
    except Exception:
        print("No model found — extracting features only")

    all_features = []
    plotted = 0
    failed = 0

    print(f"\nProcessing {len(records)} records...")
    for i, record_path in enumerate(records):
        if i % 50 == 0 and i > 0:
            print(f"  {i}/{len(records)} done ({failed} failed)...")

        features = extract_features(record_path)
        if features is None:
            failed += 1
            continue

        classification_result = {
            'prediction': 'Unknown',
            'confidence': 0.0,
            'prob_neurogenic': 0.0,
            'prob_ischemic': 0.0,
            'annotations': {},
            'qtc_ms': features.get('qtc_ms', np.nan),
            'heart_rate': features.get('heart_rate', np.nan),
            'rmssd': features.get('rmssd', np.nan),
        }

        if model_obj is not None:
            try:
                from model import predict
                feat_row = {k: features.get(k, np.nan) for k in model_obj['feature_names']}
                result = predict(model_obj, pd.DataFrame([feat_row]))
                features['prediction'] = result['prediction'].iloc[0]
                features['prob_neurogenic'] = float(result['prob_neurogenic'].iloc[0])
                features['prob_ischemic'] = float(result['prob_ischemic'].iloc[0])
                features['confidence'] = float(result['confidence'].iloc[0])
                classification_result.update({
                    'prediction': features['prediction'],
                    'confidence': features['confidence'],
                    'prob_neurogenic': features['prob_neurogenic'],
                    'prob_ischemic': features['prob_ischemic'],
                })
            except Exception:
                pass

        if plotted < plot_first_n:
            try:
                from waveform_analyzer import plot_waveform_analysis
                signals, fs, sig_names = load_wfdb_record(record_path)
                if signals is not None:
                    out_path = str(plot_dir / f'{Path(record_path).name}_ecg.png')
                    plot_waveform_analysis(
                        signals=signals[:min(12, signals.shape[0])],
                        fs=fs,
                        lead_names=list(sig_names)[:12],
                        classification_result=classification_result,
                        output_path=out_path,
                        title=f'MIMIC-IV-ECG Demo — {Path(record_path).name}'
                    )
                    print(f"  Plot saved: {Path(out_path).name}")
                    plotted += 1
            except Exception as e:
                print(f"  Plot failed: {e}")

        all_features.append(features)

    if all_features:
        df = pd.DataFrame(all_features)
        out_csv = str(Path(output_dir) / 'mimic_demo_features.csv')
        df.to_csv(out_csv, index=False)
        print(f"\nDone. {len(all_features)} records processed, {failed} failed.")
        print(f"Feature CSV: {out_csv}")
        print(f"Plots: {plot_dir}")

        if 'prediction' in df.columns:
            print(f"\nClassification breakdown:")
            for label, count in df['prediction'].value_counts().items():
                print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")

        print(f"\nKey metrics (mean across all records):")
        for col in ['qtc_ms', 'heart_rate', 'rmssd']:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals):
                    print(f"  {col}: {vals.mean():.1f} ± {vals.std():.1f}")
        return df
    else:
        print("No records processed.")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_path', type=str,
        default='/Users/Abhiram/Downloads/physionet.org/files/mimic-iv-ecg-demo/0.1/files')
    parser.add_argument('--max_records', type=int, default=None)
    parser.add_argument('--plot_n', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    print("MIMIC-IV-ECG Demo Processor")
    print(f"Path: {args.demo_path}")
    process_all(args.demo_path, args.max_records, args.plot_n, args.output_dir)
