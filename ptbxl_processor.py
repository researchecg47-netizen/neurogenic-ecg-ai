"""
ptbxl_processor.py
------------------
Processes PTB-XL waveforms through the full pipeline.
Extracts features from real 12-lead ECGs, maps SCP-ECG codes
to ischemic/normal labels, and produces a labeled CSV ready
for model retraining.

Usage:
  python3 ptbxl_processor.py
  python3 ptbxl_processor.py --ptbxl_path /custom/path --sampling 100
"""

import sys
import argparse
import ast
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import preprocess_lead, detect_r_peaks, segment_beats, median_beat
from features import compute_st_features, compute_t_wave_features, compute_qtc

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# SCP-ECG codes for ischemic classification
# Source: PTB-XL paper (Wagner et al. 2020) and scp_statements.csv
ISCHEMIC_CODES = {
    'MI', 'STEMI', 'NSTEMI', 'ISC_', 'ISCA', 'ISCAN',
    'ISCAL', 'ISCIN', 'ISCIL', 'ISCAS', 'ISCLA', 'ISCLAT',
    'IMI', 'AMI', 'ALMI', 'INJAS', 'INJAL', 'INJIN',
    'INJLA', 'LMI', 'ILMI'
}

NORMAL_CODES = {'NORM', 'SR'}

# Neurogenic-like codes — diffuse repolarization changes
# These are NOT confirmed neurogenic but show similar ECG patterns
NEUROGENIC_LIKE_CODES = {
    'NDT',   # non-diagnostic T abnormalities
    'DIG',   # digitalis effect (diffuse repolarization)
    'LNGQT', # long QT
    'TAB_',  # T-wave abnormality
}


def parse_scp_codes(scp_str):
    """Parse the scp_codes column from ptbxl_database.csv."""
    try:
        if pd.isna(scp_str):
            return {}
        return ast.literal_eval(scp_str)
    except Exception:
        return {}


def assign_label(scp_codes: dict):
    """
    Assign label from SCP codes.
    Returns: 0=ischemic, 1=neurogenic-like, 2=normal, -1=ambiguous
    """
    codes = set(scp_codes.keys())

    has_ischemic = bool(codes & ISCHEMIC_CODES)
    has_normal = bool(codes & NORMAL_CODES)
    has_neuro_like = bool(codes & NEUROGENIC_LIKE_CODES)

    if has_ischemic and not has_neuro_like:
        return 0  # ischemic
    elif has_neuro_like and not has_ischemic:
        return 1  # neurogenic-like (diffuse repolarization)
    elif has_normal and not has_ischemic and not has_neuro_like:
        return 2  # normal
    else:
        return -1  # ambiguous — exclude


def extract_features(record_path: str, fs: float):
    """Extract features from a single PTB-XL WFDB record."""
    try:
        import wfdb
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal
        if signals is None or np.all(np.isnan(signals)):
            return None
        signals = signals.T  # (n_leads, n_samples)
        sig_names = record.sig_name
    except Exception:
        return None

    features = {'record_path': record_path}

    # R-peak detection on lead II
    ii_idx = 1
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

    # Per-lead ST and T-wave features
    for i, lead in enumerate(LEAD_NAMES):
        matched_idx = None
        for j, name in enumerate(sig_names):
            if name.upper().replace(' ', '') == lead.upper().replace(' ', ''):
                matched_idx = j
                break
        if matched_idx is None or matched_idx >= signals.shape[0]:
            for col in ['st_j0','st_j20','st_j60','st_slope',
                        't_peak','t_peak_idx','t_area','t_asymmetry']:
                features[f'{lead}_{col}'] = np.nan
            continue

        sig = preprocess_lead(signals[matched_idx], fs)
        beats = segment_beats(sig, r_peaks, fs)
        if len(beats) == 0:
            for col in ['st_j0','st_j20','st_j60','st_slope',
                        't_peak','t_peak_idx','t_area','t_asymmetry']:
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


def process_ptbxl(ptbxl_path: str, sampling: int = 100,
                  max_records: int = None, output_dir: str = 'outputs'):
    """
    Main processor. Loads PTB-XL metadata, assigns labels,
    extracts waveform features, and saves labeled CSV.
    """
    Path(output_dir).mkdir(exist_ok=True)
    ptbxl_path = Path(ptbxl_path)

    print("Loading PTB-XL metadata...")
    db = pd.read_csv(ptbxl_path / 'ptbxl_database.csv')
    print(f"Total records in database: {len(db)}")

    # Assign labels
    db['scp_parsed'] = db['scp_codes'].apply(parse_scp_codes)
    db['label'] = db['scp_parsed'].apply(assign_label)

    label_counts = db['label'].value_counts()
    print(f"\nLabel distribution:")
    print(f"  Ischemic (0):        {label_counts.get(0, 0)}")
    print(f"  Neurogenic-like (1): {label_counts.get(1, 0)}")
    print(f"  Normal (2):          {label_counts.get(2, 0)}")
    print(f"  Ambiguous (-1):      {label_counts.get(-1, 0)}")

    # Keep only ischemic and neurogenic-like for binary classification
    usable = db[db['label'].isin([0, 1])].copy()
    print(f"\nUsable records (ischemic + neurogenic-like): {len(usable)}")

    if max_records:
        usable = usable.head(max_records)
        print(f"Limited to {max_records} records for this run")

    fs = 100.0 if sampling == 100 else 500.0
    records_folder = 'records100' if sampling == 100 else 'records500'

    all_features = []
    failed = 0

    print(f"\nExtracting features from {len(usable)} records at {sampling}Hz...")
    for i, (_, row) in enumerate(usable.iterrows()):
        if i % 200 == 0 and i > 0:
            print(f"  {i}/{len(usable)} done ({failed} failed)...")

        # Build record path from filename_hr or filename_lr
        fname_col = 'filename_lr' if sampling == 100 else 'filename_hr'
        if fname_col not in db.columns:
            fname_col = 'filename_lr'

        record_rel = str(row.get(fname_col, '')).strip()
        if not record_rel:
            failed += 1
            continue

        record_path = str(ptbxl_path / record_rel)
        record_path = record_path.replace('.hea', '')

        features = extract_features(record_path, fs)
        if features is None:
            failed += 1
            continue

        features['label'] = int(row['label'])
        features['patient_id'] = row.get('patient_id', '')
        features['age'] = row.get('age', np.nan)
        features['sex'] = row.get('sex', np.nan)
        features['scp_codes'] = str(row.get('scp_codes', ''))

        all_features.append(features)

    if not all_features:
        print("No records processed successfully.")
        return None

    df = pd.DataFrame(all_features)

    # Binary label: ischemic=0, neurogenic-like=1
    out_csv = str(Path(output_dir) / 'ptbxl_features.csv')
    df.to_csv(out_csv, index=False)

    print(f"\nDone.")
    print(f"  Records processed: {len(df)}")
    print(f"  Records failed: {failed}")
    print(f"  Ischemic: {(df['label']==0).sum()}")
    print(f"  Neurogenic-like: {(df['label']==1).sum()}")
    print(f"  Feature CSV: {out_csv}")

    print(f"\nKey metrics:")
    for col in ['qtc_ms', 'heart_rate', 'rmssd']:
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals):
                print(f"  {col}: {vals.mean():.1f} +/- {vals.std():.1f}")

    print(f"\nNext step — retrain the model on these features:")
    print(f"  python3 pipeline.py --mode upload --file {out_csv}")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PTB-XL Waveform Processor')
    parser.add_argument('--ptbxl_path', type=str,
        default='/Users/Abhiram/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
    parser.add_argument('--sampling', type=int, default=100,
        choices=[100, 500],
        help='Sampling rate: 100Hz (faster) or 500Hz (full resolution)')
    parser.add_argument('--max_records', type=int, default=None,
        help='Limit records for a quick test run (e.g. 500)')
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    print("PTB-XL Waveform Processor")
    print(f"Path: {args.ptbxl_path}")
    print(f"Sampling: {args.sampling}Hz")

    process_ptbxl(
        ptbxl_path=args.ptbxl_path,
        sampling=args.sampling,
        max_records=args.max_records,
        output_dir=args.output_dir
    )
