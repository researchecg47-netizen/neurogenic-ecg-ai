"""
features.py
-----------
ECG feature extraction.
Works on two input types:
  1. Raw signal arrays (after preprocessing) — full 60+ feature set
  2. machine_measurements.csv rows — subset of pre-computed features

All features validated against neurogenic/ischemic ECG literature
(see ECG comparison table in project plan).
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List


# ---------------------------------------------------------------------------
# Features from pre-computed machine_measurements (no raw signal needed)
# ---------------------------------------------------------------------------

def features_from_machine_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract model-ready features from MIMIC-IV machine_measurements.csv.
    These are the global (all-lead) measures the ECG machine computes.
    Returns a clean feature dataframe.
    """
    feature_cols = []
    out = pd.DataFrame()

    # QTc — key neurogenic marker (prolonged >500ms in stroke/SAH)
    if 'qtc_interval' in df.columns:
        out['qtc_ms'] = pd.to_numeric(df['qtc_interval'], errors='coerce')
        out['qtc_prolonged'] = (out['qtc_ms'] > 450).astype(int)
        out['qtc_markedly_prolonged'] = (out['qtc_ms'] > 500).astype(int)
        feature_cols += ['qtc_ms', 'qtc_prolonged', 'qtc_markedly_prolonged']

    # QT interval raw
    if 'qt_interval' in df.columns:
        out['qt_ms'] = pd.to_numeric(df['qt_interval'], errors='coerce')
        feature_cols.append('qt_ms')

    # RR interval and derived heart rate
    if 'rr_interval' in df.columns:
        out['rr_ms'] = pd.to_numeric(df['rr_interval'], errors='coerce')
        out['hr_from_rr'] = 60000.0 / out['rr_ms'].replace(0, np.nan)
        feature_cols += ['rr_ms', 'hr_from_rr']

    # Heart rate
    if 'heart_rate' in df.columns:
        out['heart_rate'] = pd.to_numeric(df['heart_rate'], errors='coerce')
        feature_cols.append('heart_rate')

    # QRS duration — wide QRS can indicate conduction abnormality
    if 'qrs_dur' in df.columns:
        out['qrs_dur_ms'] = pd.to_numeric(df['qrs_dur'], errors='coerce')
        out['wide_qrs'] = (out['qrs_dur_ms'] > 120).astype(int)
        feature_cols += ['qrs_dur_ms', 'wide_qrs']

    # QRS onset/end absolute positions (for relative timing features)
    for col in ['qrs_onset', 'qrs_end', 'p_onset', 'p_end', 't_end']:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors='coerce')
            feature_cols.append(col)

    # PR interval (p_end to qrs_onset)
    if 'p_end' in out.columns and 'qrs_onset' in out.columns:
        out['pr_interval'] = out['qrs_onset'] - out['p_end']
        feature_cols.append('pr_interval')

    # ST proxy: distance from QRS end to T end (crude but available)
    if 'qrs_end' in out.columns and 't_end' in out.columns:
        out['st_t_duration'] = out['t_end'] - out['qrs_end']
        feature_cols.append('st_t_duration')

    # Carry label if present
    if 'label' in df.columns:
        out['label'] = df['label'].values

    return out[feature_cols + (['label'] if 'label' in out.columns else [])]


# ---------------------------------------------------------------------------
# Features from raw ECG signals (full pipeline)
# ---------------------------------------------------------------------------

def compute_hrv_features(r_peaks: np.ndarray, fs: float = 500.0) -> Dict[str, float]:
    """
    HRV features from R-peak positions.
    Key for neurogenic cases: autonomic dysregulation shows in HRV.
    """
    if len(r_peaks) < 3:
        return {'rmssd': np.nan, 'sdnn': np.nan, 'pnn50': np.nan,
                'mean_rr': np.nan, 'hr_mean': np.nan, 'hr_std': np.nan}

    rr_intervals = np.diff(r_peaks) / fs * 1000.0  # ms
    successive_diff = np.diff(rr_intervals)

    rmssd = np.sqrt(np.mean(successive_diff ** 2))
    sdnn = np.std(rr_intervals)
    pnn50 = np.mean(np.abs(successive_diff) > 50) * 100.0
    mean_rr = np.mean(rr_intervals)
    hr_vals = 60000.0 / rr_intervals
    hr_mean = np.mean(hr_vals)
    hr_std = np.std(hr_vals)

    return {
        'rmssd': rmssd,
        'sdnn': sdnn,
        'pnn50': pnn50,
        'mean_rr': mean_rr,
        'hr_mean': hr_mean,
        'hr_std': hr_std,
    }


def compute_qtc(qt_ms: float, rr_ms: float, method: str = 'bazett') -> float:
    """
    Compute QTc. Bazett's formula: QTc = QT / sqrt(RR in seconds).
    Person 3 (cardio) validates this formula choice.
    """
    if np.isnan(qt_ms) or np.isnan(rr_ms) or rr_ms <= 0:
        return np.nan
    rr_sec = rr_ms / 1000.0
    if method == 'bazett':
        return qt_ms / np.sqrt(rr_sec)
    elif method == 'fridericia':
        return qt_ms / (rr_sec ** (1.0 / 3.0))
    return np.nan


def compute_st_features(beat: np.ndarray, r_idx: int,
                         fs: float = 500.0) -> Dict[str, float]:
    """
    Extract ST-segment features from a single median beat.
    Measures ST level at J+0, J+20ms, J+60ms relative to isoelectric baseline.

    Neurogenic: diffuse ST changes, not territory-specific.
    Ischemic: focal ST elevation/depression by lead.
    """
    j_point = r_idx + int(0.04 * fs)  # ~40ms after R (approximate J point)
    baseline_start = max(0, r_idx - int(0.05 * fs))
    baseline_end = max(0, r_idx - int(0.02 * fs))

    if baseline_end <= baseline_start or j_point >= len(beat):
        return {'st_j0': np.nan, 'st_j20': np.nan, 'st_j60': np.nan, 'st_slope': np.nan}

    baseline = np.mean(beat[baseline_start:baseline_end])

    j0 = beat[j_point] - baseline if j_point < len(beat) else np.nan
    j20_idx = j_point + int(0.02 * fs)
    j60_idx = j_point + int(0.06 * fs)
    j20 = beat[j20_idx] - baseline if j20_idx < len(beat) else np.nan
    j60 = beat[j60_idx] - baseline if j60_idx < len(beat) else np.nan

    # ST slope between J+20 and J+60
    if not (np.isnan(j20) or np.isnan(j60)):
        st_slope = (j60 - j20) / (0.04 * fs)
    else:
        st_slope = np.nan

    return {'st_j0': j0, 'st_j20': j20, 'st_j60': j60, 'st_slope': st_slope}


def compute_t_wave_features(beat: np.ndarray, qrs_end_idx: int,
                              t_end_idx: int) -> Dict[str, float]:
    """
    T-wave amplitude and morphology.
    Neurogenic: deep symmetric inversions ("cerebral T-waves").
    Ischemic: peaked then inverts in infarct zone only.
    """
    if qrs_end_idx >= t_end_idx or t_end_idx > len(beat):
        return {'t_peak': np.nan, 't_peak_idx': np.nan,
                't_area': np.nan, 't_asymmetry': np.nan}

    t_segment = beat[qrs_end_idx:t_end_idx]

    # Peak (could be positive or negative)
    max_val = np.max(t_segment)
    min_val = np.min(t_segment)
    t_peak = max_val if abs(max_val) >= abs(min_val) else min_val
    t_peak_rel_idx = np.argmax(np.abs(t_segment))

    # Area under T wave (signed — negative = inversion)
    t_area = np.trapezoid(t_segment) if hasattr(np, 'trapezoid') else np.trapz(t_segment)

    # Asymmetry: ratio of upstroke to downstroke width
    if t_peak_rel_idx > 0 and t_peak_rel_idx < len(t_segment) - 1:
        upstroke = t_peak_rel_idx
        downstroke = len(t_segment) - t_peak_rel_idx
        t_asymmetry = upstroke / max(downstroke, 1)
    else:
        t_asymmetry = np.nan

    return {
        't_peak': t_peak,
        't_peak_idx': float(t_peak_rel_idx),
        't_area': t_area,
        't_asymmetry': t_asymmetry,
    }


def extract_features_from_signal(signals_12lead: np.ndarray,
                                   r_peaks: np.ndarray,
                                   fs: float = 500.0) -> Dict[str, float]:
    """
    Full feature extraction from a 12-lead ECG signal array.
    signals_12lead: shape (12, n_samples) — leads in standard order:
      I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6

    Returns a flat feature dict ready for a pandas DataFrame row.
    """
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    features = {}

    # HRV from lead II R-peaks
    hrv = compute_hrv_features(r_peaks, fs)
    features.update(hrv)

    # Per-lead ST and T-wave features
    from preprocessing import segment_beats, median_beat

    for i, lead in enumerate(lead_names):
        if i >= signals_12lead.shape[0]:
            break
        signal = signals_12lead[i]
        beats = segment_beats(signal, r_peaks, fs)
        if len(beats) == 0:
            continue
        med_beat = median_beat(beats)
        beat_len = len(med_beat)
        r_center = beat_len // 2

        st_feats = compute_st_features(med_beat, r_center, fs)
        for k, v in st_feats.items():
            features[f'{lead}_{k}'] = v

        # Approximate QRS end and T end from signal timing
        qrs_end = r_center + int(0.06 * fs)
        t_end = r_center + int(0.3 * fs)
        t_feats = compute_t_wave_features(med_beat, qrs_end, min(t_end, beat_len))
        for k, v in t_feats.items():
            features[f'{lead}_{k}'] = v

    # QTc from mean RR
    if 'mean_rr' in features and not np.isnan(features['mean_rr']):
        # Approximate QT as mean beat length minus QRS
        qt_approx = features['mean_rr'] * 0.42  # rough population QT estimate
        features['qtc_bazett'] = compute_qtc(qt_approx, features['mean_rr'])
    else:
        features['qtc_bazett'] = np.nan

    return features


# ---------------------------------------------------------------------------
# Synthetic data generator (used until real data is loaded)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(n_neurogenic: int = 200,
                                 n_ischemic: int = 200,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic feature dataset that mimics expected real patterns.
    Based on known ECG differences from the project's clinical table.

    Neurogenic profile: prolonged QTc, diffuse ST, deep T-waves, AF/bradycardia
    Ischemic profile: focal ST elevation, peaked T-waves, normal QTc, VT tendency
    """
    rng = np.random.default_rng(seed)

    def make_group(n, label, qtc_mean, qtc_std, st_mean, st_std,
                   t_peak_mean, hr_mean, hr_std, rmssd_mean):
        data = {
            'label': [label] * n,
            'qtc_ms': rng.normal(qtc_mean, qtc_std, n).clip(300, 700),
            'qtc_prolonged': None,
            'qtc_markedly_prolonged': None,
            'qt_ms': rng.normal(qtc_mean * 0.85, qtc_std, n).clip(250, 600),
            'rr_ms': rng.normal(60000 / hr_mean, 80, n).clip(400, 1500),
            'heart_rate': rng.normal(hr_mean, hr_std, n).clip(30, 180),
            'qrs_dur_ms': rng.normal(95, 15, n).clip(60, 160),
            'wide_qrs': None,
            'rmssd': rng.normal(rmssd_mean, 10, n).clip(5, 150),
            'sdnn': rng.normal(rmssd_mean * 1.2, 12, n).clip(5, 200),
            'hr_std': rng.normal(hr_std, 3, n).clip(1, 40),
        }
        # ST features — 12 leads, neurogenic is diffuse, ischemic is focal
        for lead in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                     'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
            if label == 1:  # neurogenic: diffuse
                data[f'{lead}_st_j60'] = rng.normal(st_mean, st_std, n)
            else:  # ischemic: only anterior leads elevated for LAD pattern
                focal = st_mean if lead in ['V1', 'V2', 'V3', 'V4'] else 0.0
                data[f'{lead}_st_j60'] = rng.normal(focal, st_std * 0.5, n)
            data[f'{lead}_t_peak'] = rng.normal(t_peak_mean, 0.1, n)
            data[f'{lead}_t_area'] = rng.normal(t_peak_mean * 20, 5, n)

        df = pd.DataFrame(data)
        df['qtc_prolonged'] = (df['qtc_ms'] > 450).astype(int)
        df['qtc_markedly_prolonged'] = (df['qtc_ms'] > 500).astype(int)
        df['wide_qrs'] = (df['qrs_dur_ms'] > 120).astype(int)
        return df

    neuro = make_group(n_neurogenic, label=1,
                       qtc_mean=490, qtc_std=40,
                       st_mean=-0.15, st_std=0.12,
                       t_peak_mean=-0.3,
                       hr_mean=75, hr_std=20,
                       rmssd_mean=25)

    isch = make_group(n_ischemic, label=0,
                      qtc_mean=420, qtc_std=30,
                      st_mean=0.25, st_std=0.15,
                      t_peak_mean=0.4,
                      hr_mean=88, hr_std=18,
                      rmssd_mean=45)

    df = pd.concat([neuro, isch], ignore_index=True).sample(
        frac=1, random_state=seed).reset_index(drop=True)
    return df
