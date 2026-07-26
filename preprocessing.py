"""
preprocessing.py
----------------
ECG signal preprocessing pipeline.
Handles both raw WFDB waveform files and pre-computed feature CSVs
(e.g. machine_measurements.csv from MIMIC-IV-ECG).
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def bandpass_filter(signal: np.ndarray, fs: float = 500.0,
                    lowcut: float = 0.5, highcut: float = 40.0) -> np.ndarray:
    """Butterworth bandpass filter (0.5–40 Hz standard for diagnostic ECG)."""
    nyq = fs / 2.0
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray, fs: float = 500.0,
                 freq: float = 60.0, Q: float = 30.0) -> np.ndarray:
    """60 Hz (or 50 Hz) powerline notch filter."""
    b, a = iirnotch(freq / (fs / 2.0), Q)
    return filtfilt(b, a, signal)


def preprocess_lead(signal: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """Apply full preprocessing chain to a single ECG lead."""
    signal = bandpass_filter(signal, fs)
    # Only apply 60Hz notch filter if sampling rate is high enough
    if fs > 125.0:
        signal = notch_filter(signal, fs)
    return signal


# ---------------------------------------------------------------------------
# R-peak detection (simple Pan-Tompkins-inspired threshold)
# ---------------------------------------------------------------------------

def detect_r_peaks(signal: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """
    Detect R-peaks using a derivative + threshold approach.
    For production use neurokit2.ecg_peaks() which is more robust.
    Returns array of sample indices.
    """
    # Differentiate and square
    diff = np.diff(signal)
    squared = diff ** 2

    # Moving window integration (~150ms)
    win = int(0.15 * fs)
    integrated = np.convolve(squared, np.ones(win) / win, mode='same')

    # Threshold: 60% of max
    threshold = 0.6 * np.max(integrated)
    above = integrated > threshold

    # Find rising edges = R-peak regions
    peaks = []
    refractory = int(0.2 * fs)  # 200ms refractory period
    last_peak = -refractory
    for i in range(1, len(above)):
        if above[i] and not above[i - 1]:
            # Find actual maximum in a window around this point
            start = max(0, i - win // 2)
            end = min(len(signal), i + win // 2)
            peak_idx = start + np.argmax(signal[start:end])
            if peak_idx - last_peak > refractory:
                peaks.append(peak_idx)
                last_peak = peak_idx

    return np.array(peaks)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_beats(signal: np.ndarray, r_peaks: np.ndarray,
                  fs: float = 500.0, window_ms: int = 600) -> np.ndarray:
    """
    Extract fixed-length windows centred on each R-peak.
    Returns array of shape (n_beats, window_samples).
    """
    half = int((window_ms / 1000.0) * fs / 2)
    beats = []
    for r in r_peaks:
        start = r - half
        end = r + half
        if start >= 0 and end < len(signal):
            beats.append(signal[start:end])
    return np.array(beats) if beats else np.empty((0, half * 2))


def median_beat(beats: np.ndarray) -> np.ndarray:
    """Compute the median beat across all segmented beats."""
    if len(beats) == 0:
        return np.array([])
    return np.median(beats, axis=0)


# ---------------------------------------------------------------------------
# CSV loader (machine_measurements.csv or custom uploads)
# ---------------------------------------------------------------------------

def load_machine_measurements(filepath: str) -> pd.DataFrame:
    """
    Load MIMIC-IV-ECG machine_measurements.csv.
    Selects clinically relevant columns and drops rows with all-NaN measurements.
    """
    cols_keep = [
        'subject_id', 'study_id', 'ecg_time',
        'rr_interval', 'p_onset', 'p_end',
        'qrs_onset', 'qrs_end', 'qrs_dur',
        't_end', 'qt_interval', 'qtc_interval',
        'heart_rate',
        # Machine report text columns
        'report_0', 'report_1', 'report_2', 'report_3', 'report_4',
        'report_5', 'report_6', 'report_7', 'report_8',
    ]
    df = pd.read_csv(filepath, usecols=lambda c: c in cols_keep, low_memory=False)
    measurement_cols = ['rr_interval', 'qrs_dur', 'qt_interval', 'qtc_interval', 'heart_rate']
    existing = [c for c in measurement_cols if c in df.columns]
    df = df.dropna(subset=existing, how='all')
    return df


def load_feature_csv(filepath: str) -> pd.DataFrame:
    """
    Load any feature CSV with a 'label' column (0=ischemic, 1=neurogenic).
    Used for custom uploads or the extracted feature dataset.
    """
    df = pd.read_csv(filepath)
    if 'label' not in df.columns:
        raise ValueError("CSV must contain a 'label' column (0=ischemic, 1=neurogenic)")
    # Drop non-numeric columns that would break the model imputer
    drop_cols = [c for c in df.columns if df[c].dtype == object and c != 'label']
    if drop_cols:
        print(f"Dropping non-numeric columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    return df


# ---------------------------------------------------------------------------
# Label extraction from machine report text
# ---------------------------------------------------------------------------

NEUROGENIC_KEYWORDS = [
    'cerebral', 'neurogenic', 'subarachnoid', 'diffuse st',
    'diffuse t', 'prolonged qt', 'qt prolongation', 'u wave',
    'autonomic', 'global st', 'non-specific st'
]

ISCHEMIC_KEYWORDS = [
    'stemi', 'nstemi', 'st elevation', 'st depression',
    'anterior infarct', 'inferior infarct', 'lateral infarct',
    'myocardial infarction', 'acute mi', 'ischemia',
    'anterolateral', 'anteroseptal', 'inferolateral'
]


def label_from_reports(row: pd.Series,
                        report_cols: Optional[list] = None) -> int:
    """
    Assign label from machine report text.
    Returns: 1 = neurogenic, 0 = ischemic, -1 = ambiguous/unknown
    """
    if report_cols is None:
        report_cols = [f'report_{i}' for i in range(9)]

    text = ' '.join([
        str(row[c]).lower() for c in report_cols if c in row.index and pd.notna(row[c])
    ])

    neuro_score = sum(kw in text for kw in NEUROGENIC_KEYWORDS)
    isch_score = sum(kw in text for kw in ISCHEMIC_KEYWORDS)

    if neuro_score > 0 and isch_score == 0:
        return 1
    elif isch_score > 0 and neuro_score == 0:
        return 0
    else:
        return -1  # ambiguous — exclude from training


def label_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply labelling to full dataframe and drop ambiguous rows."""
    df = df.copy()
    df['label'] = df.apply(label_from_reports, axis=1)
    labeled = df[df['label'] != -1].copy()
    print(f"Labeled: {len(labeled)} records ({labeled['label'].sum()} neurogenic, "
          f"{(labeled['label'] == 0).sum()} ischemic) from {len(df)} total")
    return labeled
