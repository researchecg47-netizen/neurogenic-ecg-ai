"""
waveform_analyzer.py
--------------------
ECG waveform analysis module.
Takes raw ECG signals and produces:
  - Filtered, clean waveform plots
  - R-peak annotations
  - ST segment markers per lead
  - T-wave morphology markers
  - Classification result overlaid on waveform
  - SHAP feature values linked back to waveform regions

Works with:
  - WFDB .hea/.dat files (MIMIC-IV format)
  - CSV files with raw signal columns (one column per lead)
  - Synthetic ECG generation for testing
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from preprocessing import preprocess_lead, detect_r_peaks, segment_beats, median_beat
from features import compute_st_features, compute_t_wave_features, compute_hrv_features

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Clinical color scheme
NEURO_COLOR  = '#185FA5'   # blue — neurogenic
ISCH_COLOR   = '#D85A30'   # coral — ischemic
ANNOT_COLOR  = '#1D9E75'   # teal — annotations
WARN_COLOR   = '#BA7517'   # amber — flagged features


# ---------------------------------------------------------------------------
# ECG generation (synthetic, clinically shaped)
# ---------------------------------------------------------------------------

def _gaussian(x, mu, sigma, amp):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def generate_ecg_beat(fs=500, duration_ms=800, beat_type='normal',
                       qt_scale=1.0, st_offset=0.0, t_invert=False,
                       t_amplitude=0.3):
    """Generate a single synthetic ECG beat with clinical morphology."""
    n = int(fs * duration_ms / 1000)
    t = np.linspace(0, duration_ms, n)
    signal = np.zeros(n)

    r_pos = duration_ms * 0.35

    # P wave
    signal += _gaussian(t, r_pos - 160, 25, 0.15)
    # Q wave
    signal += _gaussian(t, r_pos - 25, 8, -0.08)
    # R wave
    signal += _gaussian(t, r_pos, 12, 1.0)
    # S wave
    signal += _gaussian(t, r_pos + 28, 10, -0.15)

    # ST segment + T wave
    qt_end = r_pos + 40 + 200 * qt_scale
    st_mid = r_pos + 60
    t_peak_pos = r_pos + 40 + 140 * qt_scale
    t_amp = -t_amplitude if t_invert else t_amplitude

    # ST segment offset (elevation or depression)
    st_start = int((r_pos + 35) / duration_ms * n)
    st_end = int(t_peak_pos / duration_ms * n)
    if st_start < st_end < n:
        signal[st_start:st_end] += st_offset * np.linspace(1, 0.3, st_end - st_start)

    # T wave
    signal += _gaussian(t, t_peak_pos, 35 * qt_scale, t_amp)

    return signal


def generate_neurogenic_ecg(fs=500, n_beats=8, lead_name='II'):
    """
    Generate neurogenic ECG pattern:
    - Prolonged QT (scale 1.4-1.6)
    - Deep symmetric T-wave inversions across all leads
    - Diffuse mild ST depression
    - No territory-specific pattern
    """
    beat_duration = 900  # ms — slightly bradycardic
    t_amp = 0.4 if lead_name in ['V2','V3','V4','V5'] else 0.25
    beat = generate_ecg_beat(
        fs=fs, duration_ms=beat_duration,
        qt_scale=1.5, st_offset=-0.12,
        t_invert=True, t_amplitude=t_amp
    )
    # Add noise
    rng = np.random.default_rng(42)
    signal = np.tile(beat, n_beats) + rng.normal(0, 0.015, len(beat) * n_beats)
    return signal


def generate_ischemic_ecg(fs=500, n_beats=8, lead_name='II'):
    """
    Generate ischemic ECG pattern (anterior STEMI):
    - ST elevation focal to anterior leads (V1-V4)
    - Peaked T waves in affected leads
    - Normal QT
    - Territory-specific pattern
    """
    beat_duration = 750  # ms — slightly tachycardic
    anterior = lead_name in ['V1', 'V2', 'V3', 'V4']
    st_elev = 0.25 if anterior else -0.05
    t_amp = 0.5 if anterior else 0.2
    beat = generate_ecg_beat(
        fs=fs, duration_ms=beat_duration,
        qt_scale=1.05, st_offset=st_elev,
        t_invert=False, t_amplitude=t_amp
    )
    rng = np.random.default_rng(42)
    signal = np.tile(beat, n_beats) + rng.normal(0, 0.015, len(beat) * n_beats)
    return signal


# ---------------------------------------------------------------------------
# WFDB / CSV loaders
# ---------------------------------------------------------------------------

def load_wfdb_record(record_path: str, leads: Optional[List[str]] = None) -> Tuple[np.ndarray, float, List[str]]:
    """
    Load a WFDB record (.hea + .dat).
    Returns (signals [n_leads, n_samples], fs, lead_names).
    """
    try:
        import wfdb
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal.T  # shape: (n_leads, n_samples)
        fs = record.fs
        sig_names = record.sig_name
        if leads:
            indices = [sig_names.index(l) for l in leads if l in sig_names]
            signals = signals[indices]
            sig_names = [sig_names[i] for i in indices]
        return signals, float(fs), sig_names
    except ImportError:
        raise ImportError("wfdb not installed: pip install wfdb")
    except Exception as e:
        raise ValueError(f"Could not load WFDB record: {e}")


def load_csv_signals(filepath: str, fs: float = 500.0) -> Tuple[np.ndarray, float, List[str]]:
    """
    Load raw ECG signals from CSV.
    Each column should be a lead (named I, II, V1, etc.).
    Returns (signals [n_leads, n_samples], fs, lead_names).
    """
    df = pd.read_csv(filepath)
    lead_cols = [c for c in df.columns if c in LEAD_NAMES or c.startswith('lead')]
    if not lead_cols:
        # Assume all numeric columns are leads
        lead_cols = df.select_dtypes(include=np.number).columns.tolist()
    signals = df[lead_cols].values.T
    return signals, fs, lead_cols


# ---------------------------------------------------------------------------
# Feature extraction from raw waveform
# ---------------------------------------------------------------------------

def extract_waveform_features(signals: np.ndarray, fs: float,
                               lead_names: List[str]) -> Dict:
    """
    Full feature extraction from multi-lead raw ECG.
    Returns feature dict + per-lead annotation data for plotting.
    """
    features = {}
    annotations = {}

    # Use lead II for R-peak detection (most reliable)
    ii_idx = lead_names.index('II') if 'II' in lead_names else 0
    lead_ii = preprocess_lead(signals[ii_idx], fs)
    r_peaks = detect_r_peaks(lead_ii, fs)

    if len(r_peaks) < 2:
        return features, annotations

    # HRV
    hrv = compute_hrv_features(r_peaks, fs)
    features.update(hrv)

    rr_intervals = np.diff(r_peaks) / fs * 1000
    features['rr_ms'] = float(np.mean(rr_intervals))
    features['heart_rate'] = float(60000 / features['rr_ms'])

    # Per-lead analysis
    for i, lead in enumerate(lead_names):
        if i >= signals.shape[0]:
            continue
        sig = preprocess_lead(signals[i], fs)
        beats = segment_beats(sig, r_peaks, fs)
        if len(beats) == 0:
            continue
        med = median_beat(beats)
        beat_len = len(med)
        r_center = beat_len // 2

        st = compute_st_features(med, r_center, fs)
        qrs_end = r_center + int(0.06 * fs)
        t_end = min(r_center + int(0.32 * fs), beat_len)
        tw = compute_t_wave_features(med, qrs_end, t_end)

        for k, v in st.items():
            features[f'{lead}_{k}'] = v
        for k, v in tw.items():
            features[f'{lead}_{k}'] = v

        # Store annotation data for plotting
        annotations[lead] = {
            'median_beat': med,
            'r_center': r_center,
            'j_point': r_center + int(0.04 * fs),
            'st_j60': r_center + int(0.06 * fs),
            't_peak_idx': qrs_end + int(tw.get('t_peak_idx', 0) or 0),
            't_end': t_end,
            'st_level': st.get('st_j60', 0),
            't_peak': tw.get('t_peak', 0),
            'r_peaks': r_peaks,
            'full_signal': sig,
        }

    # QTc (Bazett)
    if features['rr_ms'] > 0:
        # Estimate QT from T-end of lead II
        if 'II' in annotations:
            ann = annotations['II']
            qt_samples = ann['t_end'] - (ann['r_center'] - int(0.04 * fs))
            qt_ms = qt_samples / fs * 1000
            features['qtc_ms'] = qt_ms / np.sqrt(features['rr_ms'] / 1000)
        else:
            features['qtc_ms'] = np.nan

    return features, annotations


# ---------------------------------------------------------------------------
# Classify from waveform
# ---------------------------------------------------------------------------

def classify_waveform(signals: np.ndarray, fs: float, lead_names: List[str],
                       model_obj: Dict) -> Dict:
    """
    Run full pipeline on raw ECG signals:
    extract features → classify → return result with feature values.
    """
    features, annotations = extract_waveform_features(signals, fs, lead_names)

    if not features:
        return {'error': 'Could not extract features — check signal quality'}

    feature_names = model_obj['feature_names']
    feat_row = {k: features.get(k, np.nan) for k in feature_names}
    X = pd.DataFrame([feat_row])

    from model import predict
    result = predict(model_obj, X)

    return {
        'prediction': result['prediction'].iloc[0],
        'confidence': float(result['confidence'].iloc[0]),
        'prob_neurogenic': float(result['prob_neurogenic'].iloc[0]),
        'prob_ischemic': float(result['prob_ischemic'].iloc[0]),
        'features': features,
        'annotations': annotations,
        'qtc_ms': features.get('qtc_ms', np.nan),
        'heart_rate': features.get('heart_rate', np.nan),
        'rmssd': features.get('rmssd', np.nan),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_waveform_analysis(signals: np.ndarray, fs: float,
                            lead_names: List[str],
                            classification_result: Dict,
                            output_path: str = 'outputs/ecg_analysis.png',
                            title: str = '') -> str:
    """
    Full 12-lead ECG plot with:
    - All leads displayed in standard layout
    - R-peak markers
    - ST segment annotations
    - T-wave markers
    - Classification result panel
    - Key metrics panel
    """
    Path(output_path).parent.mkdir(exist_ok=True)

    pred = classification_result.get('prediction', 'Unknown')
    conf = classification_result.get('confidence', 0)
    prob_n = classification_result.get('prob_neurogenic', 0)
    prob_i = classification_result.get('prob_ischemic', 0)
    annotations = classification_result.get('annotations', {})
    qtc = classification_result.get('qtc_ms', np.nan)
    hr = classification_result.get('heart_rate', np.nan)

    pred_color = NEURO_COLOR if pred == 'Neurogenic' else ISCH_COLOR

    # Standard 12-lead layout: 3 rows × 4 cols + result panel
    fig = plt.figure(figsize=(22, 14), facecolor='white')
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.3,
                           top=0.88, bottom=0.06, left=0.05, right=0.97)

    standard_layout = [
        ['I',   'aVR', 'V1', 'V4'],
        ['II',  'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6'],
    ]

    def plot_lead(ax, lead, signals, fs, annotations, pred_color):
        sig_idx = lead_names.index(lead) if lead in lead_names else None
        if sig_idx is None:
            ax.text(0.5, 0.5, f'{lead}\n(not available)',
                    ha='center', va='center', fontsize=9,
                    color='#aaa', transform=ax.transAxes)
            ax.axis('off')
            return

        ann = annotations.get(lead, {})
        full_sig = ann.get('full_signal', preprocess_lead(signals[sig_idx], fs))

        # Show 4 seconds
        show_samples = min(int(4 * fs), len(full_sig))
        t = np.arange(show_samples) / fs

        ax.plot(t, full_sig[:show_samples], color='#333', linewidth=0.9, zorder=3)

        # ECG grid
        ax.set_facecolor('#fafafa')
        for x in np.arange(0, 4.1, 0.2):
            ax.axvline(x, color='#ffcccc', linewidth=0.3, zorder=1)
        for y in np.arange(-1.5, 1.6, 0.5):
            ax.axhline(y, color='#ffcccc', linewidth=0.3, zorder=1)
        for x in np.arange(0, 4.1, 1.0):
            ax.axvline(x, color='#ffaaaa', linewidth=0.6, zorder=1)
        for y in np.arange(-1.5, 1.6, 0.5):
            ax.axhline(y, color='#ffaaaa', linewidth=0.6, zorder=2)

        # R-peak markers
        r_peaks = ann.get('r_peaks', np.array([]))
        visible_peaks = r_peaks[r_peaks < show_samples]
        if len(visible_peaks):
            ax.scatter(visible_peaks / fs, full_sig[visible_peaks],
                      color='#E24B4A', s=20, zorder=5, marker='^')

        # ST level annotation
        st_level = ann.get('st_level', 0)
        if not np.isnan(st_level) and abs(st_level) > 0.05:
            j60 = ann.get('st_j60', 0)
            if j60 < show_samples:
                xpos = j60 / fs
                ax.annotate(f'ST{st_level:+.2f}',
                           xy=(xpos, full_sig[j60]),
                           xytext=(xpos + 0.15, full_sig[j60] + 0.2),
                           fontsize=7, color=pred_color,
                           arrowprops=dict(arrowstyle='->', color=pred_color,
                                         lw=0.8),
                           zorder=6)

        # T-wave marker
        t_peak_idx = ann.get('t_peak_idx', 0)
        t_peak_val = ann.get('t_peak', 0)
        if t_peak_idx and t_peak_idx < show_samples and not np.isnan(t_peak_val):
            if abs(t_peak_val) > 0.15:
                marker = 'v' if t_peak_val < 0 else '^'
                ax.scatter([t_peak_idx / fs], [full_sig[t_peak_idx]],
                          color=ANNOT_COLOR, s=15, marker=marker, zorder=5)

        ax.set_xlim(0, 4)
        sig_range = np.max(full_sig[:show_samples]) - np.min(full_sig[:show_samples])
        pad = max(sig_range * 0.3, 0.2)
        ax.set_ylim(np.min(full_sig[:show_samples]) - pad,
                   np.max(full_sig[:show_samples]) + pad)
        ax.set_ylabel('mV', fontsize=7, color='#888')
        ax.tick_params(axis='both', labelsize=6, colors='#888')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#ddd')
        ax.spines['left'].set_color('#ddd')

        # Lead label
        ax.text(0.02, 0.92, lead, transform=ax.transAxes,
               fontsize=10, fontweight='bold', color='#333', zorder=7)

    # Plot all 12 leads
    for row_idx, row in enumerate(standard_layout):
        for col_idx, lead in enumerate(row):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            plot_lead(ax, lead, signals, fs, annotations, pred_color)

    # Bottom panel: classification result + metrics
    ax_result = fig.add_subplot(gs[3, :])
    ax_result.axis('off')

    # Result box
    result_box = dict(boxstyle='round,pad=0.6', facecolor=pred_color,
                     alpha=0.12, edgecolor=pred_color, linewidth=2)
    ax_result.text(0.01, 0.75,
                  f'Classification: {pred.upper()}',
                  transform=ax_result.transAxes,
                  fontsize=16, fontweight='bold', color=pred_color,
                  bbox=result_box, va='top')

    ax_result.text(0.01, 0.25,
                  f'Confidence: {conf:.1%}',
                  transform=ax_result.transAxes,
                  fontsize=12, color='#444', va='top')

    # Probability bar
    bar_x = 0.22
    bar_w = 0.25
    bar_h = 0.3
    bar_y = 0.35

    ax_result.add_patch(Rectangle((bar_x, bar_y), bar_w * prob_n, bar_h,
                                   transform=ax_result.transAxes,
                                   color=NEURO_COLOR, alpha=0.7, zorder=3))
    ax_result.add_patch(Rectangle((bar_x + bar_w * prob_n, bar_y),
                                   bar_w * prob_i, bar_h,
                                   transform=ax_result.transAxes,
                                   color=ISCH_COLOR, alpha=0.7, zorder=3))
    ax_result.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h,
                                   transform=ax_result.transAxes,
                                   fill=False, edgecolor='#ccc', linewidth=1))
    ax_result.text(bar_x, bar_y + bar_h + 0.08,
                  f'Neurogenic {prob_n:.1%}',
                  transform=ax_result.transAxes,
                  fontsize=9, color=NEURO_COLOR)
    ax_result.text(bar_x + bar_w - 0.01, bar_y + bar_h + 0.08,
                  f'Ischemic {prob_i:.1%}',
                  transform=ax_result.transAxes,
                  fontsize=9, color=ISCH_COLOR, ha='right')

    # Key metrics
    metrics = [
        ('QTc', f'{qtc:.0f} ms' if not np.isnan(qtc) else 'N/A',
         qtc > 450 if not np.isnan(qtc) else False),
        ('Heart rate', f'{hr:.0f} bpm' if not np.isnan(hr) else 'N/A', False),
        ('RMSSD', f'{classification_result.get("rmssd", float("nan")):.1f} ms'
         if not np.isnan(classification_result.get("rmssd", float("nan"))) else 'N/A', False),
    ]

    mx = 0.52
    for i, (label, val, flagged) in enumerate(metrics):
        xpos = mx + i * 0.16
        color = WARN_COLOR if flagged else '#333'
        ax_result.text(xpos, 0.85, label,
                      transform=ax_result.transAxes,
                      fontsize=9, color='#888', ha='center')
        ax_result.text(xpos, 0.45, val,
                      transform=ax_result.transAxes,
                      fontsize=13, fontweight='bold', color=color, ha='center')
        if flagged:
            ax_result.text(xpos, 0.1, '⚠ Prolonged',
                          transform=ax_result.transAxes,
                          fontsize=8, color=WARN_COLOR, ha='center')

    # Disclaimer
    ax_result.text(0.99, 0.05,
                  'RESEARCH PROTOTYPE — NOT FOR CLINICAL USE',
                  transform=ax_result.transAxes,
                  fontsize=8, color='#aaa', ha='right', style='italic')

    # Title
    fig.suptitle(
        f'ECG Waveform Analysis  |  {title}' if title else 'ECG Waveform Analysis',
        fontsize=14, fontweight='500', color='#222', y=0.95
    )

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# Demo: run on synthetic neurogenic and ischemic ECGs
# ---------------------------------------------------------------------------

def run_demo(output_dir: str = 'outputs'):
    """
    Generate synthetic neurogenic and ischemic ECGs,
    extract features, classify, and produce annotated waveform plots.
    """
    from model import load_model

    Path(output_dir).mkdir(exist_ok=True)

    try:
        model_obj = load_model()
    except Exception:
        print("No saved model found. Run: python3 pipeline.py --mode train_synthetic first.")
        return

    print("\nGenerating waveform analysis demo...")

    for ecg_type in ['neurogenic', 'ischemic']:
        print(f"\n--- {ecg_type.upper()} ECG ---")

        # Generate all 12 leads
        signals = []
        for lead in LEAD_NAMES:
            if ecg_type == 'neurogenic':
                sig = generate_neurogenic_ecg(lead_name=lead)
            else:
                sig = generate_ischemic_ecg(lead_name=lead)
            signals.append(sig)
        signals = np.array(signals)

        result = classify_waveform(signals, fs=500.0,
                                    lead_names=LEAD_NAMES,
                                    model_obj=model_obj)

        print(f"Prediction:    {result['prediction']}")
        print(f"Confidence:    {result['confidence']:.1%}")
        print(f"Prob neuro:    {result['prob_neurogenic']:.1%}")
        print(f"Prob ischemic: {result['prob_ischemic']:.1%}")
        if not np.isnan(result.get('qtc_ms', float('nan'))):
            print(f"QTc:           {result['qtc_ms']:.0f} ms")
        print(f"Heart rate:    {result['heart_rate']:.0f} bpm")

        out_path = f"{output_dir}/ecg_analysis_{ecg_type}.png"
        plot_waveform_analysis(
            signals=signals, fs=500.0,
            lead_names=LEAD_NAMES,
            classification_result=result,
            output_path=out_path,
            title=f'Synthetic {ecg_type.capitalize()} ECG'
        )
        print(f"Plot saved: {out_path}")

    print("\nDemo complete. Check outputs/ folder for annotated ECG plots.")


if __name__ == '__main__':
    run_demo()
