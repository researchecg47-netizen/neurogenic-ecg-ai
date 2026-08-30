"""
app.py
------
Streamlit web app for neurogenic vs ischemic ECG classification.
Upload a CSV of ECG features or a raw WFDB signal file.
Returns classification, confidence, SHAP top features, and annotated ECG plot.

Run locally:
  streamlit run app.py

Deploy to Hugging Face Spaces:
  Upload this file + model files to a new Space
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import os
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Neurogenic ECG Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------
# Styling
# ---------------------------------------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 2rem;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 0.25rem;
}
.subtitle {
    font-size: 1rem;
    color: #666;
    margin-bottom: 2rem;
}
.result-box-neuro {
    background: #e6f1fb;
    border-left: 4px solid #185FA5;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.result-box-isch {
    background: #faece7;
    border-left: 4px solid #D85A30;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.result-label {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.result-conf {
    font-size: 1rem;
    color: #444;
}
.metric-card {
    background: #f8f8f8;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    text-align: center;
    border: 1px solid #eee;
}
.disclaimer {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #856404;
    margin-top: 1rem;
}
.shap-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 0;
    border-bottom: 1px solid #f0f0f0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# SHAP clinical descriptions
# ---------------------------------------------------------------
SHAP_DESCRIPTIONS = {
    'st_t_duration': 'ST-T segment duration — proxy for QT interval. Prolonged in neurogenic ECGs due to autonomic dysregulation.',
    't_end': 'T-wave end timing. Delayed T-wave end indicates repolarization abnormality consistent with neurogenic injury.',
    'qrs_end': 'QRS complex end (J point). Defines start of ST segment — important for ST measurement.',
    'hr_from_rr': 'Heart rate derived from RR intervals. Autonomic instability from insular cortex damage alters heart rate pattern.',
    'rr_ms': 'RR interval in milliseconds. Beat-to-beat variation reflects autonomic nervous system regulation.',
    'qrs_onset': 'QRS complex onset timing. Reflects ventricular depolarization initiation.',
    'pr_interval': 'PR interval. Reflects AV node conduction — may be altered by autonomic dysregulation.',
    'p_end': 'P-wave end timing. Reflects atrial depolarization completion.',
    'p_onset': 'P-wave onset timing. Reflects atrial depolarization initiation.',
    'qtc_ms': 'QTc interval (Bazett corrected). Markedly prolonged in neurogenic ECGs — key distinguishing feature.',
    'heart_rate': 'Heart rate in beats per minute. Neurogenic cases show autonomic instability pattern.',
    'rmssd': 'RMSSD — heart rate variability metric. Reflects parasympathetic activity.',
}

def get_shap_description(feature_name):
    if feature_name in SHAP_DESCRIPTIONS:
        return SHAP_DESCRIPTIONS[feature_name]
    parts = feature_name.split('_')
    if len(parts) >= 2:
        lead = parts[0]
        feat = '_'.join(parts[1:])
        feat_map = {
            'st_j0': 'ST level at J point',
            'st_j20': 'ST level at J+20ms',
            'st_j60': 'ST level at J+60ms',
            'st_slope': 'ST segment slope',
            't_peak': 'T-wave peak amplitude',
            't_peak_idx': 'T-wave peak timing',
            't_area': 'T-wave area (integral)',
            't_asymmetry': 'T-wave shape asymmetry',
        }
        desc = feat_map.get(feat, feat)
        return f'Lead {lead}: {desc}.'
    return feature_name

# ---------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------
@st.cache_resource
def load_model_cached(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def find_model():
    candidates = [
        'models/neurogenic_ecg_latest.pkl',
        '../models/neurogenic_ecg_latest.pkl',
        'neurogenic_ecg_latest.pkl',
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return None

# ---------------------------------------------------------------
# Feature extraction from CSV
# ---------------------------------------------------------------
def extract_from_csv(df, feature_names):
    drop_cols = [c for c in df.columns if df[c].dtype == object and c != 'label']
    df = df.drop(columns=drop_cols, errors='ignore')
    X = df.reindex(columns=feature_names, fill_value=np.nan)
    return X

# ---------------------------------------------------------------
# Feature extraction from WFDB
# ---------------------------------------------------------------
def extract_from_wfdb(record_path, feature_names):
    try:
        import wfdb
        from scipy.signal import butter, filtfilt
    except ImportError:
        return None, "wfdb or scipy not installed"

    try:
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal
        if signals is None:
            return None, "No signal data in file"
        signals = signals.T
        fs = float(record.fs)
        sig_names = record.sig_name
    except Exception as e:
        return None, str(e)

    features = {}

    # Bandpass filter
    def bandpass(signal, fs, lo=0.5, hi=40.0):
        nyq = fs / 2.0
        b, a = butter(4, [lo/nyq, hi/nyq], btype='band')
        return filtfilt(b, a, signal)

    # R-peak detection
    def detect_r(signal, fs):
        diff = np.diff(signal)
        sq = diff ** 2
        win = int(0.15 * fs)
        integ = np.convolve(sq, np.ones(win)/win, mode='same')
        thresh = 0.6 * np.max(integ)
        above = integ > thresh
        peaks = []
        ref = int(0.2 * fs)
        last = -ref
        for i in range(1, len(above)):
            if above[i] and not above[i-1]:
                s = max(0, i - win//2)
                e = min(len(signal), i + win//2)
                p = s + np.argmax(signal[s:e])
                if p - last > ref:
                    peaks.append(p)
                    last = p
        return np.array(peaks)

    ii_idx = 1
    for j, name in enumerate(sig_names):
        if name.upper().replace(' ', '') == 'II':
            ii_idx = j
            break

    lead_ii = bandpass(signals[ii_idx], fs)
    r_peaks = detect_r(lead_ii, fs)

    if len(r_peaks) < 2:
        return None, "Could not detect R-peaks — check signal quality"

    rr = np.diff(r_peaks) / fs * 1000
    features['rr_ms'] = float(np.mean(rr))
    features['heart_rate'] = float(60000 / features['rr_ms'])
    features['hr_std'] = float(np.std(60000 / rr))
    features['hr_from_rr'] = features['heart_rate']

    if len(r_peaks) >= 3:
        sd = np.diff(rr)
        features['rmssd'] = float(np.sqrt(np.mean(sd**2)))
        features['sdnn'] = float(np.std(rr))
        features['pnn50'] = float(np.mean(np.abs(sd) > 50) * 100)

    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

    for i, lead in enumerate(lead_names):
        midx = None
        for j, name in enumerate(sig_names):
            if name.upper().replace(' ','') == lead.upper().replace(' ',''):
                midx = j
                break
        if midx is None or midx >= signals.shape[0]:
            continue

        sig = bandpass(signals[midx], fs)
        half = int(0.3 * fs)
        beats = []
        for r in r_peaks:
            if r - half >= 0 and r + half < len(sig):
                beats.append(sig[r-half:r+half])
        if not beats:
            continue
        med = np.median(beats, axis=0)
        rc = len(med) // 2

        j0 = rc + int(0.04*fs)
        bl_s = max(0, rc - int(0.05*fs))
        bl_e = max(0, rc - int(0.02*fs))
        if bl_e > bl_s and j0 < len(med):
            bl = np.mean(med[bl_s:bl_e])
            features[f'{lead}_st_j0'] = float(med[j0] - bl) if j0 < len(med) else np.nan
            j20 = j0 + int(0.02*fs)
            j60 = j0 + int(0.06*fs)
            features[f'{lead}_st_j20'] = float(med[j20] - bl) if j20 < len(med) else np.nan
            features[f'{lead}_st_j60'] = float(med[j60] - bl) if j60 < len(med) else np.nan

        qe = rc + int(0.06*fs)
        te = min(rc + int(0.32*fs), len(med))
        if qe < te:
            tseg = med[qe:te]
            mx = np.max(tseg)
            mn = np.min(tseg)
            tp = mx if abs(mx) >= abs(mn) else mn
            features[f'{lead}_t_peak'] = float(tp)
            tpi = np.argmax(np.abs(tseg))
            features[f'{lead}_t_peak_idx'] = float(tpi)
            features[f'{lead}_t_area'] = float(np.trapz(tseg))

    qt_approx = features['rr_ms'] * 0.42
    qtc = qt_approx / np.sqrt(features['rr_ms'] / 1000)
    features['qtc_ms'] = float(qtc)
    features['qtc_prolonged'] = int(qtc > 450)
    features['qtc_markedly_prolonged'] = int(qtc > 500)
    features['qrs_dur_ms'] = np.nan
    features['wide_qrs'] = 0

    X = pd.DataFrame([{k: features.get(k, np.nan) for k in feature_names}])
    return X, signals, sig_names, fs, r_peaks

# ---------------------------------------------------------------
# ECG plot
# ---------------------------------------------------------------
def plot_ecg(signals, sig_names, fs, r_peaks, prediction, confidence,
             prob_n, prob_i, qtc, hr):
    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    NEURO = '#185FA5'
    ISCH  = '#D85A30'
    pred_color = NEURO if prediction == 'Neurogenic' else ISCH

    fig = plt.figure(figsize=(18, 11), facecolor='white')
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.3,
                           top=0.88, bottom=0.06, left=0.05, right=0.97)
    layout = [
        ['I','aVR','V1','V4'],
        ['II','aVL','V2','V5'],
        ['III','aVF','V3','V6'],
    ]

    def plot_lead(ax, lead):
        sidx = None
        for j, name in enumerate(sig_names):
            if name.upper().replace(' ','') == lead.upper().replace(' ',''):
                sidx = j
                break
        if sidx is None or sidx >= signals.shape[0]:
            ax.text(0.5, 0.5, f'{lead}\n(N/A)', ha='center', va='center',
                   fontsize=9, color='#aaa', transform=ax.transAxes)
            ax.axis('off')
            return

        from scipy.signal import butter, filtfilt
        def bp(s, fs):
            nyq = fs/2
            b, a = butter(4, [0.5/nyq, min(40/nyq, 0.99)], btype='band')
            return filtfilt(b, a, s)

        sig = bp(signals[sidx], fs)
        show = min(int(4*fs), len(sig))
        t = np.arange(show) / fs
        ax.plot(t, sig[:show], color='#222', linewidth=0.9, zorder=3)
        ax.set_facecolor('#fafafa')
        for x in np.arange(0, 4.1, 0.2):
            ax.axvline(x, color='#ffcccc', linewidth=0.3, zorder=1)
        for x in np.arange(0, 4.1, 1.0):
            ax.axvline(x, color='#ffaaaa', linewidth=0.6, zorder=1)
        ax.axhline(0, color='#ffaaaa', linewidth=0.4, zorder=1)

        vp = r_peaks[r_peaks < show]
        if len(vp):
            ax.scatter(vp/fs, sig[vp], color='#E24B4A', s=18, zorder=5, marker='^')

        ax.set_xlim(0, 4)
        rng = np.ptp(sig[:show])
        pad = max(rng*0.3, 0.2)
        ax.set_ylim(sig[:show].min()-pad, sig[:show].max()+pad)
        ax.set_ylabel('mV', fontsize=7, color='#888')
        ax.tick_params(labelsize=6, colors='#888')
        for sp in ['top','right']:
            ax.spines[sp].set_visible(False)
        ax.text(0.02, 0.92, lead, transform=ax.transAxes,
               fontsize=10, fontweight='bold', color='#333', zorder=7)

    for ri, row in enumerate(layout):
        for ci, lead in enumerate(row):
            ax = fig.add_subplot(gs[ri, ci])
            plot_lead(ax, lead)

    ax_res = fig.add_subplot(gs[3, :])
    ax_res.axis('off')

    rb = dict(boxstyle='round,pad=0.5', facecolor=pred_color, alpha=0.12,
              edgecolor=pred_color, linewidth=2)
    ax_res.text(0.01, 0.75, f'Classification: {prediction.upper()}',
               transform=ax_res.transAxes, fontsize=15,
               fontweight='bold', color=pred_color, bbox=rb, va='top')
    ax_res.text(0.01, 0.25, f'Confidence: {confidence:.1%}',
               transform=ax_res.transAxes, fontsize=11, color='#444', va='top')

    bx, bw, bh, by = 0.22, 0.22, 0.28, 0.35
    ax_res.add_patch(plt.Rectangle((bx, by), bw*prob_n, bh,
                    transform=ax_res.transAxes, color=NEURO, alpha=0.7, zorder=3))
    ax_res.add_patch(plt.Rectangle((bx+bw*prob_n, by), bw*prob_i, bh,
                    transform=ax_res.transAxes, color=ISCH, alpha=0.7, zorder=3))
    ax_res.add_patch(plt.Rectangle((bx, by), bw, bh,
                    transform=ax_res.transAxes, fill=False,
                    edgecolor='#ccc', linewidth=1))
    ax_res.text(bx, by+bh+0.08, f'Neurogenic {prob_n:.1%}',
               transform=ax_res.transAxes, fontsize=9, color=NEURO)
    ax_res.text(bx+bw-0.01, by+bh+0.08, f'Ischemic {prob_i:.1%}',
               transform=ax_res.transAxes, fontsize=9, color=ISCH, ha='right')

    metrics = [
        ('QTc', f'{qtc:.0f} ms' if not np.isnan(qtc) else 'N/A', qtc > 450 if not np.isnan(qtc) else False),
        ('Heart rate', f'{hr:.0f} bpm' if not np.isnan(hr) else 'N/A', False),
    ]
    for i, (label, val, flag) in enumerate(metrics):
        xp = 0.52 + i*0.16
        col = '#BA7517' if flag else '#333'
        ax_res.text(xp, 0.85, label, transform=ax_res.transAxes,
                   fontsize=9, color='#888', ha='center')
        ax_res.text(xp, 0.45, val, transform=ax_res.transAxes,
                   fontsize=13, fontweight='bold', color=col, ha='center')
        if flag:
            ax_res.text(xp, 0.1, '⚠ Prolonged', transform=ax_res.transAxes,
                       fontsize=8, color='#BA7517', ha='center')

    ax_res.text(0.99, 0.05, 'RESEARCH PROTOTYPE — NOT FOR CLINICAL USE',
               transform=ax_res.transAxes, fontsize=8, color='#aaa',
               ha='right', style='italic')

    fig.suptitle('ECG Waveform Analysis', fontsize=13, fontweight='500',
                color='#222', y=0.95)
    return fig

# ---------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------
def compute_shap_values(model_obj, X):
    try:
        import shap
        model = model_obj['model']
        feature_names = model_obj['feature_names']

        if hasattr(model, 'named_estimators_'):
            rf_pipe = model.named_estimators_['rf']
        else:
            rf_pipe = model

        rf_clf = rf_pipe.named_steps['clf']
        X_t = rf_pipe.named_steps['imputer'].transform(X)
        X_t = rf_pipe.named_steps['scaler'].transform(X_t)
        n_cols = X_t.shape[1]
        col_names = list(X.columns)[:n_cols]
        X_df = pd.DataFrame(X_t, columns=col_names)

        explainer = shap.TreeExplainer(rf_clf)
        shap_values = explainer.shap_values(X_df)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
        elif hasattr(shap_values, 'ndim') and shap_values.ndim == 2:
            sv = shap_values[0, :]
        else:
            sv = shap_values[0]

        feat_shap = pd.Series(np.abs(sv), index=col_names)
        return feat_shap.nlargest(10)
    except Exception as e:
        return None

# ---------------------------------------------------------------
# Main app
# ---------------------------------------------------------------
def main():
    # Header
    st.markdown('<div class="main-title">🫀 Neurogenic ECG Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered tool to distinguish neurogenic ECG changes (stroke, SAH, TBI) from primary ischemic changes (STEMI/NSTEMI)</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **Model:** RF + XGBoost ensemble  
        **Training data:** 953,128 ECGs  
        **AUC:** 0.9974 (CV), 0.9981 (test)  
        **Sensitivity:** 0.9954  
        **Specificity:** 0.9597  
        **PPV:** 0.9611  
        """)

        st.markdown("---")
        st.header("How to use")
        st.markdown("""
        1. Upload a CSV file with ECG features, or a WFDB .hea file
        2. The model classifies as neurogenic or ischemic
        3. Review the confidence score, ECG plot, and SHAP explanation
        """)

        st.markdown("---")
        st.markdown("**Top SHAP features:**")
        top_feats = {
            'ST-T duration': 0.1819,
            'T-wave end': 0.1030,
            'Heart rate': 0.0359,
            'RR interval': 0.0342,
        }
        for feat, val in top_feats.items():
            pct = val / 0.1819
            st.progress(pct, text=f"{feat}: {val:.4f}")

        st.markdown("---")
        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Research prototype only.</b><br>
        Not validated for clinical use. Do not use for medical decisions.
        </div>
        """, unsafe_allow_html=True)

    # Find model
    model_path = find_model()

    # File upload
    st.subheader("Upload ECG Data")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Option 1: Feature CSV**")
        st.caption("CSV with pre-extracted ECG features. Must have numeric columns.")
        csv_file = st.file_uploader("Upload CSV", type=['csv'], key='csv')

    with col2:
        st.markdown("**Option 2: WFDB Signal File**")
        st.caption("Upload both .hea and .dat files from MIMIC or PTB-XL.")
        hea_file = st.file_uploader("Upload .hea file", type=['hea'], key='hea')
        dat_file = st.file_uploader("Upload .dat file", type=['dat'], key='dat')

    # Demo mode if no model
    if model_path is None:
        st.warning("No trained model found. Running in demo mode with example results.")
        st.info("To use the real model, place `neurogenic_ecg_latest.pkl` in a `models/` folder.")

        if st.button("Run demo classification", type="primary"):
            show_demo_results()
        return

    model_obj = load_model_cached(model_path)
    feature_names = model_obj['feature_names']

    # Process CSV
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        st.success(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")

        if st.button("Classify", type="primary", key='csv_btn'):
            with st.spinner("Running classification..."):
                X = extract_from_csv(df, feature_names)
                model = model_obj['model']
                probs = model.predict_proba(X)
                preds = model.predict(X)

                st.subheader("Results")
                for i in range(min(len(df), 5)):
                    pred = 'Neurogenic' if preds[i] == 1 else 'Ischemic'
                    conf = float(probs[i].max())
                    prob_n = float(probs[i][1])
                    prob_i = float(probs[i][0])

                    box_class = 'result-box-neuro' if pred == 'Neurogenic' else 'result-box-isch'
                    color = '#185FA5' if pred == 'Neurogenic' else '#D85A30'
                    st.markdown(f"""
                    <div class="{box_class}">
                        <div class="result-label" style="color:{color}">Record {i+1}: {pred}</div>
                        <div class="result-conf">Confidence: {conf:.1%} | Neurogenic: {prob_n:.1%} | Ischemic: {prob_i:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # SHAP for first record
                st.subheader("Feature explanation (Record 1)")
                shap_vals = compute_shap_values(model_obj, X.iloc[[0]])
                if shap_vals is not None:
                    show_shap_table(shap_vals)
                else:
                    st.info("Install shap library for feature explanations: pip install shap")

    # Process WFDB
    elif hea_file is not None and dat_file is not None:
        with st.spinner("Processing ECG signal..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                hea_path = Path(tmpdir) / hea_file.name
                dat_path = Path(tmpdir) / dat_file.name
                hea_path.write_bytes(hea_file.read())
                dat_path.write_bytes(dat_file.read())
                record_path = str(hea_path).replace('.hea', '')

                result = extract_from_wfdb(record_path, feature_names)

                if result[0] is None:
                    st.error(f"Could not process signal: {result[1]}")
                    return

                X, signals, sig_names, fs, r_peaks = result
                model = model_obj['model']
                probs = model.predict_proba(X)
                pred_label = 'Neurogenic' if model.predict(X)[0] == 1 else 'Ischemic'
                conf = float(probs[0].max())
                prob_n = float(probs[0][1])
                prob_i = float(probs[0][0])
                qtc = float(X.get('qtc_ms', pd.Series([np.nan])).iloc[0])
                hr = float(X.get('heart_rate', pd.Series([np.nan])).iloc[0])

                st.subheader("Classification Result")
                box_class = 'result-box-neuro' if pred_label == 'Neurogenic' else 'result-box-isch'
                color = '#185FA5' if pred_label == 'Neurogenic' else '#D85A30'
                st.markdown(f"""
                <div class="{box_class}">
                    <div class="result-label" style="color:{color}">{pred_label}</div>
                    <div class="result-conf">
                        Confidence: {conf:.1%} &nbsp;|&nbsp;
                        Neurogenic probability: {prob_n:.1%} &nbsp;|&nbsp;
                        Ischemic probability: {prob_i:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                with m1:
                    qtc_str = f"{qtc:.0f} ms" if not np.isnan(qtc) else "N/A"
                    flag = "⚠️ Prolonged" if not np.isnan(qtc) and qtc > 450 else ""
                    st.metric("QTc Interval", qtc_str, flag)
                with m2:
                    hr_str = f"{hr:.0f} bpm" if not np.isnan(hr) else "N/A"
                    st.metric("Heart Rate", hr_str)
                with m3:
                    st.metric("R-peaks detected", len(r_peaks))

                # ECG plot
                st.subheader("12-Lead ECG Waveform")
                fig = plot_ecg(signals, sig_names, fs, r_peaks,
                               pred_label, conf, prob_n, prob_i, qtc, hr)
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # SHAP
                st.subheader("Feature Explanation")
                st.caption("Which ECG features drove this classification, and why they matter clinically.")
                shap_vals = compute_shap_values(model_obj, X)
                if shap_vals is not None:
                    show_shap_table(shap_vals)
                else:
                    st.info("Install shap for feature explanations: pip install shap")

    else:
        st.info("Upload an ECG file above to begin classification.")
        show_model_stats()


def show_shap_table(shap_vals):
    max_val = shap_vals.max()
    color_neuro = '#185FA5'
    for feat, val in shap_vals.items():
        pct = val / max_val
        desc = get_shap_description(feat)
        col1, col2, col3 = st.columns([2, 1, 3])
        with col1:
            st.markdown(f"**{feat}**")
        with col2:
            st.progress(float(pct), text=f"{val:.4f}")
        with col3:
            st.caption(desc)


def show_model_stats():
    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("AUC-ROC", "0.9981"),
        ("Sensitivity", "99.5%"),
        ("Specificity", "96.0%"),
        ("PPV", "96.1%"),
    ]
    for col, (label, val) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.metric(label, val)

    st.subheader("Clinical Background")
    st.markdown("""
    When a patient suffers a **stroke, subarachnoid hemorrhage (SAH), or traumatic brain injury (TBI)**,
    the resulting neurological stress floods the heart with adrenaline and distorts the ECG in ways
    that closely resemble a heart attack.

    Emergency physicians may misidentify these **neurogenic ECG changes** as acute coronary syndrome,
    leading to unnecessary cardiac catheterizations while the underlying brain injury goes untreated.

    This tool uses machine learning trained on **953,128 real hospital ECGs** to flag potentially
    neurogenic ECG patterns in real time, giving clinicians one additional piece of information
    in the critical first ten minutes.
    """)


def show_demo_results():
    st.subheader("Demo Results — Synthetic Neurogenic ECG")
    st.markdown("""
    <div class="result-box-neuro">
        <div class="result-label" style="color:#185FA5">NEUROGENIC</div>
        <div class="result-conf">Confidence: 93.2% | Neurogenic: 93.2% | Ischemic: 6.8%</div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("QTc Interval", "487 ms", "⚠️ Prolonged")
    with col2:
        st.metric("Heart Rate", "67 bpm")

    st.subheader("Top features driving this classification")
    demo_feats = {
        'st_t_duration': 0.1819,
        't_end': 0.1030,
        'hr_from_rr': 0.0359,
        'rr_ms': 0.0342,
        'qtc_ms': 0.0280,
    }
    for feat, val in demo_feats.items():
        pct = val / 0.1819
        desc = get_shap_description(feat)
        c1, c2, c3 = st.columns([2, 1, 3])
        with c1:
            st.markdown(f"**{feat}**")
        with c2:
            st.progress(float(pct), text=f"{val:.4f}")
        with c3:
            st.caption(desc)


if __name__ == '__main__':
    main()
