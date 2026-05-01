# -*- coding: utf-8 -*-
"""
Paper 4 — Spectral Decomposition Validation Pipeline
=====================================================
원본 분석(잔차 기반)과 3가지 독립적 방법을 비교합니다.

Method 1 (Primary)  : Residual-based log-log regression (원본 v6과 동일)
Method 2 (Corrected): specparam v2 — 수정된 파라미터 + LINEAR PSD 입력
Method 3 (IRASA)    : Irregular-Resampling Auto-Spectral Analysis (Wen & Liu 2016)
Method 4 (EMD)      : Band-filtered Hilbert amplitude (fallback from EEMD)

데이터 경로: C:\SOP_data\analyzed data-pdh-eLoreta\
파일 형식:   wake_30s_XX.dat / trans_30s_XX.dat / early_30s_XX.dat / late_30s_XX.dat

Usage:
    python 05_validation_pipeline.py

Output: C:\SOP_data\validation_results\
"""

import os, re, sys, warnings, traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt
from scipy.signal import hilbert as scipy_hilbert
from scipy.stats import pearsonr, ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ── specparam / fooof import ──────────────────────────────────────────────────
try:
    from fooof import FOOOF
    _SP_VERSION = "fooof"
    print("specparam: using fooof v1.1")
except ImportError:
    try:
        from specparam import SpectralModel as FOOOF
        _SP_VERSION = "specparam"
        print("specparam: using specparam v2")
    except ImportError:
        FOOOF = None
        print("WARNING: fooof/specparam not installed. Run: pip install fooof")

# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS  (원본 v6과 동일하게 유지)
# ──────────────────────────────────────────────────────────────────────────────
SFREQ            = 500
N_CHANNELS       = 30
EXPECTED_SAMPLES = 15000
DEFAULT_ROOT     = Path(r"C:\SOP_data")
RESULT_DIR_NAME  = "validation_results"

STRICT_STAGE_PATTERNS = {
    "WS":   re.compile(r"wake_30s_(\d+)\.dat$",  re.I),
    "TS":   re.compile(r"trans_30s_(\d+)\.dat$", re.I),
    "ESS1": re.compile(r"early_30s_(\d+)\.dat$", re.I),
    "LSS1": re.compile(r"late_30s_(\d+)\.dat$",  re.I),
}
STAGE_ORDER = ["WS", "TS", "ESS1", "LSS1"]

MONTAGE = [
    "O2","O1","Oz","Pz","P4","CP4","P8/T6","C4","TP8","T8/T4",
    "P7/T5","P3","CP3","CPz","Cz","FC4","FT8","TP7","C3","FCz",
    "Fz","F4","F8","T7/T3","FT7","FC3","F3","Fp2","F7","Fp1"
]
DAT_TO_LC = [29,27,28,26,20,21,22,23,18,14,
              7, 9,10,11, 3, 4, 6, 1, 2, 0,
             25,19,15,12,13, 5,24,16,17, 8]

BANDS = {
    "delta":  (1.0,  3.5),
    "theta":  (4.0,  7.5),
    "alpha1": (8.0,  10.0),
    "alpha2": (10.5, 12.0),
    "beta1":  (12.5, 18.0),
    "beta2":  (18.5, 21.0),
    "beta3":  (21.5, 30.0),
}

# 9 anatomical ROIs (논문 primary level)
ANAT_UNITS = {
    "L_DLPFC":      ["Fp1","F3","FC3"],
    "R_DLPFC":      ["Fp2","F4","FC4"],
    "VmPFC_OrbPFC": ["Fp1","Fp2","Fz","FCz"],
    "ACC":          ["Fz","FCz","Cz","CPz"],
    "SPG":          ["P3","Pz","P4"],
    "IPG":          ["P7/T5","P8/T6","TP7","TP8"],
    "L_STG":        ["F7","FT7","T7/T3"],
    "R_STG":        ["F8","FT8","T8/T4"],
    "Occipital":    ["O1","Oz","O2"],
}
CH_LOOKUP = {ch: i for i, ch in enumerate(MONTAGE)}

# specparam — 수정된 파라미터
SP_PARAMS = dict(
    peak_width_limits=[0.5, 12.0],
    max_n_peaks=8,          # ← 원래 오류: max_peaks (존재하지 않는 파라미터)
    min_peak_height=0.05,
    peak_threshold=2.0,
    aperiodic_mode='knee',  # ← 원래 오류: 'fixed' (수면 EEG에 부적합)
    verbose=False
)
SP_FREQ_RANGE = [2, 30]     # delta 저주파 클리핑으로 안정성 향상

# IRASA
IRASA_HSET    = np.arange(1.1, 1.95, 0.05)
IRASA_NPERSEG = int(4 * SFREQ)   # 4초 window (원본과 동일)
IRASA_NOVERLAP= IRASA_NPERSEG // 2

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (원본 v6과 동일)
# ──────────────────────────────────────────────────────────────────────────────
def classify_stage(path):
    for stage, pat in STRICT_STAGE_PATTERNS.items():
        if pat.search(path.name):
            return stage
    return None

def infer_subject_id(path):
    for stage, pat in STRICT_STAGE_PATTERNS.items():
        m = pat.search(path.name)
        if m:
            return f"S{int(m.group(1)):02d}"
    return None

def infer_sex(path):
    txt = str(path).lower()
    if "\\female\\" in txt or "/female/" in txt:
        return "female"
    if "\\male\\" in txt or "/male/" in txt:
        return "male"
    return "unknown"

def try_load_text(path):
    for kw in [dict(), dict(sep=r"\s+", engine="python")]:
        try:
            arr = pd.read_csv(path, header=None, **kw).values.astype(float)
            if arr.size > 0:
                return arr
        except Exception:
            pass
    return None

def try_load_binary(path):
    raw = path.read_bytes()
    for dtype in (np.float32, np.float64, np.int16, np.int32):
        try:
            arr = np.frombuffer(raw, dtype=dtype)
            for shape in ((EXPECTED_SAMPLES, N_CHANNELS), (N_CHANNELS, EXPECTED_SAMPLES)):
                total = shape[0] * shape[1]
                if arr.size >= total:
                    return arr[:total].reshape(shape).astype(float)
        except Exception:
            pass
    return None

def reshape_eeg(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, N_CHANNELS)
    if arr.shape[1] == N_CHANNELS:
        eeg = arr
    elif arr.shape[0] == N_CHANNELS:
        eeg = arr.T
    else:
        eeg = arr if arr.shape[1] == N_CHANNELS else arr.T
    return eeg[:, DAT_TO_LC]

def load_dat(path):
    arr = try_load_text(path)
    if arr is None:
        arr = try_load_binary(path)
    if arr is None:
        raise ValueError(f"Cannot read: {path}")
    return reshape_eeg(arr)

def find_files(root):
    rows = []
    for p in root.rglob("*.dat"):
        stage = classify_stage(p)
        subj  = infer_subject_id(p)
        if stage and subj:
            rows.append({"path": str(p), "stage": stage,
                          "subject": subj, "sex": infer_sex(p)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Deduplicate: keep shortest path (avoid results/ backup/ copies)
    df["_len"] = df["path"].str.len()
    df = df.sort_values(["subject","stage","_len"]).drop_duplicates(
         subset=["subject","stage"], keep="first").drop(columns="_len")
    return df.reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# PSD  (원본 v6과 동일 — LINEAR scale 반환)
# ──────────────────────────────────────────────────────────────────────────────
def compute_psd(eeg):
    nperseg  = min(int(4 * SFREQ), eeg.shape[0])
    noverlap = nperseg // 2
    freqs, psd = welch(eeg, fs=SFREQ, nperseg=nperseg,
                       noverlap=noverlap, axis=0,
                       scaling="density", detrend="constant")
    mask = (freqs >= 1.0) & (freqs <= 30.0)
    return freqs[mask], psd[mask, :]   # (n_freqs, n_channels) — LINEAR

def roi_spectrum(psd_ch, channels):
    idx = [CH_LOOKUP[c] for c in channels if c in CH_LOOKUP]
    return np.mean(psd_ch[:, idx], axis=1) if idx else None

# ──────────────────────────────────────────────────────────────────────────────
# METHOD 1: RESIDUAL-BASED  (원본 v6과 동일)
# ──────────────────────────────────────────────────────────────────────────────
def residual_decompose(freqs, psd_linear, eps=1e-12):
    """Log-log linear aperiodic fit; residual = oscillatory component."""
    logf = np.log10(freqs)
    logp = np.log10(np.maximum(psd_linear, eps))
    slope, intercept = np.polyfit(logf, logp, 1)
    fitted   = intercept + slope * logf
    residual = logp - fitted
    return {
        "ap_offset":   float(intercept),
        "ap_exponent": float(-slope),
        "residual":    residual,
        "logp":        logp,
        "fitted":      fitted,
    }

def residual_band_metrics(freqs, decomp, band_name):
    lo, hi = BANDS[band_name]
    m = (freqs >= lo) & (freqs <= hi)
    if not m.any():
        return dict(peak_cf=np.nan, peak_height=np.nan, peak_present=np.nan)
    resid  = decomp["residual"][m]
    idx    = int(np.argmax(resid))
    return {
        "peak_cf":     float(freqs[m][idx]),
        "peak_height": float(resid[idx]),
        "peak_present":float(resid[idx] > 0),
    }

def band_power(freqs, psd_linear, band_name):
    lo, hi = BANDS[band_name]
    m = (freqs >= lo) & (freqs <= hi)
    return float(np.mean(psd_linear[m])) if m.any() else np.nan

# ──────────────────────────────────────────────────────────────────────────────
# METHOD 2: specparam (수정 버전)
# ──────────────────────────────────────────────────────────────────────────────
def specparam_decompose(freqs, psd_linear):
    """
    올바른 specparam 사용:
      - LINEAR scale PSD 직접 입력 (log 변환 금지)
      - max_n_peaks (not max_peaks)
      - aperiodic_mode='knee'
    """
    if FOOOF is None:
        return None
    f_mask = (freqs >= SP_FREQ_RANGE[0]) & (freqs <= SP_FREQ_RANGE[1])
    f_fit  = freqs[f_mask]
    p_fit  = psd_linear[f_mask]   # ← LINEAR (specparam이 내부에서 log10 변환)
    try:
        fm = FOOOF(**SP_PARAMS)
        fm.fit(f_fit, p_fit)
        if not np.isfinite(fm.r_squared_):
            return None
        peaks = fm.gaussian_params_ if len(fm.gaussian_params_) > 0 else []
        return {
            "converged":   True,
            "r_squared":   float(fm.r_squared_),
            "ap_offset":   float(fm.aperiodic_params_[0]),
            "ap_exponent": float(fm.aperiodic_params_[-1]),
            "peaks":       peaks,
        }
    except Exception:
        return None

def specparam_band_metrics(sp_result, band_name):
    if sp_result is None or not sp_result["converged"]:
        return dict(peak_cf=np.nan, peak_height=np.nan, peak_present=0.0,
                    converged=False)
    lo, hi = BANDS[band_name]
    peaks  = sp_result["peaks"]
    in_band = [p for p in peaks if lo <= p[0] <= hi] if len(peaks) > 0 else []
    if in_band:
        bp = max(in_band, key=lambda p: p[1])
        return dict(peak_cf=float(bp[0]), peak_height=float(bp[1]),
                    peak_present=1.0, converged=True)
    return dict(peak_cf=np.nan, peak_height=0.0, peak_present=0.0, converged=True)

# ──────────────────────────────────────────────────────────────────────────────
# METHOD 3: IRASA  (Wen & Liu, 2016)
# ──────────────────────────────────────────────────────────────────────────────
def irasa_decompose(x, hset=None):
    """
    시계열 x → (freqs, psd_aperiodic, psd_oscillatory)
    """
    if hset is None:
        hset = IRASA_HSET
    from scipy.signal import resample as sp_resample

    nperseg  = IRASA_NPERSEG
    noverlap = IRASA_NOVERLAP
    freqs_base, psd_total = welch(x, fs=SFREQ, nperseg=nperseg,
                                   noverlap=noverlap, scaling="density")
    f_mask = (freqs_base >= 1.0) & (freqs_base <= 30.0)
    freqs_out  = freqs_base[f_mask]
    total_out  = psd_total[f_mask]

    ap_set = []
    for h in hset:
        n_up  = int(len(x) * h)
        x_up  = sp_resample(x, n_up)
        nps_u = int(nperseg * h)
        f_u, p_u = welch(x_up, fs=SFREQ*h, nperseg=nps_u,
                         noverlap=nps_u//2, scaling="density")

        n_dn  = max(int(len(x) / h), 4)
        x_dn  = sp_resample(x, n_dn)
        nps_d = max(int(nperseg / h), 8)
        f_d, p_d = welch(x_dn, fs=SFREQ/h, nperseg=nps_d,
                         noverlap=nps_d//2, scaling="density")

        pu_i = np.interp(freqs_out, f_u, p_u, left=np.nan, right=np.nan)
        pd_i = np.interp(freqs_out, f_d, p_d, left=np.nan, right=np.nan)
        geo  = np.sqrt(np.maximum(pu_i * pd_i, 0.0))
        ap_set.append(geo)

    ap_median = np.nanmedian(np.array(ap_set), axis=0)
    osc       = np.maximum(total_out - ap_median, 0.0)
    return freqs_out, ap_median, osc, total_out

def irasa_band_metrics(freqs, psd_osc, psd_ap, band_name):
    lo, hi = BANDS[band_name]
    m = (freqs >= lo) & (freqs <= hi)
    if not m.any():
        return dict(peak_cf=np.nan, peak_height=np.nan)
    osc_band = psd_osc[m]
    idx = int(np.argmax(osc_band))
    height = np.log10(max(osc_band[idx], 1e-30))
    cf     = float(freqs[m][idx])
    return dict(peak_cf=cf, peak_height=height)

def irasa_aperiodic(freqs, psd_ap):
    valid = psd_ap > 0
    if valid.sum() < 3:
        return np.nan, np.nan
    logf   = np.log10(freqs[valid])
    logp   = np.log10(psd_ap[valid])
    coeffs = np.polyfit(logf, logp, 1)
    return float(-coeffs[0]), float(coeffs[1])

# ──────────────────────────────────────────────────────────────────────────────
# METHOD 4: Band-filtered Hilbert amplitude
# ──────────────────────────────────────────────────────────────────────────────
def hilbert_band_amplitude(x, lo, hi, order=4):
    """Bandpass → Hilbert → mean instantaneous power → log10."""
    nyq  = SFREQ / 2.0
    low  = max(lo / nyq, 0.002)
    high = min(hi / nyq, 0.998)
    if low >= high:
        return np.nan
    b, a  = butter(order, [low, high], btype='band')
    xf    = filtfilt(b, a, x)
    power = np.mean(np.abs(scipy_hilbert(xf))**2)
    return float(np.log10(max(power, 1e-30)))

# ──────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ──────────────────────────────────────────────────────────────────────────────
def cohens_dz(delta):
    d = np.asarray(delta, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan
    sd = np.std(d, ddof=1)
    return float(np.mean(d) / sd) if sd > 0 else np.nan

def paired_stats(vals_a, vals_b):
    mask = ~np.isnan(vals_a) & ~np.isnan(vals_b)
    a, b = vals_a[mask], vals_b[mask]
    if len(a) < 2:
        return dict(n=len(a), dz=np.nan, t=np.nan, p=np.nan, mean_delta=np.nan)
    delta = b - a
    t, p  = ttest_rel(b, a)
    return dict(n=int(len(a)), dz=cohens_dz(delta),
                t=float(t), p=float(p), mean_delta=float(np.mean(delta)))

# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print("Paper 4 — Spectral Decomposition Validation Pipeline")
    print("Methods: Residual | specparam (corrected) | IRASA | Hilbert-Amp")
    print("="*70)

    # Choose root
    root = DEFAULT_ROOT if DEFAULT_ROOT.exists() else Path(
        input("C:\\SOP_data not found. Enter path: ").strip().strip('"'))
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    out_dir = root / RESULT_DIR_NAME
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Find files
    print(f"\nScanning: {root}")
    files_df = find_files(root)
    if files_df.empty:
        raise FileNotFoundError("No .dat files found.")

    n_subj   = files_df["subject"].nunique()
    n_files  = len(files_df)
    print(f"Found: {n_subj} subjects × {n_files} files total")
    files_df.to_csv(out_dir / "selected_files.csv", index=False)

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────
    all_rows = []
    sp_conv  = {"total": 0, "converged": 0}

    for idx, row in enumerate(files_df.itertuples(index=False), 1):
        path  = Path(row.path)
        stage = row.stage
        subj  = row.subject
        sex   = row.sex

        print(f"[{idx:3d}/{n_files}] {subj} {stage} ({sex})  {path.name}")

        try:
            eeg = load_dat(path)  # (n_samples, 30)
        except Exception as e:
            print(f"  !! Load error: {e}")
            continue

        freqs, psd_ch = compute_psd(eeg)  # LINEAR PSD (n_freqs, 30)

        for roi_name, roi_chs in ANAT_UNITS.items():
            psd_roi = roi_spectrum(psd_ch, roi_chs)
            if psd_roi is None:
                continue

            # Representative single channel for time-series methods
            # (use first available channel in ROI)
            rep_ch_idx = next(
                (CH_LOOKUP[c] for c in roi_chs if c in CH_LOOKUP), 0)
            x_roi = eeg[:, rep_ch_idx]

            rec = dict(subject=subj, stage=stage, sex=sex, roi=roi_name)

            # ── Method 1: Residual ────────────────────────────────────────
            dec1 = residual_decompose(freqs, psd_roi)
            rec["res_ap_exponent"] = dec1["ap_exponent"]
            rec["res_ap_offset"]   = dec1["ap_offset"]
            for b in BANDS:
                m1 = residual_band_metrics(freqs, dec1, b)
                rec[f"res_{b}_peak_height"] = m1["peak_height"]
                rec[f"res_{b}_peak_cf"]     = m1["peak_cf"]
                rec[f"res_{b}_peak_present"]= m1["peak_present"]
                rec[f"res_{b}_band_power"]  = band_power(freqs, psd_roi, b)

            # ── Method 2: specparam (corrected) ───────────────────────────
            sp_conv["total"] += 1
            sp2 = specparam_decompose(freqs, psd_roi)  # LINEAR 입력!
            if sp2 and sp2["converged"]:
                sp_conv["converged"] += 1
                rec["sp_ap_exponent"] = sp2["ap_exponent"]
                rec["sp_ap_offset"]   = sp2["ap_offset"]
                rec["sp_r_squared"]   = sp2["r_squared"]
                rec["sp_converged"]   = True
            else:
                rec["sp_ap_exponent"] = np.nan
                rec["sp_ap_offset"]   = np.nan
                rec["sp_r_squared"]   = np.nan
                rec["sp_converged"]   = False
            for b in BANDS:
                m2 = specparam_band_metrics(sp2, b)
                rec[f"sp_{b}_peak_height"] = m2["peak_height"]
                rec[f"sp_{b}_peak_cf"]     = m2["peak_cf"]
                rec[f"sp_{b}_peak_present"]= m2["peak_present"]

            # ── Method 3: IRASA ───────────────────────────────────────────
            try:
                f_ir, ap_ir, osc_ir, _ = irasa_decompose(x_roi)
                exp_ir, off_ir = irasa_aperiodic(f_ir, ap_ir)
                rec["ir_ap_exponent"] = exp_ir
                rec["ir_ap_offset"]   = off_ir
                for b in BANDS:
                    m3 = irasa_band_metrics(f_ir, osc_ir, ap_ir, b)
                    rec[f"ir_{b}_peak_height"] = m3["peak_height"]
                    rec[f"ir_{b}_peak_cf"]     = m3["peak_cf"]
            except Exception as e:
                rec["ir_ap_exponent"] = np.nan
                rec["ir_ap_offset"]   = np.nan
                for b in BANDS:
                    rec[f"ir_{b}_peak_height"] = np.nan
                    rec[f"ir_{b}_peak_cf"]     = np.nan

            # ── Method 4: Hilbert amplitude ───────────────────────────────
            for b, (lo, hi) in BANDS.items():
                rec[f"hl_{b}_amplitude"] = hilbert_band_amplitude(x_roi, lo, hi)

            all_rows.append(rec)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "all_methods_parameters.csv", index=False)

    print(f"\nspecparam convergence: "
          f"{sp_conv['converged']}/{sp_conv['total']} = "
          f"{100*sp_conv['converged']/max(sp_conv['total'],1):.1f}%")

    # ── PAIRED STATISTICS ─────────────────────────────────────────────────────
    contrasts = [("WS","TS"), ("TS","ESS1"), ("ESS1","LSS1")]
    METHODS   = {
        "Residual":  "res",
        "specparam": "sp",
        "IRASA":     "ir",
        "Hilbert":   "hl",
    }
    METRIC_SUFFIX = {
        "Residual":  "peak_height",
        "specparam": "peak_height",
        "IRASA":     "peak_height",
        "Hilbert":   "amplitude",
    }

    stat_rows = []
    for roi in df["roi"].unique():
        df_roi = df[df["roi"] == roi]
        wide   = df_roi.pivot_table(index="subject", columns="stage",
                                     values=[c for c in df_roi.columns
                                             if c not in ("subject","stage","sex","roi")],
                                     aggfunc="first")
        wide.columns = [f"{m}_{s}" for m, s in wide.columns]
        wide = wide.reset_index()

        for mname, prefix in METHODS.items():
            sfx = METRIC_SUFFIX[mname]
            for b in BANDS:
                col = f"{prefix}_{b}_{sfx}"
                for s1, s2 in contrasts:
                    ca = f"{col}_{s1}"
                    cb = f"{col}_{s2}"
                    if ca not in wide.columns or cb not in wide.columns:
                        continue
                    st = paired_stats(wide[ca].values.astype(float),
                                       wide[cb].values.astype(float))
                    st.update(dict(roi=roi, method=mname, band=b,
                                   contrast=f"{s1}→{s2}"))
                    stat_rows.append(st)

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(out_dir / "paired_stats_all_methods.csv", index=False)

    # ── CROSS-METHOD CORRELATION ──────────────────────────────────────────────
    print("\nCross-method correlation (dz values, ACC ROI):")
    print("-"*60)
    acc_df  = df[df["roi"] == "ACC"]
    dz_data = {}

    for mname, prefix in METHODS.items():
        sfx   = METRIC_SUFFIX[mname]
        dz_vals = []
        for b in BANDS:
            col   = f"{prefix}_{b}_{sfx}"
            acc_w = acc_df.pivot_table(index="subject", columns="stage",
                                        values=col, aggfunc="first")
            for s1, s2 in contrasts:
                if s1 in acc_w.columns and s2 in acc_w.columns:
                    delta = acc_w[s2].values - acc_w[s1].values
                    dz_vals.append(cohens_dz(delta))
        dz_data[mname] = np.array(dz_vals)

    corr_rows = []
    method_list = list(METHODS.keys())
    for i, m1 in enumerate(method_list):
        for j, m2 in enumerate(method_list):
            if j <= i:
                continue
            v1 = dz_data[m1]; v2 = dz_data[m2]
            ok = ~np.isnan(v1) & ~np.isnan(v2)
            if ok.sum() < 5:
                continue
            r, p = pearsonr(v1[ok], v2[ok])
            d_agree = (np.sign(v1[ok]) == np.sign(v2[ok])).mean() * 100
            print(f"  {m1:12s} vs {m2:12s}: r = {r:.3f}  "
                  f"dir.agree = {d_agree:.1f}%  (k={ok.sum()})")
            corr_rows.append(dict(m1=m1, m2=m2, r=round(r,3),
                                   dir_agreement=round(d_agree,1), k=int(ok.sum())))

    pd.DataFrame(corr_rows).to_csv(out_dir / "cross_method_correlations.csv", index=False)

    # ── FIGURE: 4-panel cross-method comparison ───────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    COLORS = {
        "Residual":  "#1f77b4",
        "specparam": "#d62728",
        "IRASA":     "#2ca02c",
        "Hilbert":   "#9467bd",
    }
    BAND_LABELS = list(BANDS.keys())

    # Panel A: dz comparison scatter (Residual vs specparam)
    ax1 = fig.add_subplot(gs[0, 0])
    x_res = dz_data["Residual"]
    for mname in ["specparam", "IRASA", "Hilbert"]:
        y = dz_data[mname]
        ok = ~np.isnan(x_res) & ~np.isnan(y)
        if ok.sum() > 2:
            r, _ = pearsonr(x_res[ok], y[ok])
            ax1.scatter(x_res[ok], y[ok], s=60, alpha=0.8,
                        color=COLORS[mname], label=f"{mname} (r={r:.2f})",
                        edgecolors="white", linewidth=0.5)
    lim = 2.5
    ax1.plot([-lim,lim], [-lim,lim], "k--", lw=1, alpha=0.4)
    ax1.axhline(0, color="gray", lw=0.7); ax1.axvline(0, color="gray", lw=0.7)
    ax1.set_xlabel("Residual-based dz (primary)", fontsize=11)
    ax1.set_ylabel("Alternative method dz", fontsize=11)
    ax1.set_title("A. Effect-Size Agreement\n(all bands × transitions, ACC ROI)",
                  fontweight="bold")
    ax1.legend(fontsize=9)

    # Panel B: Alpha-1 peak height trajectories (all methods)
    ax2 = fig.add_subplot(gs[0, 1])
    for mname, prefix in METHODS.items():
        sfx = METRIC_SUFFIX[mname]
        col = f"{prefix}_alpha1_{sfx}"
        if col not in acc_df.columns:
            continue
        means = acc_df.groupby("stage")[col].mean().reindex(STAGE_ORDER)
        sems  = acc_df.groupby("stage")[col].sem().reindex(STAGE_ORDER)
        ax2.errorbar(range(4), means, yerr=sems, marker="o",
                     lw=2, ms=7, capsize=3,
                     color=COLORS[mname], label=mname)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(STAGE_ORDER)
    ax2.set_ylabel("Peak Height / Amplitude (log₁₀)", fontsize=11)
    ax2.set_title("B. Alpha-1 Trajectories — All Methods\n(ACC ROI)",
                  fontweight="bold")
    ax2.legend(fontsize=9)

    # Panel C: Beta-3 trajectories (false-positive test)
    ax3 = fig.add_subplot(gs[1, 0])
    for mname, prefix in METHODS.items():
        sfx = METRIC_SUFFIX[mname]
        col = f"{prefix}_beta3_{sfx}"
        if col not in acc_df.columns:
            continue
        means = acc_df.groupby("stage")[col].mean().reindex(STAGE_ORDER)
        sems  = acc_df.groupby("stage")[col].sem().reindex(STAGE_ORDER)
        ax3.errorbar(range(4), means, yerr=sems, marker="s",
                     lw=2, ms=7, capsize=3,
                     color=COLORS[mname], label=mname)
        # Also plot band power for residual
        if mname == "Residual":
            bp_col = "res_beta3_band_power"
            if bp_col in acc_df.columns:
                bpm = acc_df.groupby("stage")[bp_col].mean().reindex(STAGE_ORDER)
                ax3.plot(range(4), np.log10(np.maximum(bpm.values, 1e-30)),
                         "k--", lw=1.5, alpha=0.6, label="Band power (log)")
    ax3.set_xticks(range(4)); ax3.set_xticklabels(STAGE_ORDER)
    ax3.set_ylabel("Peak Height / Amplitude (log₁₀)", fontsize=11)
    ax3.set_title("C. Beta-3 Trajectories — All Methods\n(false-positive band, ACC ROI)",
                  fontweight="bold")
    ax3.legend(fontsize=9)

    # Panel D: Aperiodic exponent (3 methods: Residual, specparam, IRASA)
    ax4 = fig.add_subplot(gs[1, 1])
    for mname, col in [("Residual","res_ap_exponent"),
                        ("specparam","sp_ap_exponent"),
                        ("IRASA","ir_ap_exponent")]:
        if col not in acc_df.columns:
            continue
        means = acc_df.groupby("stage")[col].mean().reindex(STAGE_ORDER)
        sems  = acc_df.groupby("stage")[col].sem().reindex(STAGE_ORDER)
        ax4.errorbar(range(4), means, yerr=sems, marker="D",
                     lw=2, ms=7, capsize=3,
                     color=COLORS[mname], label=mname)
    ax4.set_xticks(range(4)); ax4.set_xticklabels(STAGE_ORDER)
    ax4.set_ylabel("Aperiodic Exponent", fontsize=11)
    ax4.set_title("D. Aperiodic Exponent — Three Methods\n(ACC ROI)",
                  fontweight="bold")
    ax4.legend(fontsize=9)

    fig.suptitle(
        "Cross-Method Validation: Residual | specparam (corrected) | IRASA | Hilbert\n"
        f"N = {n_subj} participants | ACC ROI | 7 bands × 3 transitions",
        fontsize=13, fontweight="bold"
    )
    fig.savefig(fig_dir / "cross_method_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── specparam convergence report ──────────────────────────────────────────
    conv_report = out_dir / "specparam_convergence_report.txt"
    with open(conv_report, "w", encoding="utf-8") as f:
        total = sp_conv["total"]
        conv  = sp_conv["converged"]
        f.write("specparam (corrected) Convergence Report\n")
        f.write("="*45 + "\n")
        f.write(f"Parameters used:\n")
        for k, v in SP_PARAMS.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"  freq_range: {SP_FREQ_RANGE}\n")
        f.write(f"  input: LINEAR PSD (μV²/Hz)\n\n")
        f.write(f"Total fits:     {total}\n")
        f.write(f"Converged:      {conv}\n")
        f.write(f"Convergence %:  {100*conv/max(total,1):.1f}%\n\n")
        if "sp_r_squared" in df.columns:
            r2 = df["sp_r_squared"].dropna()
            f.write(f"Mean R²:        {r2.mean():.4f}\n")
            f.write(f"Median R²:      {r2.median():.4f}\n")
            f.write(f"R² ≥ 0.95:      {(r2>=0.95).mean()*100:.1f}%\n")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("RESULTS SAVED:")
    for fname in ["all_methods_parameters.csv",
                  "paired_stats_all_methods.csv",
                  "cross_method_correlations.csv",
                  "specparam_convergence_report.txt",
                  "figures/cross_method_comparison.png"]:
        p = out_dir / fname
        if p.exists():
            print(f"  ✓ {p}")
    print("\n[Pipeline complete]")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
    finally:
        if os.name == "nt":
            os.system("pause")
        else:
            input("\nPress Enter to exit...")
