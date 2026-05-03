# -*- coding: utf-8 -*-
"""
utils.py
========
Shared utility functions used across all pipeline scripts.

Functions are grouped into:
    1. File I/O         — .dat loading, path utilities
    2. Signal processing — PSD, aperiodic decomposition, band metrics
    3. Statistics       — Cohen's dz, paired t-test, correlation
    4. Classification   — transition type assignment, decoupling index
    5. Reporting        — console and CSV helpers
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch

from config import (
    SFREQ, N_CHANNELS, EXPECTED_SAMPLES, MONTAGE, DAT_TO_LC,
    STAGE_PATTERNS, BANDS, ANAT_UNITS,
    WELCH_NPERSEG, WELCH_NOVERLAP, APERIODIC_FIT_RANGE,
    CLS_EFFECT_THRESHOLD, CLS_ZERO_THRESHOLD,
    CLS_VALID, CLS_FP, CLS_PWR_ONLY, CLS_FN, CLS_OPP, CLS_NOREP,
    MIN_N_SUBJECTS,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. FILE I/O
# ══════════════════════════════════════════════════════════════════════════════

def classify_stage(path: Path) -> Optional[str]:
    """Return stage label for a .dat file path, or None if unrecognised."""
    for stage, pattern in STAGE_PATTERNS.items():
        if re.search(pattern, path.name, re.IGNORECASE):
            return stage
    return None


def infer_subject_id(path: Path) -> str:
    """
    Extract subject number from filename.
    wake_30s_04.dat  ->  S004
    trans_30s_61.dat ->  S061
    The subject number is the LAST numeric group before .dat
    """
    m = re.search(r"_(\d+)\.dat$", path.name, re.IGNORECASE)
    if m:
        return f"S{m.group(1).zfill(3)}"
    return path.parent.name


def infer_sex(path: Path) -> str:
    """
    Infer sex from folder name.
    Detects 'female' or 'male' subfolder in path.
    e.g. 1_wake/female/wake_30s_04.dat -> F
         1_wake/male/wake_30s_07.dat   -> M
    """
    # Check each part (works on Windows with real Path objects)
    for part in path.parts:
        if part.lower() == "female":
            return "F"
        if part.lower() == "male":
            return "M"
    # Fallback: search full path string
    path_str = str(path).lower()
    if "\\female\\" in path_str or "/female/" in path_str:
        return "F"
    if "\\male\\" in path_str or "/male/" in path_str:
        return "M"
    return "unknown"


def try_load_text_matrix(path: Path) -> Optional[np.ndarray]:
    """Attempt to load a .dat file as whitespace-delimited text matrix."""
    try:
        arr = np.loadtxt(path)
        return arr if arr.ndim == 2 else None
    except Exception:
        return None


def try_load_binary_matrix(path: Path) -> Optional[np.ndarray]:
    """Attempt to load a .dat file as raw float32/float64 binary."""
    for dtype in [np.float32, np.float64]:
        try:
            arr = np.fromfile(path, dtype=dtype)
            if arr.size == EXPECTED_SAMPLES * N_CHANNELS:
                return arr.reshape(EXPECTED_SAMPLES, N_CHANNELS)
        except Exception:
            pass
    return None


def load_dat_file(path: Path) -> Optional[np.ndarray]:
    """
    Load a .dat EEG file → ndarray (samples × channels) in MONTAGE order.

    Returns None if the file cannot be parsed or has unexpected dimensions.
    """
    arr = try_load_text_matrix(path)
    if arr is None:
        arr = try_load_binary_matrix(path)
    if arr is None:
        return None
    if arr.shape != (EXPECTED_SAMPLES, N_CHANNELS):
        warnings.warn(f"Unexpected shape {arr.shape} in {path.name}")
        return None
    # Rearrange columns from file order to MONTAGE order
    return arr[:, DAT_TO_LC]


def find_dat_candidates(root: Path) -> pd.DataFrame:
    """
    Recursively find all .dat files under root.

    Returns a DataFrame with columns: path, subject, sex, stage.
    Only files matching a known stage pattern are included.
    """
    records = []
    for p in root.rglob("*.dat"):
        stage = classify_stage(p)
        if stage is None:
            continue
        records.append({
            "path":    p,
            "subject": infer_subject_id(p),
            "sex":     infer_sex(p),
            "stage":   stage,
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    # Deduplicate: keep highest-quality candidate per (subject, stage)
    df["score"] = df["path"].apply(lambda p: -len(p.parts))
    df = (df.sort_values("score")
            .drop_duplicates(subset=["subject", "stage"])
            .drop(columns="score")
            .reset_index(drop=True))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def compute_channel_psd(
    eeg: np.ndarray,
    sfreq: float = SFREQ,
    nperseg: int  = WELCH_NPERSEG,
    noverlap: int = WELCH_NOVERLAP,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch PSD for each channel.

    Parameters
    ----------
    eeg : ndarray (samples × channels)

    Returns
    -------
    freqs : ndarray (n_freqs,)
    psd   : ndarray (n_freqs, n_channels)  — linear scale, μV²/Hz
    """
    freqs, psd_0 = welch(eeg[:, 0], fs=sfreq, nperseg=nperseg,
                          noverlap=noverlap, scaling="density")
    psd = np.zeros((len(freqs), eeg.shape[1]))
    psd[:, 0] = psd_0
    for ch in range(1, eeg.shape[1]):
        _, psd[:, ch] = welch(eeg[:, ch], fs=sfreq, nperseg=nperseg,
                               noverlap=noverlap, scaling="density")
    return freqs, psd


def roi_mean_psd(
    psd: np.ndarray,
    freqs: np.ndarray,
    channel_names: List[str],
    roi_channels: List[str],
) -> Optional[np.ndarray]:
    """
    Average PSD across the channels belonging to a ROI.

    Returns None if no channels are found.
    """
    ch_idx = [channel_names.index(ch) for ch in roi_channels
              if ch in channel_names]
    if not ch_idx:
        return None
    return psd[:, ch_idx].mean(axis=1)


def fit_aperiodic(
    freqs: np.ndarray,
    psd_linear: np.ndarray,
    fit_range: Tuple[float, float] = APERIODIC_FIT_RANGE,
    eps: float = 1e-12,
) -> Dict:
    """
    Residual-based aperiodic estimation via log-log linear regression.

    The aperiodic component is modelled as:
        log10(PSD) ≈ offset + slope × log10(freq)
    where exponent = −slope (positive convention).

    The residual (oscillatory component) is:
        residual = log10(PSD) − fitted

    Parameters
    ----------
    freqs      : frequency array (Hz)
    psd_linear : PSD in linear scale (μV²/Hz)
    fit_range  : (fmin, fmax) Hz

    Returns
    -------
    dict with keys:
        aperiodic_exponent : float   positive exponent (−slope)
        aperiodic_offset   : float   intercept in log-log space
        residual_log10     : ndarray oscillatory residual (full freq range)
        log10_psd          : ndarray log10 of PSD (full freq range)
        fitted_log10       : ndarray fitted aperiodic line (full freq range)
    """
    # Avoid log10(0): clamp frequencies to eps before log transform
    safe_freqs = np.maximum(freqs, eps)
    log_f   = np.log10(safe_freqs)
    log_psd = np.log10(np.maximum(psd_linear, eps))

    mask    = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    slope, intercept = np.polyfit(log_f[mask], log_psd[mask], 1)

    fitted   = intercept + slope * log_f
    residual = log_psd - fitted

    return {
        "aperiodic_exponent": float(-slope),
        "aperiodic_offset":   float(intercept),
        "residual_log10":     residual,
        "log10_psd":          log_psd,
        "fitted_log10":       fitted,
    }


def band_power(
    freqs: np.ndarray,
    psd_linear: np.ndarray,
    band: Tuple[float, float],
) -> float:
    """Trapezoidal integration of linear PSD within a frequency band."""
    lo, hi = band
    mask   = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return np.nan
    from scipy.integrate import trapezoid as _trapezoid
    return float(_trapezoid(psd_linear[mask], freqs[mask]))


def band_peak_metrics(
    freqs: np.ndarray,
    residual_log10: np.ndarray,
    log10_psd: np.ndarray,
    band_name: str,
) -> Dict:
    """
    Extract oscillatory peak metrics from the residual spectrum.

    The peak is defined as the frequency within the band where the
    residual (oscillatory component above the aperiodic fit) is maximal.

    Returns
    -------
    dict with keys:
        {band}_peak_cf_hz         : float  centre frequency
        {band}_peak_height_log10  : float  residual at peak (oscillatory amplitude)
        {band}_peak_power_log10   : float  absolute log10 PSD at peak
        {band}_peak_present       : float  1.0 if peak_height > 0, else 0.0
    """
    prefix = band_name
    if band_name == "broadband":
        return {
            f"{prefix}_peak_cf_hz":         np.nan,
            f"{prefix}_peak_height_log10":  np.nan,
            f"{prefix}_peak_power_log10":   np.nan,
            f"{prefix}_peak_present":       np.nan,
        }

    lo, hi = BANDS[band_name]
    mask   = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return {
            f"{prefix}_peak_cf_hz":         np.nan,
            f"{prefix}_peak_height_log10":  np.nan,
            f"{prefix}_peak_power_log10":   np.nan,
            f"{prefix}_peak_present":       0.0,
        }

    band_freqs  = freqs[mask]
    band_resid  = residual_log10[mask]
    band_logpsd = log10_psd[mask]
    idx         = int(np.argmax(band_resid))

    peak_height = float(band_resid[idx])
    return {
        f"{prefix}_peak_cf_hz":         float(band_freqs[idx]),
        f"{prefix}_peak_height_log10":  peak_height,
        f"{prefix}_peak_power_log10":   float(band_logpsd[idx]),
        f"{prefix}_peak_present":       float(peak_height > 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def cohens_dz(before: np.ndarray, after: np.ndarray) -> Tuple[float, float, int, float]:
    """
    Paired Cohen's dz.

    Parameters
    ----------
    before, after : paired observations (nan values are excluded pairwise)

    Returns
    -------
    (dz, p_value, n, mean_delta)
    """
    before = np.asarray(before, dtype=float)
    after  = np.asarray(after,  dtype=float)
    mask   = np.isfinite(before) & np.isfinite(after)
    b, a   = before[mask], after[mask]

    n      = len(b)
    if n < 2:
        return np.nan, np.nan, n, np.nan

    diff   = a - b
    sd     = np.std(diff, ddof=1)
    dz     = float(np.mean(diff) / sd) if sd > 0 else np.nan

    try:
        _, p = stats.ttest_rel(a, b, nan_policy="omit")
    except Exception:
        p = np.nan

    return dz, float(p), n, float(np.mean(diff))


def pearson_r(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    """
    Pearson correlation with pairwise nan exclusion.

    Returns (r, p, n_valid).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, n

    r, p = stats.pearsonr(x, y)
    return float(r), float(p), n


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_transition(
    power_dz:    float,
    peak_dz:     float,
    exp_dz:      float,
    eff_thresh:  float = CLS_EFFECT_THRESHOLD,
    zero_thresh: float = CLS_ZERO_THRESHOLD,
) -> str:
    """
    Classify a band–ROI–contrast observation into one of six categories.

    Decision logic
    --------------
    VALID_OSCILLATORY:
        Both band power and oscillatory peak height show meaningful effects
        in the same direction — genuinely tracks oscillatory change.

    FALSE_POSITIVE:
        Band power shows a meaningful change; peak height is effectively zero;
        and the aperiodic exponent also shows a meaningful change.
        → Band power change is attributable to aperiodic slope shifts,
          not oscillatory amplitude modulation.

    POWER_ONLY:
        Band power shows a meaningful change; peak height is effectively zero;
        but the aperiodic exponent change is sub-threshold.
        → Mechanism unclear; aperiodic attribution cannot be confirmed.

    FALSE_NEGATIVE:
        Oscillatory peak height shows a meaningful change; band power does not.
        → Genuine oscillatory event invisible in the band power metric,
          possibly masked by aperiodic offset or slope changes.

    OPPOSITE_DIRECTION:
        Both metrics show meaningful effects but in opposing directions.
        → Competing mechanisms: aperiodic changes drive band power one way
          while genuine oscillatory activity changes in the other.

    NO_CLEAR_PATTERN:
        Neither metric reaches the effect threshold.
    """
    if not (np.isfinite(power_dz) and np.isfinite(peak_dz)):
        return CLS_NOREP

    pw_sig  = abs(power_dz) >= eff_thresh
    pk_sig  = abs(peak_dz)  >= eff_thresh
    pw_zero = abs(power_dz) <= zero_thresh
    pk_zero = abs(peak_dz)  <= zero_thresh

    if pw_sig and pk_sig and np.sign(power_dz) == np.sign(peak_dz):
        return CLS_VALID

    if pw_sig and pk_zero:
        if np.isfinite(exp_dz) and abs(exp_dz) >= eff_thresh:
            return CLS_FP
        return CLS_PWR_ONLY

    if pw_zero and pk_sig:
        return CLS_FN

    if pw_sig and pk_sig and np.sign(power_dz) != np.sign(peak_dz):
        return CLS_OPP

    return CLS_NOREP


def decoupling_index(
    delta_exp: np.ndarray,
    delta_blp: np.ndarray,
    delta_oph: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute the Decoupling Index (DI).

    DI = |r(Δexponent, ΔBLP)| − |r(Δexponent, ΔOPH)|

    A positive DI indicates that the aperiodic exponent preferentially
    predicts band-limited power (BLP) change over oscillatory peak height
    (OPH) change — the defining signature of a false-positive inference.

    Returns
    -------
    (DI, r_exp_blp, r_exp_oph)
    """
    r_blp, _, _ = pearson_r(delta_exp, delta_blp)
    r_oph, _, _ = pearson_r(delta_exp, delta_oph)

    if np.isfinite(r_blp) and np.isfinite(r_oph):
        di = abs(r_blp) - abs(r_oph)
    else:
        di = np.nan

    return float(di), float(r_blp), float(r_oph)


# ══════════════════════════════════════════════════════════════════════════════
# 5. REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fmt(value: float, decimals: int = 3) -> str:
    """Format a float for display; return '—' for NaN."""
    if not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f}"


def p_stars(p: float) -> str:
    """Convert p-value to significance stars."""
    if not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.010:
        return "**"
    if p < 0.050:
        return "*"
    return "ns"
