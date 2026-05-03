# -*- coding: utf-8 -*-
"""
04_validation_methods.py
========================
Step 4 of the analysis pipeline.

Validates the primary residual-based spectral decomposition against three
independent methods, applied to the same raw .dat files:

    Method 1 (Primary):   Residual-based log-log aperiodic regression
                          Uniform, assumption-light comparative framework.
                          No single method is treated as ground truth;
                          inference is defined by convergence across methods.

    Method 2 (specparam): Parametric aperiodic + Gaussian peak fitting
                          (fooof / specparam v2, knee mode, linear PSD input)

    Method 3 (IRASA):     Irregular-Resampling Auto-Spectral Analysis
                          (Wen & Liu, 2016)

    Method 4 (Hilbert):   Band-filtered Hilbert instantaneous amplitude

For each method, Cohen's dz is computed for all 7 bands × 9 ROIs × 5 contrasts.
Cross-method correlations and directional agreement rates are reported.

Output
------
{OUTPUT_DIR}/
    all_methods_parameters.csv          per-subject stage-level parameters
    paired_stats_all_methods.csv        dz per method × band × roi × contrast
    cross_method_correlations.csv       r, directional agreement between methods
    cross_method_dz_comparison.csv      wide table for Supplementary Table S2

Usage
-----
    python 04_validation_methods.py
    python 04_validation_methods.py --root "D:/mydata"
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt
from scipy.signal import hilbert as scipy_hilbert

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_ROOT, OUTPUT_DIR, SFREQ, MONTAGE, ANAT_UNITS,
    BANDS, BANDS_ANALYSIS, CONTRASTS, CONTRAST_LABELS,
    WELCH_NPERSEG, WELCH_NOVERLAP,
    SP_PARAMS, SP_FREQ_RANGE,
    IRASA_HSET, IRASA_NPERSEG, IRASA_NOVERLAP,
)
from utils import (
    find_dat_candidates, load_dat_file,
    compute_channel_psd, roi_mean_psd,
    fit_aperiodic, band_power, band_peak_metrics,
    cohens_dz, pearson_r, ensure_dir, p_stars,
)

# ── specparam import ──────────────────────────────────────────────────────────
try:
    from specparam import SpectralModel as FOOOF
    _SP_OK = True
except ImportError:
    try:
        from fooof import FOOOF
        _SP_OK = True
    except ImportError:
        _SP_OK = False
        warnings.warn("specparam/fooof not installed. Method 2 will be skipped.")


# ══════════════════════════════════════════════════════════════════════════════
# Method 2: specparam (corrected)
# ══════════════════════════════════════════════════════════════════════════════

def specparam_decompose(freqs: np.ndarray, psd_linear: np.ndarray) -> dict | None:
    """
    Fit specparam to a single spectrum.

    Critical implementation notes:
    - Input must be LINEAR scale PSD (specparam converts internally to log10)
    - aperiodic_mode='knee' is required for sleep EEG
    - Returns None on convergence failure
    """
    if not _SP_OK:
        return None
    try:
        f_mask = (freqs >= SP_FREQ_RANGE[0]) & (freqs <= SP_FREQ_RANGE[1])
        fm     = FOOOF(**SP_PARAMS)
        fm.fit(freqs[f_mask], psd_linear[f_mask])
        return fm
    except Exception:
        return None


def specparam_band_peak(fm, band_name: str) -> dict:
    """Extract peak height within band from a fitted specparam model."""
    prefix = "sp"
    if fm is None:
        return {f"{prefix}_{band_name}_peak_height": np.nan}
    try:
        lo, hi = BANDS[band_name]
        # specparam v2+: use get_results()
        if hasattr(fm, "get_results"):
            peaks = fm.get_results().peak_params
        elif hasattr(fm, "peak_params_"):
            peaks = fm.peak_params_
        else:
            return {f"{prefix}_{band_name}_peak_height": np.nan}

        if peaks is None or len(peaks) == 0:
            return {f"{prefix}_{band_name}_peak_height": 0.0}

        peaks = np.atleast_2d(peaks)
        in_band = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)]
        if len(in_band) == 0:
            return {f"{prefix}_{band_name}_peak_height": 0.0}
        return {f"{prefix}_{band_name}_peak_height": float(in_band[:, 1].max())}
    except Exception:
        return {f"{prefix}_{band_name}_peak_height": np.nan}


def specparam_aperiodic(fm) -> dict:
    """Extract aperiodic parameters from a fitted specparam model."""
    prefix = "sp"
    if fm is None:
        return {f"{prefix}_exponent": np.nan, f"{prefix}_offset": np.nan}
    try:
        # specparam v2+: use get_results()
        if hasattr(fm, "get_results"):
            ap = fm.get_results().aperiodic_params
        # fooof v1 / specparam v1: direct attribute
        elif hasattr(fm, "aperiodic_params_"):
            ap = fm.aperiodic_params_
        else:
            return {f"{prefix}_exponent": np.nan, f"{prefix}_offset": np.nan}
        # knee mode: [offset, knee, exponent]  fixed mode: [offset, exponent]
        if len(ap) == 3:
            return {f"{prefix}_exponent": float(ap[2]), f"{prefix}_offset": float(ap[0])}
        return {f"{prefix}_exponent": float(ap[1]), f"{prefix}_offset": float(ap[0])}
    except Exception:
        return {f"{prefix}_exponent": np.nan, f"{prefix}_offset": np.nan}


# ══════════════════════════════════════════════════════════════════════════════
# Method 3: IRASA
# ══════════════════════════════════════════════════════════════════════════════

def irasa_decompose(
    eeg_roi: np.ndarray,
    hset: np.ndarray = IRASA_HSET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    IRASA decomposition of EEG signal into aperiodic and oscillatory components.

    Returns (freqs, psd_oscillatory, psd_aperiodic) or None on failure.
    """
    try:
        nperseg  = IRASA_NPERSEG
        noverlap = IRASA_NOVERLAP

        freqs_b, psd_total = welch(
            eeg_roi, fs=SFREQ, nperseg=nperseg, noverlap=noverlap
        )

        psds_resampled = []
        for h in hset:
            # Up-sample
            x_up = np.interp(
                np.arange(0, len(eeg_roi), 1/h),
                np.arange(len(eeg_roi)),
                eeg_roi,
            )
            nps_u = int(nperseg * h)
            nov_u = nps_u // 2
            f_u, p_u = welch(x_up, fs=SFREQ * h, nperseg=nps_u, noverlap=nov_u)

            # Down-sample
            x_dn = np.interp(
                np.arange(0, len(eeg_roi), h),
                np.arange(len(eeg_roi)),
                eeg_roi,
            )
            nps_d = max(int(nperseg / h), 4)
            nov_d = nps_d // 2
            f_d, p_d = welch(x_dn, fs=SFREQ / h, nperseg=nps_d, noverlap=nov_d)

            # Interpolate to base frequency grid
            p_u_interp = np.interp(freqs_b, f_u, p_u)
            p_d_interp = np.interp(freqs_b, f_d, p_d)

            psds_resampled.append(np.sqrt(p_u_interp * p_d_interp))

        psd_aperiodic   = np.median(psds_resampled, axis=0)
        psd_oscillatory = psd_total - psd_aperiodic
        psd_oscillatory = np.maximum(psd_oscillatory, 0)

        return freqs_b, psd_oscillatory, psd_aperiodic

    except Exception as e:
        warnings.warn(f"IRASA failed: {e}")
        return None


def irasa_band_peak(
    freqs: np.ndarray,
    psd_osc: np.ndarray,
    band_name: str,
) -> dict:
    """Peak oscillatory power within band from IRASA decomposition."""
    prefix = "ir"
    lo, hi = BANDS[band_name]
    mask   = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return {f"{prefix}_{band_name}_peak_power": np.nan}
    peak_power = float(np.max(psd_osc[mask]))
    return {f"{prefix}_{band_name}_peak_power": np.log10(peak_power + 1e-30)}


def irasa_exponent(freqs: np.ndarray, psd_ap: np.ndarray) -> dict:
    """Estimate aperiodic exponent from IRASA aperiodic component."""
    prefix = "ir"
    mask   = (freqs >= 1.0) & (freqs <= 30.0)
    if not np.any(mask):
        return {f"{prefix}_exponent": np.nan}
    ap = fit_aperiodic(freqs[mask], psd_ap[mask])
    return {f"{prefix}_exponent": ap["aperiodic_exponent"]}


# ══════════════════════════════════════════════════════════════════════════════
# Method 4: Band-filtered Hilbert amplitude
# ══════════════════════════════════════════════════════════════════════════════

def hilbert_band_amplitude(
    eeg_roi: np.ndarray,
    band_name: str,
    order: int = 4,
) -> dict:
    """
    Bandpass filter → Hilbert transform → mean instantaneous power (log10).
    """
    prefix = "hl"
    lo, hi = BANDS[band_name]
    nyq    = SFREQ / 2.0
    b, a   = butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    try:
        xf    = filtfilt(b, a, eeg_roi)
        power = float(np.mean(np.abs(scipy_hilbert(xf)) ** 2))
        return {f"{prefix}_{band_name}_amplitude": np.log10(power + 1e-30)}
    except Exception:
        return {f"{prefix}_{band_name}_amplitude": np.nan}


# ══════════════════════════════════════════════════════════════════════════════
# Main extraction loop
# ══════════════════════════════════════════════════════════════════════════════

def extract_all_methods(path: Path, channel_names: list) -> dict | None:
    """Extract parameters for all four methods from one .dat file."""
    from scipy.signal import welch as _welch

    eeg = load_dat_file(path)
    if eeg is None:
        return None

    try:
        freqs, psd_ch = compute_channel_psd(eeg)
    except Exception:
        return None

    roi_results = {}
    for roi_name, roi_channels in ANAT_UNITS.items():
        psd_roi = roi_mean_psd(psd_ch, freqs, channel_names, roi_channels)
        if psd_roi is None:
            continue

        # Mean EEG signal for ROI (for IRASA and Hilbert)
        ch_idx  = [channel_names.index(ch) for ch in roi_channels
                   if ch in channel_names]
        eeg_roi = eeg[:, ch_idx].mean(axis=1)

        rec = {}

        # ── Method 1: Residual-based ───────────────────────────────────────
        ap = fit_aperiodic(freqs, psd_roi)
        rec["res_exponent"] = ap["aperiodic_exponent"]
        rec["res_offset"]   = ap["aperiodic_offset"]
        for b in BANDS_ANALYSIS:
            lo, hi = BANDS[b]
            bp     = band_power(freqs, psd_roi, (lo, hi))
            rec[f"res_{b}_power"]  = np.log10(bp) if bp > 0 else np.nan
            rec.update(band_peak_metrics(
                freqs, ap["residual_log10"], ap["log10_psd"], b
            ))

        # ── Method 2: specparam ────────────────────────────────────────────
        fm = specparam_decompose(freqs, psd_roi)
        rec.update(specparam_aperiodic(fm))
        for b in BANDS_ANALYSIS:
            rec.update(specparam_band_peak(fm, b))

        # ── Method 3: IRASA ────────────────────────────────────────────────
        irasa_result = irasa_decompose(eeg_roi)
        if irasa_result is not None:
            f_ir, psd_osc, psd_ap = irasa_result
            rec.update(irasa_exponent(f_ir, psd_ap))
            for b in BANDS_ANALYSIS:
                rec.update(irasa_band_peak(f_ir, psd_osc, b))
        else:
            rec["ir_exponent"] = np.nan
            for b in BANDS_ANALYSIS:
                rec[f"ir_{b}_peak_power"] = np.nan

        # ── Method 4: Hilbert ──────────────────────────────────────────────
        for b in BANDS_ANALYSIS:
            rec.update(hilbert_band_amplitude(eeg_roi, b))

        roi_results[roi_name] = rec

    return roi_results


def run(root: Path, out_dir: Path) -> pd.DataFrame:
    ensure_dir(out_dir)

    candidates    = find_dat_candidates(root)
    channel_names = MONTAGE

    print(f"Validation: {len(candidates)} files")
    print(f"Methods: Residual | specparam ({'OK' if _SP_OK else 'SKIP'}) | IRASA | Hilbert")

    all_rows = []
    for _, file_row in candidates.iterrows():
        roi_params = extract_all_methods(file_row["path"], channel_names)
        if roi_params is None:
            continue
        for roi_name, params in roi_params.items():
            rec = {
                "subject": file_row["subject"],
                "sex":     file_row["sex"],
                "stage":   file_row["stage"],
                "roi":     roi_name,
            }
            rec.update(params)
            all_rows.append(rec)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "all_methods_parameters.csv", index=False)

    # ── Paired statistics for all methods ─────────────────────────────────────
    METHODS = {
        "Residual":  ("res", "peak_height_log10"),
        "specparam": ("sp",  "peak_height"),
        "IRASA":     ("ir",  "peak_power"),
        "Hilbert":   ("hl",  "amplitude"),
    }

    stat_rows = []
    for roi in df["roi"].unique():
        df_roi = df[df["roi"] == roi]
        wide   = df_roi.pivot_table(
            index="subject", columns="stage",
            values=[c for c in df_roi.columns if c not in ("subject","sex","stage","roi")],
            aggfunc="first",
        )
        wide.columns = [f"{m}_{s}" for m, s in wide.columns]
        wide = wide.reset_index()

        for mname, (prefix, suffix) in METHODS.items():
            for b in BANDS_ANALYSIS:
                col = f"{prefix}_{b}_{suffix}"
                if mname == "Residual":
                    col = f"{b}_peak_height_log10"

                for s1, s2 in CONTRASTS:
                    ca = f"{col}_{s1}"
                    cb = f"{col}_{s2}"
                    if ca not in wide.columns or cb not in wide.columns:
                        continue
                    dz, p, n, md = cohens_dz(
                        wide[ca].values.astype(float),
                        wide[cb].values.astype(float),
                    )
                    stat_rows.append(dict(
                        roi=roi, method=mname, band=b,
                        contrast=f"{s1}->{s2}", n=n,
                        dz=dz, p=p, stars=p_stars(p), mean_delta=md,
                    ))

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(out_dir / "paired_stats_all_methods.csv", index=False)

    # ── Cross-method correlations ──────────────────────────────────────────────
    # Per (roi, contrast): correlate dz vectors across bands
    corr_rows = []
    method_list = list(METHODS.keys())
    for roi in stat_df["roi"].unique():
        for contrast in stat_df["contrast"].unique():
            sub = stat_df[
                (stat_df["roi"] == roi) & (stat_df["contrast"] == contrast)
            ]
            dz_data = {}
            for m in method_list:
                vals = sub[sub["method"] == m].set_index("band")["dz"]
                dz_data[m] = np.array([
                    vals.get(b, np.nan) for b in BANDS_ANALYSIS
                ])

            for i, m1 in enumerate(method_list):
                for m2 in method_list[i+1:]:
                    v1, v2  = dz_data[m1], dz_data[m2]
                    ok      = np.isfinite(v1) & np.isfinite(v2)
                    if ok.sum() < 3:
                        continue
                    r, p, _ = pearson_r(v1[ok], v2[ok])
                    d_agree = (np.sign(v1[ok]) == np.sign(v2[ok])).mean() * 100
                    corr_rows.append(dict(
                        roi=roi, contrast=contrast,
                        method1=m1, method2=m2,
                        r=r, p=p, dir_agreement_pct=d_agree,
                        n_bands=int(ok.sum()),
                    ))

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out_dir / "cross_method_correlations.csv", index=False)

    # ── Wide dz comparison table (Supplementary Table S2) ─────────────────────
    dz_wide = stat_df.pivot_table(
        index=["roi", "band", "contrast"],
        columns="method",
        values="dz",
        aggfunc="first",
    ).reset_index()
    dz_wide.to_csv(out_dir / "cross_method_dz_comparison.csv", index=False)

    print(f"\nValidation complete.")
    print(f"  Parameters: {len(df)} rows")
    print(f"  Statistics: {len(stat_df)} rows")
    print(f"\nFC-midline cross-method correlations (all contrasts):")
    fc_corr = corr_df[corr_df["roi"].isin(["FC_midline", "ACC"])]
    print(fc_corr[["contrast","method1","method2","r","dir_agreement_pct"]]
          .sort_values("contrast").to_string(index=False))

    print(f"\nOutput: {out_dir}")
    return stat_df


def main():
    parser = argparse.ArgumentParser(
        description="Cross-method spectral validation."
    )
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--out",  type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run(Path(args.root), Path(args.out))


if __name__ == "__main__":
    main()
