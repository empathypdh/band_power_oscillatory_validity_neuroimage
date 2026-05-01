
# -*- coding: utf-8 -*-
"""
Paper 4 full-band + 3-level spectral decomposition pipeline (v6)

What this does
--------------
1) Finds the 61 subjects and 4 SOP stages from *.dat files under C:\SOP_data
2) Computes PSD (Welch) for each subject-stage
3) Runs spectral decomposition across 8 bands:
   delta, theta, alpha1, alpha2, beta1, beta2, beta3, broadband
4) Runs the same analysis at 3 spatial levels:
   - electrode level (30 electrodes)
   - lobe ROI level (5 bilateral lobes)
   - anatomical ROI level (9 ROIs)
5) Writes stage means and paired statistics (WS->TS, TS->ESS1, ESS1->LSS1)
6) Produces basic comparison figures

Notes
-----
- Aperiodic fit is global (1-30 Hz) and shared within each spatial unit.
- Peak-based metrics are band-specific, computed from the residual after aperiodic subtraction.
- Broadband peak metrics are omitted by design because they are not meaningful as a single narrowband peak.
"""

import os
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy import stats
import matplotlib.pyplot as plt

# =========================================================
# Core settings
# =========================================================

SFREQ = 500
N_CHANNELS = 30
EXPECTED_SAMPLES = 15000

DEFAULT_ROOT = Path(r"C:\SOP_data")
RESULT_SUBDIR = "paper4_fullband_3level_results_v6"

STRICT_STAGE_FILE_PATTERNS = {
    "WS": re.compile(r"wake_30s_(\d+)\.dat$", re.I),
    "TS": re.compile(r"trans_30s_(\d+)\.dat$", re.I),
    "ESS1": re.compile(r"early_30s_(\d+)\.dat$", re.I),
    "LSS1": re.compile(r"late_30s_(\d+)\.dat$", re.I),
}

EXPECTED_STAGE_ORDER = ["WS", "TS", "ESS1", "LSS1"]

# Original LC order from your prior pipelines
MONTAGE = [
    "O2","O1","Oz","Pz","P4","CP4","P8/T6","C4","TP8","T8/T4",
    "P7/T5","P3","CP3","CPz","Cz","FC4","FT8","TP7","C3","FCz",
    "Fz","F4","F8","T7/T3","FT7","FC3","F3","Fp2","F7","Fp1"
]

DAT_TO_LC = [29,27,28,26,20,21,22,23,18,14,
              7, 9,10,11, 3, 4, 6, 1, 2, 0,
             25,19,15,12,13, 5,24,16,17, 8]

BANDS = {
    "delta": (1.0, 3.5),
    "theta": (4.0, 7.5),
    "alpha1": (8.0, 10.0),
    "alpha2": (10.5, 12.0),
    "beta1": (12.5, 18.0),
    "beta2": (18.5, 21.0),
    "beta3": (21.5, 30.0),
    "broadband": (1.0, 30.0),
}

# =========================================================
# Three spatial levels
# =========================================================

# Level 1: electrode = each single channel
ELECTRODE_UNITS = {ch: [ch] for ch in MONTAGE}

# Level 2: 5 bilateral lobes (chosen to mirror the prior paper conceptually)
LOBE_UNITS = {
    "Frontal":    ["Fp1","Fp2","F7","F3","Fz","F4","F8","FC3","FCz","FC4"],
    "Central":    ["C3","Cz","C4","CP3","CPz","CP4"],
    "Parietal":   ["P3","Pz","P4"],
    "Temporal":   ["FT7","T7/T3","TP7","P7/T5","FT8","T8/T4","TP8","P8/T6"],
    "Occipital":  ["O1","Oz","O2"],
}

# Level 3: 9 anatomical ROIs from your supplementary mapping
# Source: prior uploaded supplementary doc (Figure S1 ROI membership)
ANAT_UNITS = {
    "L_DLPFC": ["Fp1","F3","FC3"],
    "R_DLPFC": ["Fp2","F4","FC4"],
    "VmPFC_OrbPFC": ["Fp1","Fp2","Fz","FCz"],
    "ACC": ["Fz","FCz","Cz","CPz"],
    "SPG": ["P3","Pz","P4"],
    "IPG": ["P7/T5","P8/T6","TP7","TP8"],
    "L_STG": ["F7","FT7","T7/T3"],
    "R_STG": ["F8","FT8","T8/T4"],
    "Occipital": ["O1","Oz","O2"],
}

# =========================================================
# Utilities
# =========================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def choose_root_folder():
    if DEFAULT_ROOT.exists():
        return DEFAULT_ROOT
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Paper 4 v6",
            "C:\\SOP_data was not found.\nPlease select your SOP_data folder."
        )
        selected = filedialog.askdirectory(title="Select SOP_data folder")
        root.destroy()
        if selected:
            return Path(selected)
    except Exception:
        pass
    text = input("C:\\SOP_data not found. Please type the SOP_data folder path: ").strip().strip('"')
    if not text:
        raise FileNotFoundError("No root folder selected.")
    return Path(text)

def classify_stage(path: Path):
    name = path.name
    for stage, pat in STRICT_STAGE_FILE_PATTERNS.items():
        if pat.search(name):
            return stage
    return None

def infer_subject_number(path: Path):
    name = path.name
    for stage, pat in STRICT_STAGE_FILE_PATTERNS.items():
        m = pat.search(name)
        if m:
            return int(m.group(1))
    return None

def infer_subject_id(path: Path):
    num = infer_subject_number(path)
    if num is None:
        return None
    return f"S{num:02d}"

def infer_sex(path: Path):
    txt = str(path).lower().replace("/", "\\")
    if "\\female\\" in txt:
        return "female"
    if "\\male\\" in txt:
        return "male"
    return "unknown"

def candidate_score(path: Path):
    txt = str(path).lower()
    score = 0
    if "results" in txt:
        score += 1000
    if "backup" in txt or "old" in txt or "copy" in txt:
        score += 500
    if "analyzed data" in txt:
        score -= 50
    if "\\female\\" in txt or "\\male\\" in txt:
        score -= 10
    return (score, len(txt), txt)

def find_dat_candidates(root: Path):
    rows = []
    for p in root.rglob("*.dat"):
        stage = classify_stage(p)
        subj = infer_subject_id(p)
        if stage is None or subj is None:
            continue
        rows.append({
            "path": str(p),
            "stage": stage,
            "subject": subj,
            "subject_num": infer_subject_number(p),
            "sex": infer_sex(p),
            "score1": candidate_score(p)[0],
            "score2": candidate_score(p)[1],
            "score3": candidate_score(p)[2],
        })
    return pd.DataFrame(rows)

def deduplicate_candidates(df: pd.DataFrame, out_dir: Path):
    if df.empty:
        return df.copy()

    df = df.sort_values(["subject", "stage", "score1", "score2", "score3"]).reset_index(drop=True)
    dup_mask = df.duplicated(subset=["subject", "stage"], keep=False)
    dup_df = df[dup_mask].copy()
    if not dup_df.empty:
        dup_df.to_csv(out_dir / "duplicates_report.csv", index=False)

    keep_df = df.drop_duplicates(subset=["subject", "stage"], keep="first").copy()
    keep_df.to_csv(out_dir / "selected_input_files.csv", index=False)
    return keep_df

def try_load_text_matrix(path: Path):
    loaders = [
        lambda p: np.loadtxt(p, dtype=float),
        lambda p: pd.read_csv(p, header=None, sep=r"\s+", engine="python").values.astype(float),
        lambda p: pd.read_csv(p, header=None).values.astype(float),
    ]
    for loader in loaders:
        try:
            arr = loader(path)
            if arr.size > 0:
                return np.asarray(arr, dtype=float)
        except Exception:
            continue
    return None

def try_load_binary_matrix(path: Path):
    raw = path.read_bytes()
    for dtype in (np.float32, np.float64, np.int16, np.int32):
        try:
            arr = np.frombuffer(raw, dtype=dtype)
            if arr.size == 0:
                continue
            for shape in ((EXPECTED_SAMPLES, N_CHANNELS), (N_CHANNELS, EXPECTED_SAMPLES)):
                total = shape[0] * shape[1]
                if arr.size >= total:
                    out = arr[:total].reshape(shape).astype(float)
                    return out
        except Exception:
            continue
    return None

def reshape_eeg(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if arr.size % N_CHANNELS != 0:
            raise ValueError(f"1D array size {arr.size} is not divisible by {N_CHANNELS}.")
        arr = arr.reshape(-1, N_CHANNELS)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D EEG matrix, got shape {arr.shape}.")

    if arr.shape[1] == N_CHANNELS:
        eeg = arr
    elif arr.shape[0] == N_CHANNELS:
        eeg = arr.T
    else:
        if abs(arr.shape[0] - EXPECTED_SAMPLES) < abs(arr.shape[1] - EXPECTED_SAMPLES):
            eeg = arr
        else:
            eeg = arr.T
        if eeg.shape[1] != N_CHANNELS:
            raise ValueError(f"Cannot identify channel dimension in shape {arr.shape}.")

    return eeg[:, DAT_TO_LC]

def load_dat_file(path: Path):
    arr = try_load_text_matrix(path)
    if arr is None:
        arr = try_load_binary_matrix(path)
    if arr is None:
        raise ValueError(f"Could not parse file: {path}")
    return reshape_eeg(arr)

def compute_channel_psd(eeg, sfreq=SFREQ):
    nperseg = min(int(4 * sfreq), eeg.shape[0])
    noverlap = nperseg // 2
    freqs, psd = welch(
        eeg,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=0,
        scaling="density",
        detrend="constant",
    )
    mask = (freqs >= 1.0) & (freqs <= 30.0)
    return freqs[mask], psd[mask, :]

def band_mean(freqs, spectrum, band):
    lo, hi = band
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return np.nan
    return float(np.mean(spectrum[m]))

def fit_global_aperiodic(freqs, spectrum, eps=1e-12):
    logf = np.log10(freqs)
    logp = np.log10(np.maximum(spectrum, eps))
    fit_mask = (freqs >= 1.0) & (freqs <= 30.0)
    x = logf[fit_mask]
    y = logp[fit_mask]
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * logf
    residual = logp - fitted
    return {
        "aperiodic_offset": float(intercept),
        "aperiodic_exponent": float(-slope),
        "fitted_log10": fitted,
        "residual_log10": residual,
        "log10_psd": logp,
    }

def fit_band_peak(freqs, residual_log10, log10_psd, band_name):
    if band_name == "broadband":
        return {
            f"{band_name}_peak_cf_hz": np.nan,
            f"{band_name}_peak_height_log10": np.nan,
            f"{band_name}_peak_power_log10": np.nan,
            f"{band_name}_peak_present": np.nan,
        }

    lo, hi = BANDS[band_name]
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return {
            f"{band_name}_peak_cf_hz": np.nan,
            f"{band_name}_peak_height_log10": np.nan,
            f"{band_name}_peak_power_log10": np.nan,
            f"{band_name}_peak_present": 0.0,
        }

    band_freqs = freqs[m]
    band_resid = residual_log10[m]
    band_logp = log10_psd[m]
    idx = int(np.argmax(band_resid))
    peak_cf = float(band_freqs[idx])
    peak_height = float(band_resid[idx])
    peak_power = float(band_logp[idx])

    return {
        f"{band_name}_peak_cf_hz": peak_cf,
        f"{band_name}_peak_height_log10": peak_height,
        f"{band_name}_peak_power_log10": peak_power,
        f"{band_name}_peak_present": float(peak_height > 0),
    }

def cohens_dz_from_delta(delta):
    delta = np.asarray(delta, dtype=float)
    delta = delta[~np.isnan(delta)]
    if delta.size < 2:
        return np.nan
    sd = np.std(delta, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(delta) / sd)

def paired_stats(df_wide: pd.DataFrame, metric: str, a: str, b: str):
    xa = f"{metric}_{a}"
    xb = f"{metric}_{b}"
    if xa not in df_wide.columns or xb not in df_wide.columns:
        return {"n": 0, "mean_delta": np.nan, "t": np.nan, "p": np.nan, "dz": np.nan}
    x = df_wide[xa].values
    y = df_wide[xb].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return {"n": int(x.size), "mean_delta": np.nan, "t": np.nan, "p": np.nan, "dz": np.nan}
    delta = y - x
    t, p = stats.ttest_rel(y, x)
    return {
        "n": int(x.size),
        "mean_delta": float(np.mean(delta)),
        "t": float(t),
        "p": float(p),
        "dz": cohens_dz_from_delta(delta),
    }

def unit_spectrum_from_psd(psd_by_channel, channel_lookup, unit_channels):
    idx = [channel_lookup[ch] for ch in unit_channels if ch in channel_lookup]
    if not idx:
        return None
    return np.mean(psd_by_channel[:, idx], axis=1)

def write_validation_summary(out_dir: Path, selected_df: pd.DataFrame, completeness_df: pd.DataFrame):
    total_subjects = selected_df["subject"].nunique() if not selected_df.empty else 0
    complete_subjects = int((completeness_df["n_stages_found"] == 4).sum()) if not completeness_df.empty else 0
    incomplete_subjects = int((completeness_df["n_stages_found"] < 4).sum()) if not completeness_df.empty else 0

    with open(out_dir / "validation_summary.txt", "w", encoding="utf-8") as f:
        f.write("Paper 4 full-band 3-level v6 validation summary\n")
        f.write("=" * 54 + "\n")
        f.write(f"Unique subjects selected: {total_subjects}\n")
        f.write(f"Complete subjects (all 4 stages): {complete_subjects}\n")
        f.write(f"Incomplete subjects: {incomplete_subjects}\n\n")
        f.write("Expected target subjects: 61\n")
        f.write(f"Matches expected 61? {'YES' if total_subjects == 61 else 'NO'}\n")
        f.write(f"All complete? {'YES' if incomplete_subjects == 0 else 'NO'}\n")
        f.write("Spatial levels: electrode, lobe, anatomical ROI\n")
        f.write("Bands: " + ", ".join(BANDS.keys()) + "\n")

def save_mapping_files(out_dir: Path):
    (out_dir / "paper4_montage.txt").write_text("\n".join(MONTAGE), encoding="utf-8")
    (out_dir / "paper4_bands.txt").write_text(
        "\n".join([f"{k}: {v[0]}-{v[1]} Hz" for k, v in BANDS.items()]),
        encoding="utf-8"
    )
    with open(out_dir / "paper4_lobe_mapping.txt", "w", encoding="utf-8") as f:
        for k, v in LOBE_UNITS.items():
            f.write(f"{k}: {', '.join(v)}\n")
    with open(out_dir / "paper4_anatomical_roi_mapping.txt", "w", encoding="utf-8") as f:
        for k, v in ANAT_UNITS.items():
            f.write(f"{k}: {', '.join(v)}\n")

def summarize_level(params_df: pd.DataFrame, level_name: str, out_dir: Path, fig_dir: Path):
    stage_order = pd.CategoricalDtype(EXPECTED_STAGE_ORDER, ordered=True)
    params_df = params_df.copy()
    params_df["stage"] = params_df["stage"].astype(stage_order)
    params_df = params_df.sort_values(["unit", "subject", "stage"]).reset_index(drop=True)

    params_df.to_csv(out_dir / f"{level_name}_parameters.csv", index=False)

    metric_cols = []
    for c in params_df.columns:
        if c in ["subject", "stage", "sex", "unit", "level", "source_file"]:
            continue
        if np.issubdtype(params_df[c].dtype, np.number):
            metric_cols.append(c)

    stage_means = params_df.groupby(["unit", "stage"], observed=True)[metric_cols].mean(numeric_only=True).reset_index()
    stage_means.to_csv(out_dir / f"{level_name}_stage_means.csv", index=False)

    wide = params_df.pivot_table(index=["subject", "unit"], columns="stage", values=metric_cols, aggfunc="first")
    wide.columns = [f"{m}_{s}" for m, s in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(out_dir / f"{level_name}_subject_stage_parameters_wide.csv", index=False)

    contrasts = [("WS", "TS"), ("TS", "ESS1"), ("ESS1", "LSS1")]
    stat_rows = []
    for unit in sorted(params_df["unit"].unique()):
        unit_wide = wide[wide["unit"] == unit].copy()
        for metric in metric_cols:
            for a, b in contrasts:
                res = paired_stats(unit_wide, metric, a, b)
                res.update({"unit": unit, "metric": metric, "contrast": f"{a}->{b}", "level": level_name})
                stat_rows.append(res)

    stats_df = pd.DataFrame(stat_rows)
    stats_df = stats_df[["level","unit","metric","contrast","n","mean_delta","t","p","dz"]]
    stats_df.to_csv(out_dir / f"{level_name}_delta_stats.csv", index=False)

    # Simple figure: band power trajectories by unit mean over units
    power_metrics = [f"power_{band}_mean" for band in BANDS.keys()]
    plot_df = params_df.groupby("stage", observed=True)[[m for m in power_metrics if m in params_df.columns]].mean(numeric_only=True).reset_index()
    plt.figure(figsize=(10, 5))
    for m in [pm for pm in power_metrics if pm in plot_df.columns]:
        plt.plot(plot_df["stage"], plot_df[m], marker="o", label=m.replace("power_","").replace("_mean",""))
    plt.title(f"{level_name}: mean band-power trajectories")
    plt.xlabel("Stage")
    plt.ylabel("Mean power")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{level_name}_band_power_trajectories.png", dpi=300)
    plt.close()

def main():
    print("=" * 78)
    print("Paper 4 full-band + 3-level spectral decomposition pipeline (v6)")
    print("=" * 78)

    root = choose_root_folder()
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    out_dir = root / RESULT_SUBDIR
    fig_dir = out_dir / "figures"
    ensure_dir(out_dir)
    ensure_dir(fig_dir)
    save_mapping_files(out_dir)

    cand_df = find_dat_candidates(root)
    if cand_df.empty:
        raise FileNotFoundError(
            f"No stage-classifiable .dat files found under: {root}\n"
            "Expected files like wake_30s_XX.dat / trans_30s_XX.dat / early_30s_XX.dat / late_30s_XX.dat."
        )

    selected_df = deduplicate_candidates(cand_df, out_dir)
    if selected_df.empty:
        raise RuntimeError("No selected files remained after deduplication.")

    electrode_rows = []
    lobe_rows = []
    anat_rows = []
    psd_rows = []

    channel_lookup = {ch: i for i, ch in enumerate(MONTAGE)}

    for i, row in enumerate(selected_df.itertuples(index=False), start=1):
        path = Path(row.path)
        stage = row.stage
        subj = row.subject
        sex = row.sex
        print(f"[{i}/{len(selected_df)}] {subj} {stage} ({sex}) -> {path.name}")

        eeg = load_dat_file(path)
        freqs, psd_ch = compute_channel_psd(eeg, sfreq=SFREQ)

        # Save channel-average PSD long
        psd_mean = np.mean(psd_ch, axis=1)
        for f, p in zip(freqs, psd_mean):
            psd_rows.append({
                "subject": subj,
                "stage": stage,
                "sex": sex,
                "freq_hz": float(f),
                "psd_mean": float(p),
            })

        # ---------- Level 1: electrode ----------
        for unit, channels in ELECTRODE_UNITS.items():
            spec = unit_spectrum_from_psd(psd_ch, channel_lookup, channels)
            if spec is None:
                continue
            apo = fit_global_aperiodic(freqs, spec)
            rec = {
                "level": "electrode",
                "unit": unit,
                "subject": subj,
                "stage": stage,
                "sex": sex,
                "source_file": str(path),
            }
            rec["aperiodic_offset"] = apo["aperiodic_offset"]
            rec["aperiodic_exponent"] = apo["aperiodic_exponent"]
            for band in BANDS:
                rec[f"power_{band}_mean"] = band_mean(freqs, spec, BANDS[band])
                rec.update(fit_band_peak(freqs, apo["residual_log10"], apo["log10_psd"], band))
            electrode_rows.append(rec)

        # ---------- Level 2: lobe ----------
        for unit, channels in LOBE_UNITS.items():
            spec = unit_spectrum_from_psd(psd_ch, channel_lookup, channels)
            if spec is None:
                continue
            apo = fit_global_aperiodic(freqs, spec)
            rec = {
                "level": "lobe",
                "unit": unit,
                "subject": subj,
                "stage": stage,
                "sex": sex,
                "source_file": str(path),
            }
            rec["aperiodic_offset"] = apo["aperiodic_offset"]
            rec["aperiodic_exponent"] = apo["aperiodic_exponent"]
            for band in BANDS:
                rec[f"power_{band}_mean"] = band_mean(freqs, spec, BANDS[band])
                rec.update(fit_band_peak(freqs, apo["residual_log10"], apo["log10_psd"], band))
            lobe_rows.append(rec)

        # ---------- Level 3: anatomical ROI ----------
        for unit, channels in ANAT_UNITS.items():
            spec = unit_spectrum_from_psd(psd_ch, channel_lookup, channels)
            if spec is None:
                continue
            apo = fit_global_aperiodic(freqs, spec)
            rec = {
                "level": "anatomical_roi",
                "unit": unit,
                "subject": subj,
                "stage": stage,
                "sex": sex,
                "source_file": str(path),
            }
            rec["aperiodic_offset"] = apo["aperiodic_offset"]
            rec["aperiodic_exponent"] = apo["aperiodic_exponent"]
            for band in BANDS:
                rec[f"power_{band}_mean"] = band_mean(freqs, spec, BANDS[band])
                rec.update(fit_band_peak(freqs, apo["residual_log10"], apo["log10_psd"], band))
            anat_rows.append(rec)

    # Save PSD long
    psd_df = pd.DataFrame(psd_rows)
    stage_order = pd.CategoricalDtype(EXPECTED_STAGE_ORDER, ordered=True)
    psd_df["stage"] = psd_df["stage"].astype(stage_order)
    psd_df = psd_df.sort_values(["subject", "stage", "freq_hz"]).reset_index(drop=True)
    psd_df.to_csv(out_dir / "paper4_psd_long.csv", index=False)

    # Save all levels
    electrode_df = pd.DataFrame(electrode_rows)
    lobe_df = pd.DataFrame(lobe_rows)
    anat_df = pd.DataFrame(anat_rows)

    summarize_level(electrode_df, "electrode", out_dir, fig_dir)
    summarize_level(lobe_df, "lobe", out_dir, fig_dir)
    summarize_level(anat_df, "anatomical_roi", out_dir, fig_dir)

    completeness = selected_df.groupby("subject", observed=True)["stage"].nunique().reset_index()
    completeness.columns = ["subject", "n_stages_found"]
    completeness.to_csv(out_dir / "paper4_subject_completeness.csv", index=False)

    write_validation_summary(out_dir, selected_df, completeness)

    print("\nDone.")
    print(f"Results folder: {out_dir}")
    print("Main outputs:")
    for name in [
        "selected_input_files.csv",
        "duplicates_report.csv",
        "validation_summary.txt",
        "paper4_psd_long.csv",
        "electrode_parameters.csv",
        "electrode_stage_means.csv",
        "electrode_subject_stage_parameters_wide.csv",
        "electrode_delta_stats.csv",
        "lobe_parameters.csv",
        "lobe_stage_means.csv",
        "lobe_subject_stage_parameters_wide.csv",
        "lobe_delta_stats.csv",
        "anatomical_roi_parameters.csv",
        "anatomical_roi_stage_means.csv",
        "anatomical_roi_subject_stage_parameters_wide.csv",
        "anatomical_roi_delta_stats.csv",
        "paper4_lobe_mapping.txt",
        "paper4_anatomical_roi_mapping.txt",
    ]:
        path = out_dir / name
        if path.exists():
            print(f" - {path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]")
        print(str(e))
        print()
        traceback.print_exc()
    finally:
        try:
            if os.name == "nt":
                os.system("pause")
            else:
                input("Press Enter to exit...")
        except Exception:
            pass
