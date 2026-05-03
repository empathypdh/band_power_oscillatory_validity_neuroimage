# -*- coding: utf-8 -*-
"""
01_extract_parameters.py
========================
Step 1 of the analysis pipeline.

Reads raw EEG .dat files → computes spectral parameters → saves wide CSV.

This script is the sole entry point for raw data. All subsequent scripts
(02–06) work from the CSV produced here, ensuring reproducibility without
repeated raw-data access.

Output
------
{OUTPUT_DIR}/
    anatomical_roi_parameters_wide.csv    primary output (N_subj × metrics)
    stage_means_by_roi.csv                group-level means per stage/ROI/metric
    subject_completeness.csv              quality check: stages found per subject
    validation_log.txt                    warnings and completeness summary

Usage
-----
    python 01_extract_parameters.py
    python 01_extract_parameters.py --root "D:/mydata"
    python 01_extract_parameters.py --root "D:/mydata" --out "D:/results"
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path when running from subdirectory
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEFAULT_ROOT, OUTPUT_DIR, SFREQ, MONTAGE, ANAT_UNITS,
    BANDS, BANDS_ANALYSIS, STAGE_ORDER, MIN_N_SUBJECTS,
)
from utils import (
    find_dat_candidates, load_dat_file,
    compute_channel_psd, roi_mean_psd,
    fit_aperiodic, band_power, band_peak_metrics,
    ensure_dir,
)


# ── Main extraction logic ──────────────────────────────────────────────────────

def extract_subject_stage(
    path: Path,
    channel_names: list,
) -> dict | None:
    """
    Load one .dat file and extract all spectral parameters.

    Returns a flat dict of parameters, or None on failure.
    """
    eeg = load_dat_file(path)
    if eeg is None:
        warnings.warn(f"Could not load: {path.name}")
        return None

    try:
        freqs, psd_ch = compute_channel_psd(eeg)
    except Exception as e:
        warnings.warn(f"PSD failed for {path.name}: {e}")
        return None

    records = {}
    for roi_name, roi_channels in ANAT_UNITS.items():
        psd_roi = roi_mean_psd(psd_ch, freqs, channel_names, roi_channels)
        if psd_roi is None:
            warnings.warn(f"No channels found for ROI {roi_name} in {path.name}")
            continue

        # Aperiodic decomposition
        ap = fit_aperiodic(freqs, psd_roi)

        rec = {
            "aperiodic_exponent": ap["aperiodic_exponent"],
            "aperiodic_offset":   ap["aperiodic_offset"],
        }

        # Band-limited power and oscillatory peak metrics
        for band_name in BANDS:
            lo, hi = BANDS[band_name]
            bp = band_power(freqs, psd_roi, (lo, hi))
            rec[f"power_{band_name}"] = bp
            rec[f"log10_power_{band_name}"] = (
                float(np.log10(bp)) if bp > 0 else np.nan
            )
            if band_name != "broadband":
                rec.update(band_peak_metrics(
                    freqs,
                    ap["residual_log10"],
                    ap["log10_psd"],
                    band_name,
                ))

        records[roi_name] = rec

    return records


def run(root: Path, out_dir: Path) -> pd.DataFrame:
    """
    Main extraction loop.

    Returns the wide-format DataFrame.
    """
    ensure_dir(out_dir)

    # Discover files
    candidates = find_dat_candidates(root)
    if candidates.empty:
        raise FileNotFoundError(
            f"No recognisable .dat files found under: {root}\n"
            "Expected filenames: wake_30s_N.dat, trans_30s_N.dat, "
            "early_30s_N.dat, late_30s_N.dat"
        )

    subjects        = sorted(candidates["subject"].unique())
    n_subj          = len(subjects)
    channel_names   = MONTAGE

    print(f"Found {len(candidates)} files for {n_subj} subjects")
    print(f"Stages: {sorted(candidates['stage'].unique())}")

    # Subject-level completeness
    completeness = (
        candidates.groupby("subject")["stage"]
        .apply(lambda x: list(x.unique()))
        .reset_index()
        .rename(columns={"stage": "stages_found"})
    )
    completeness["n_stages"]  = completeness["stages_found"].apply(len)
    completeness["complete"]  = completeness["n_stages"] == 4
    n_complete = completeness["complete"].sum()
    print(f"Complete (all 4 stages): {n_complete}/{n_subj}")

    # Extract parameters
    all_rows = []
    for subj in subjects:
        subj_files = candidates[candidates["subject"] == subj]
        sex = subj_files["sex"].iloc[0]

        for _, row in subj_files.iterrows():
            stage = row["stage"]
            path  = row["path"]

            roi_params = extract_subject_stage(path, channel_names)
            if roi_params is None:
                continue

            for roi_name, params in roi_params.items():
                rec = {
                    "subject": subj,
                    "sex":     sex,
                    "stage":   stage,
                    "roi":     roi_name,
                }
                rec.update(params)
                all_rows.append(rec)

    long_df = pd.DataFrame(all_rows)

    if long_df.empty:
        raise RuntimeError("No parameters extracted. Check data files and paths.")

    # Pivot to wide format: one row per (subject, roi)
    # columns: metric_STAGE
    metric_cols = [c for c in long_df.columns
                   if c not in ("subject", "sex", "stage", "roi")]

    wide = long_df.pivot_table(
        index=["subject", "roi"],
        columns="stage",
        values=metric_cols,
        aggfunc="first",
    )
    wide.columns   = [f"{m}_{s}" for m, s in wide.columns]
    wide           = wide.reset_index()

    # Add sex column
    sex_map = candidates.drop_duplicates("subject").set_index("subject")["sex"]
    wide["sex"] = wide["subject"].map(sex_map)

    # Reorder columns
    id_cols   = ["subject", "sex", "roi"]
    data_cols = [c for c in wide.columns if c not in id_cols]
    wide      = wide[id_cols + sorted(data_cols)]

    # ── Stage means (group level) ──────────────────────────────────────────────
    stage_means = (
        long_df
        .groupby(["roi", "stage"])[metric_cols]
        .agg(["mean", "std", "sem"])
        .reset_index()
    )
    stage_means.columns = [
        "_".join(c).strip("_") for c in stage_means.columns.values
    ]

    # ── Save ──────────────────────────────────────────────────────────────────
    wide_path        = out_dir / "anatomical_roi_parameters_wide.csv"
    stage_means_path = out_dir / "stage_means_by_roi.csv"
    completeness_path= out_dir / "subject_completeness.csv"

    wide.to_csv(wide_path, index=False)
    stage_means.to_csv(stage_means_path, index=False)
    completeness.to_csv(completeness_path, index=False)

    # Validation log
    with open(out_dir / "validation_log.txt", "w") as f:
        f.write(f"Extraction log\n{'='*50}\n")
        f.write(f"Root:     {root}\n")
        f.write(f"Subjects: {n_subj}\n")
        f.write(f"Complete: {n_complete}/{n_subj}\n\n")
        incomplete = completeness[~completeness["complete"]]
        if not incomplete.empty:
            f.write("Incomplete subjects:\n")
            for _, row in incomplete.iterrows():
                f.write(f"  {row['subject']}: {row['stages_found']}\n")

    print(f"\nOutput:")
    print(f"  Wide CSV:     {wide_path}  ({len(wide)} rows)")
    print(f"  Stage means:  {stage_means_path}")
    print(f"  Completeness: {completeness_path}")

    return wide


def main():
    parser = argparse.ArgumentParser(
        description="Extract EEG spectral parameters from .dat files."
    )
    parser.add_argument(
        "--root", type=str, default=str(DEFAULT_ROOT),
        help="Root folder containing subject .dat files"
    )
    parser.add_argument(
        "--out", type=str, default=str(OUTPUT_DIR),
        help="Output directory"
    )
    args = parser.parse_args()
    run(Path(args.root), Path(args.out))


if __name__ == "__main__":
    main()
