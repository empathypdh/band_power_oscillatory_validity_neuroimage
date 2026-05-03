# -*- coding: utf-8 -*-
"""
05_split_half.py
================
Step 5 of the analysis pipeline.

Quantifies the internal reproducibility of all stage-contrast effect sizes
using 1,000 random split-half iterations.

For each iteration:
    - 30 subjects are randomly drawn (half of N=61)
    - Cohen's dz is recomputed for band power and peak height
      across all 7 bands × 9 ROIs × 5 contrasts

Reliability metric:
    Directional agreement (%) = proportion of iterations in which the
    sign of the half-sample dz matches the sign of the full-sample dz.
    Threshold for "reliable": ≥ 90%.

This addresses the question: "Are the observed effect directions stable
across different subject sub-samples, or are they sample-specific?"

Output
------
{OUTPUT_DIR}/
    split_half_detailed.csv       per band/ROI/contrast/metric: full results
    split_half_summary.csv        summary for Supplementary Table S4
    split_half_reliability.csv    % reliable per contrast (≥90% dir.agree)

Usage
-----
    python 05_split_half.py
    python 05_split_half.py --input results/anatomical_roi_parameters_wide.csv
    python 05_split_half.py --n-iter 500   (faster test run)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    OUTPUT_DIR, CONTRASTS, CONTRAST_LABELS, BANDS_ANALYSIS,
    SPLIT_HALF_N_ITER, SPLIT_HALF_SEED, SPLIT_HALF_RELIABILITY_THRESHOLD,
    MIN_N_SUBJECTS,
)
from utils import cohens_dz, ensure_dir


# ── Metric lookup ──────────────────────────────────────────────────────────────

METRICS_OF_INTEREST = [
    ("band_power",   lambda b: f"log10_power_{b}"),
    ("peak_height",  lambda b: f"{b}_peak_height_log10"),
]


def get_values(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").values


# ── Split-half reliability ─────────────────────────────────────────────────────

def directional_agreement(
    half_dz: np.ndarray,
    full_dz: float,
) -> float:
    """% of valid iterations where sign(half_dz) == sign(full_dz)."""
    finite = half_dz[np.isfinite(half_dz)]
    if len(finite) < 10 or not np.isfinite(full_dz):
        return np.nan
    full_sign = np.sign(full_dz)
    return float((np.sign(finite) == full_sign).mean() * 100)


def run(input_csv: Path, out_dir: Path, n_iter: int = SPLIT_HALF_N_ITER) -> pd.DataFrame:
    ensure_dir(out_dir)
    np.random.seed(SPLIT_HALF_SEED)

    wide   = pd.read_csv(input_csv)
    all_subj = wide["subject"].unique()
    n_total  = len(all_subj)
    n_half   = n_total // 2

    print(f"Split-half reliability")
    print(f"  N subjects: {n_total}  |  Half size: {n_half}  |  Iterations: {n_iter}")
    print(f"  Reliability threshold: ≥{SPLIT_HALF_RELIABILITY_THRESHOLD}% directional agreement")

    results = []

    rois = sorted(wide["roi"].unique())

    for roi in rois:
        roi_df = wide[wide["roi"] == roi].copy()

        for band in BANDS_ANALYSIS:
            for metric_type, col_fn in METRICS_OF_INTEREST:
                col_base = col_fn(band)

                for s_before, s_after in CONTRASTS:
                    col_b = f"{col_base}_{s_before}"
                    col_a = f"{col_base}_{s_after}"

                    if col_b not in roi_df.columns or col_a not in roi_df.columns:
                        continue

                    vals_b = get_values(roi_df, col_b)
                    vals_a = get_values(roi_df, col_a)
                    mask   = np.isfinite(vals_b) & np.isfinite(vals_a)

                    if mask.sum() < MIN_N_SUBJECTS:
                        continue

                    # Full-sample dz
                    full_dz, _, n_full, _ = cohens_dz(vals_b, vals_a)

                    # Subjects with valid paired data
                    valid_subj = roi_df["subject"].values[mask]

                    # 1000 iterations
                    half_dz_arr = np.full(n_iter, np.nan)
                    for i in range(n_iter):
                        # Sample n_half from valid subjects
                        half_subj = np.random.choice(
                            valid_subj,
                            size=min(n_half, len(valid_subj)),
                            replace=False,
                        )
                        half_mask = roi_df["subject"].isin(half_subj).values
                        dz_h, _, n_h, _ = cohens_dz(
                            vals_b[half_mask], vals_a[half_mask]
                        )
                        if n_h >= MIN_N_SUBJECTS:
                            half_dz_arr[i] = dz_h

                    finite_hz = half_dz_arr[np.isfinite(half_dz_arr)]
                    da        = directional_agreement(half_dz_arr, full_dz)

                    results.append({
                        "roi":           roi,
                        "band":          band,
                        "metric_type":   metric_type,
                        "contrast":      f"{s_before}->{s_after}",
                        "contrast_label":CONTRAST_LABELS.get(
                                             f"{s_before}->{s_after}", ""),
                        "n_full":        int(n_full),
                        "full_dz":       round(float(full_dz), 4)
                                         if np.isfinite(full_dz) else np.nan,
                        "half_dz_mean":  round(float(np.nanmean(half_dz_arr)), 4),
                        "half_dz_sd":    round(float(np.nanstd(half_dz_arr)), 4),
                        "ci95_lo":       round(float(np.nanpercentile(half_dz_arr, 2.5)), 4),
                        "ci95_hi":       round(float(np.nanpercentile(half_dz_arr, 97.5)), 4),
                        "dir_agree_pct": round(da, 1),
                        "reliable":      da >= SPLIT_HALF_RELIABILITY_THRESHOLD
                                         if np.isfinite(da) else False,
                        "n_valid_iters": int(np.isfinite(half_dz_arr).sum()),
                    })

    detail_df = pd.DataFrame(results)

    # ── Summary for Supplementary Table S4 ───────────────────────────────────
    summary_rows = []
    for (roi, contrast), grp in detail_df.groupby(["roi", "contrast"]):
        pw = grp[grp["metric_type"] == "band_power"]
        pk = grp[grp["metric_type"] == "peak_height"]
        summary_rows.append({
            "roi":                 roi,
            "contrast":            contrast,
            "contrast_label":      CONTRAST_LABELS.get(contrast, ""),
            "n_bands":             len(pw),
            "power_mean_abs_dz":   round(pw["full_dz"].abs().mean(), 3),
            "power_dir_agree_pct": round(pw["dir_agree_pct"].mean(), 1),
            "power_n_reliable":    f"{int(pw['reliable'].sum())}/{len(pw)}",
            "peak_mean_abs_dz":    round(pk["full_dz"].abs().mean(), 3),
            "peak_dir_agree_pct":  round(pk["dir_agree_pct"].mean(), 1),
            "peak_n_reliable":     f"{int(pk['reliable'].sum())}/{len(pk)}",
        })
    summary_df = pd.DataFrame(summary_rows)

    # ── Reliability per contrast ──────────────────────────────────────────────
    rel_df = (
        detail_df.groupby("contrast")
        .agg(
            mean_dir_agree=("dir_agree_pct", "mean"),
            pct_reliable  =("reliable", lambda x: x.mean() * 100),
            n_obs         =("reliable", "count"),
        )
        .round({"mean_dir_agree": 1, "pct_reliable": 1})
        .reset_index()
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    detail_df.to_csv(out_dir / "split_half_detailed.csv", index=False)
    summary_df.to_csv(out_dir / "split_half_summary.csv", index=False)
    rel_df.to_csv(out_dir / "split_half_reliability.csv", index=False)

    # Console summary
    print("\nDirectional agreement by contrast (mean across all bands × ROIs × metrics):")
    for c_str in [f"{a}->{b}" for a, b in CONTRASTS]:
        sub = detail_df[detail_df["contrast"] == c_str]
        if sub.empty:
            continue
        da_mean = sub["dir_agree_pct"].mean()
        n_rel   = sub["reliable"].sum()
        n_tot   = len(sub)
        print(f"  {c_str:16}: mean dir.agree = {da_mean:.1f}%  "
              f"reliable = {n_rel}/{n_tot} ({n_rel/n_tot*100:.0f}%)")

    print(f"\nOutput: {out_dir}")
    return detail_df


def main():
    parser = argparse.ArgumentParser(
        description="Split-half internal reproducibility."
    )
    parser.add_argument(
        "--input", type=str,
        default=str(OUTPUT_DIR / "anatomical_roi_parameters_wide.csv"),
    )
    parser.add_argument("--out",    type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--n-iter", type=int, default=SPLIT_HALF_N_ITER)
    args = parser.parse_args()
    run(Path(args.input), Path(args.out), args.n_iter)


if __name__ == "__main__":
    main()
