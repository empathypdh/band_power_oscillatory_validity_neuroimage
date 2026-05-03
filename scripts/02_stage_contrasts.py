# -*- coding: utf-8 -*-
"""
02_stage_contrasts.py
=====================
Step 2 of the analysis pipeline.

Reads the wide CSV produced by 01_extract_parameters.py and computes
paired statistics for all stage contrasts across all bands and ROIs.

Two classes of contrasts are computed:

    Sequential contrasts (fine-grained temporal progression):
        WS → TS       Transition boundary (operationally defined)
        TS → ESS1     Boundary to early Stage 1
        ESS1 → LSS1   Stage 1 consolidation

    Conventional stage-based contrasts (clinically interpretable):
        WS → ESS1     Stable wakefulness vs established Stage 1
                      Primary external comparison (corresponds to W→N1)
        WS → LSS1     Full sleep-onset span (corresponds to W→N2)

For each band × ROI × contrast combination, the following are computed:
    - Cohen's dz (paired effect size)
    - Two-tailed paired t-test (p-value)
    - Mean change (after − before)
    - 95% CI of mean change (t-distribution)

Output
------
{OUTPUT_DIR}/
    contrast_statistics.csv        full 315-row results table
    contrast_statistics_wide.csv   pivoted for downstream classification

Usage
-----
    python 02_stage_contrasts.py
    python 02_stage_contrasts.py --input results/anatomical_roi_parameters_wide.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    OUTPUT_DIR, CONTRASTS, CONTRAST_LABELS,
    BANDS_ANALYSIS, MIN_N_SUBJECTS,
)
from utils import cohens_dz, pearson_r, ensure_dir, fmt, p_stars


# ── Statistical helpers ────────────────────────────────────────────────────────

def confidence_interval_95(delta: np.ndarray) -> tuple[float, float]:
    """95% CI of paired differences using the t-distribution."""
    delta  = delta[np.isfinite(delta)]
    n      = len(delta)
    if n < 2:
        return np.nan, np.nan
    se     = np.std(delta, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    m      = float(np.mean(delta))
    return m - t_crit * se, m + t_crit * se


# ── Column lookup ──────────────────────────────────────────────────────────────

def get_values(wide: pd.DataFrame, metric_base: str, stage: str) -> np.ndarray | None:
    """
    Return per-subject values for metric_base at given stage.

    metric_base examples: 'log10_power_alpha1', 'alpha1_peak_height_log10'
    """
    col = f"{metric_base}_{stage}"
    if col not in wide.columns:
        return None
    return pd.to_numeric(wide[col], errors="coerce").values


def iter_metrics(wide: pd.DataFrame) -> list[str]:
    """
    Return all unique metric base names present in the wide DataFrame.
    These are metric names without the trailing _STAGE suffix.
    """
    stage_suffixes = ["_WS", "_TS", "_ESS1", "_LSS1"]
    bases = set()
    for col in wide.columns:
        for suf in stage_suffixes:
            if col.endswith(suf):
                bases.add(col[: -len(suf)])
    return sorted(bases)


# ── Main computation ───────────────────────────────────────────────────────────

def compute_contrasts(wide: pd.DataFrame, min_n: int = MIN_N_SUBJECTS) -> pd.DataFrame:
    """
    Compute all paired statistics for all metric x ROI x contrast combinations.

    Returns a long-format DataFrame with one row per (roi, metric, contrast).
    min_n: minimum number of paired subjects required (use --min-n 1 for testing).
    """
    rois    = sorted(wide["roi"].unique())
    metrics = iter_metrics(wide)
    rows    = []
    n_skipped = 0

    n_subjects = wide["subject"].nunique()
    if n_subjects < min_n:
        print(f"\n  WARNING: Only {n_subjects} subject(s) found (minimum required: {min_n}).")
        print(f"  Check that --root points to the folder containing ALL subjects.")
        print(f"  For testing with few subjects, use:  python 02_stage_contrasts.py --min-n 1\n")

    for roi in rois:
        roi_df = wide[wide["roi"] == roi].copy()

        for metric in metrics:
            for s_before, s_after in CONTRASTS:
                vals_b = get_values(roi_df, metric, s_before)
                vals_a = get_values(roi_df, metric, s_after)

                if vals_b is None or vals_a is None:
                    continue

                dz, p, n, mean_delta = cohens_dz(vals_b, vals_a)

                mask   = np.isfinite(vals_b) & np.isfinite(vals_a)
                if mask.sum() < min_n:
                    n_skipped += 1
                    continue

                ci_lo, ci_hi = confidence_interval_95(
                    vals_a[mask] - vals_b[mask]
                )

                contrast_str = f"{s_before}->{s_after}"
                rows.append({
                    "roi":          roi,
                    "metric":       metric,
                    "contrast":     contrast_str,
                    "contrast_label": CONTRAST_LABELS.get(contrast_str, contrast_str),
                    "n":            n,
                    "dz":           dz,
                    "p":            p,
                    "stars":        p_stars(p),
                    "mean_delta":   mean_delta,
                    "ci95_lo":      ci_lo,
                    "ci95_hi":      ci_hi,
                })

    return pd.DataFrame(rows)


def run(input_csv: Path, out_dir: Path, min_n: int = MIN_N_SUBJECTS) -> pd.DataFrame:
    ensure_dir(out_dir)

    wide = pd.read_csv(input_csv)
    print(f"Input: {input_csv}")
    print(f"  Rows: {len(wide)}, ROIs: {wide['roi'].nunique()}, "
          f"Subjects: {wide['subject'].nunique()}")

    stats_df = compute_contrasts(wide, min_n=min_n)

    if stats_df.empty:
        n_subj = wide["subject"].nunique()
        raise RuntimeError(
            f"No statistics computed.\n"
            f"  Subjects found: {n_subj} (minimum required: {min_n})\n"
            f"  Solutions:\n"
            f"  1. Check --root path contains all {n_subj} subjects\' .dat files\n"
            f"  2. For testing: python 02_stage_contrasts.py --min-n 1"
        )

    # ── Band-specific statistics for paper (primary analysis) ──────────────────
    # Focus: log10 band power and oscillatory peak height across 7 bands
    band_stats = stats_df[
        stats_df["metric"].apply(
            lambda m: any(
                (f"log10_power_{b}" == m or f"{b}_peak_height_log10" == m)
                for b in BANDS_ANALYSIS
            )
        )
    ].copy()

    # Add band and metric_type columns for easy filtering
    def parse_metric(m: str) -> tuple[str, str]:
        for b in BANDS_ANALYSIS:
            if m == f"log10_power_{b}":
                return b, "band_power"
            if m == f"{b}_peak_height_log10":
                return b, "peak_height"
        return "other", "other"

    band_stats[["band", "metric_type"]] = pd.DataFrame(
        band_stats["metric"].apply(parse_metric).tolist(),
        index=band_stats.index,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    full_path = out_dir / "contrast_statistics_all_metrics.csv"
    band_path = out_dir / "contrast_statistics_band_metrics.csv"

    stats_df.to_csv(full_path, index=False)
    band_stats.to_csv(band_path, index=False)

    # Console summary
    n_roi       = stats_df["roi"].nunique()
    n_contrasts = stats_df["contrast"].nunique()
    n_metrics   = stats_df["metric"].nunique()
    print(f"\nStatistics computed:")
    print(f"  {len(stats_df):,} total observations")
    print(f"  {n_roi} ROIs × {n_metrics} metrics × {n_contrasts} contrasts")
    print(f"  Band-metric observations: {len(band_stats)}")

    print("\nEffect sizes by contrast (mean |dz|, band metrics):")
    for c in [f"{a}->{b}" for a, b in CONTRASTS]:
        sub = band_stats[band_stats["contrast"] == c]
        if not sub.empty:
            print(f"  {c:16}: mean|dz| = {sub['dz'].abs().mean():.3f}  "
                  f"(range {sub['dz'].abs().min():.2f}–{sub['dz'].abs().max():.2f})")

    print(f"\nOutput:")
    print(f"  Full stats: {full_path}")
    print(f"  Band stats: {band_path}")

    return stats_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute paired stage-contrast statistics."
    )
    parser.add_argument(
        "--input", type=str,
        default=str(OUTPUT_DIR / "anatomical_roi_parameters_wide.csv"),
        help="Path to wide CSV from 01_extract_parameters.py",
    )
    parser.add_argument(
        "--out", type=str, default=str(OUTPUT_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--min-n", type=int, default=MIN_N_SUBJECTS,
        help=f"Minimum subjects for paired stats (default: {MIN_N_SUBJECTS}). "
             "Use --min-n 1 for single-subject testing.",
    )
    args = parser.parse_args()
    run(Path(args.input), Path(args.out), args.min_n)


if __name__ == "__main__":
    main()
