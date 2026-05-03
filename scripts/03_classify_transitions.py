# -*- coding: utf-8 -*-
"""
03_classify_transitions.py
==========================
Step 3 of the analysis pipeline.

Reads the wide CSV (from 01) and the band-metric statistics (from 02) to:

    1. Classify each band × ROI × contrast observation into one of six
       mutually exclusive inference categories based on the joint pattern
       of band-limited power (BLP) and oscillatory peak height (OPH) changes.

    2. Compute the Decoupling Index (DI) for each observation, quantifying
       the degree to which BLP change is driven by aperiodic rather than
       oscillatory dynamics.

The three principal error types identified:
    FALSE_POSITIVE:  BLP changes; OPH unchanged; driven by aperiodic exponent.
                     Examples: beta-3 (WS→ESS1 frontal), beta-1 (WS→LSS1),
                               delta (WS→TS)
    FALSE_NEGATIVE:  OPH changes; BLP unchanged; masked by aperiodic offset.
                     Examples: delta (WS→ESS1 — slow oscillation emergence
                               invisible to BLP), theta (WS→TS),
                               alpha-1 (ESS1→LSS1 — continued suppression
                               masked by aperiodic slope recovery)
    OPPOSITE_DIR:    Both metrics change but in opposing directions.
                     Examples: posterior beta-3 (WS→ESS1 — genuine posterior
                               beta activation coexisting with aperiodic-driven
                               BLP suppression), posterior alpha-1 (ESS1→LSS1)

Output
------
{OUTPUT_DIR}/
    classification_table.csv          full results (315 rows)
    classification_summary.csv        counts per classification per contrast
    beta3_focus.csv                   beta-3-only table (for paper focal analysis)
    decoupling_index_by_roi.csv       DI table for all ROIs and contrasts

Usage
-----
    python 03_classify_transitions.py
    python 03_classify_transitions.py --wide results/anatomical_roi_parameters_wide.csv
                                      --stats results/contrast_statistics_band_metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    OUTPUT_DIR, CONTRASTS, CONTRAST_LABELS,
    BANDS_ANALYSIS, CLS_EFFECT_THRESHOLD, CLS_ZERO_THRESHOLD,
    CLS_VALID, CLS_FP, CLS_PWR_ONLY, CLS_FN, CLS_OPP, CLS_NOREP, CLS_ORDER,
)
from utils import (
    classify_transition, decoupling_index as compute_di,
    pearson_r, ensure_dir, fmt,
)


# ── Per-subject delta computation ──────────────────────────────────────────────

def subject_deltas(
    wide: pd.DataFrame,
    roi: str,
    metric_base: str,
    s_before: str,
    s_after: str,
) -> np.ndarray | None:
    """Return per-subject change scores (after − before) for a metric."""
    col_b = f"{metric_base}_{s_before}"
    col_a = f"{metric_base}_{s_after}"
    sub   = wide[wide["roi"] == roi]
    if col_b not in sub.columns or col_a not in sub.columns:
        return None
    b = pd.to_numeric(sub[col_b], errors="coerce").values
    a = pd.to_numeric(sub[col_a], errors="coerce").values
    mask = np.isfinite(b) & np.isfinite(a)
    if mask.sum() < 3:
        return None
    delta = np.full(len(b), np.nan)
    delta[mask] = a[mask] - b[mask]
    return delta


# ── Main classification loop ───────────────────────────────────────────────────

def run(wide_csv: Path, stats_csv: Path, out_dir: Path) -> pd.DataFrame:
    ensure_dir(out_dir)

    wide     = pd.read_csv(wide_csv)
    stats_df = pd.read_csv(stats_csv)   # band metrics (power + peak height)

    # Also load full stats which contains aperiodic_exponent dz
    # (needed for FALSE_POSITIVE classification)
    all_stats_path = stats_csv.parent / "contrast_statistics_all_metrics.csv"
    if all_stats_path.exists():
        all_stats_df = pd.read_csv(all_stats_path)
    else:
        all_stats_df = stats_df  # fallback

    # Filter to band metrics only
    band_stats = stats_df[stats_df["metric_type"].isin(
        ["band_power", "peak_height"]
    )].copy() if "metric_type" in stats_df.columns else stats_df.copy()

    results = []

    for band in BANDS_ANALYSIS:
        pw_metric = f"log10_power_{band}"
        pk_metric = f"{band}_peak_height_log10"
        ex_metric = "aperiodic_exponent"

        for roi in sorted(wide["roi"].unique()):
            for s_before, s_after in CONTRASTS:
                contrast_str = f"{s_before}->{s_after}"

                # Retrieve dz from stats table
                def get_dz(metric: str) -> float:
                    row = band_stats[
                        (band_stats["roi"]      == roi) &
                        (band_stats["metric"]   == metric) &
                        (band_stats["contrast"] == contrast_str)
                    ]
                    return float(row["dz"].iloc[0]) if not row.empty else np.nan

                def get_p(metric: str) -> float:
                    row = band_stats[
                        (band_stats["roi"]      == roi) &
                        (band_stats["metric"]   == metric) &
                        (band_stats["contrast"] == contrast_str)
                    ]
                    return float(row["p"].iloc[0]) if not row.empty else np.nan

                pw_dz  = get_dz(pw_metric)
                pk_dz  = get_dz(pk_metric)

                # Get exponent dz from all_metrics CSV
                # (aperiodic_exponent is not in band_metrics CSV)
                ex_dz_row = all_stats_df[
                    (all_stats_df["roi"]      == roi) &
                    (all_stats_df["metric"]   == ex_metric) &
                    (all_stats_df["contrast"] == contrast_str)
                ]
                ex_dz = float(ex_dz_row["dz"].iloc[0]) if not ex_dz_row.empty else np.nan

                # Classify
                cls = classify_transition(pw_dz, pk_dz, ex_dz)

                # Decoupling Index
                d_exp = subject_deltas(wide, roi, ex_metric, s_before, s_after)
                d_blp = subject_deltas(wide, roi, pw_metric, s_before, s_after)
                d_oph = subject_deltas(wide, roi, pk_metric, s_before, s_after)

                if d_exp is not None and d_blp is not None and d_oph is not None:
                    di, r_blp, r_oph = compute_di(d_exp, d_blp, d_oph)
                else:
                    di = r_blp = r_oph = np.nan

                results.append({
                    "roi":              roi,
                    "band":             band,
                    "contrast":         contrast_str,
                    "contrast_label":   CONTRAST_LABELS.get(contrast_str, contrast_str),
                    "classification":   cls,
                    "power_dz":         pw_dz,
                    "power_p":          get_p(pw_metric),
                    "peak_dz":          pk_dz,
                    "peak_p":           get_p(pk_metric),
                    "exp_dz":           ex_dz,
                    "r_exp_blp":        r_blp,
                    "r_exp_oph":        r_oph,
                    "decoupling_index": di,
                })

    cls_df = pd.DataFrame(results)

    # ── Summary counts per classification per contrast ─────────────────────────
    summary = (
        cls_df.groupby(["contrast", "classification"])
        .size()
        .reset_index(name="count")
    )
    summary["total"] = summary.groupby("contrast")["count"].transform("sum")
    summary["pct"]   = (summary["count"] / summary["total"] * 100).round(1)

    # ── Beta-3 focus table ────────────────────────────────────────────────────
    beta3 = cls_df[cls_df["band"] == "beta3"].copy()

    # ── DI summary: mean DI by ROI × contrast ────────────────────────────────
    di_summary = (
        cls_df.groupby(["roi", "contrast"])["decoupling_index"]
        .mean()
        .reset_index()
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    cls_df.to_csv(out_dir / "classification_table.csv", index=False)
    summary.to_csv(out_dir / "classification_summary.csv", index=False)
    beta3.to_csv(out_dir / "beta3_focus.csv", index=False)
    di_summary.to_csv(out_dir / "decoupling_index_by_roi.csv", index=False)

    # ── Console report ────────────────────────────────────────────────────────
    print(f"\nClassification complete: {len(cls_df)} observations")
    print(f"\nOverall classification distribution:")
    overall = cls_df["classification"].value_counts()
    for cls_label in CLS_ORDER:
        count = overall.get(cls_label, 0)
        pct   = count / len(cls_df) * 100
        print(f"  {cls_label:30s}: {count:4d}  ({pct:5.1f}%)")

    print("\nKey findings by band (count of non-VALID / non-NO_CLEAR observations):")
    for band in BANDS_ANALYSIS:
        sub = cls_df[cls_df["band"] == band]
        fp  = (sub["classification"] == CLS_FP ).sum()
        fn  = (sub["classification"] == CLS_FN ).sum()
        opp = (sub["classification"] == CLS_OPP).sum()
        val = (sub["classification"] == CLS_VALID).sum()
        print(f"  {band:8s}: VALID={val:2d} | FALSE+={fp:2d} | FALSE-={fn:2d} | OPP-DIR={opp:2d}")

    print(f"\nBeta-3 temporal trajectory:")
    for c_str in [f"{a}->{b}" for a, b in CONTRASTS]:
        sub = beta3[beta3["contrast"] == c_str]
        if sub.empty:
            continue
        counts = sub["classification"].value_counts()
        print(f"  {c_str:16}: "
              + "  ".join(f"{k}={v}" for k, v in counts.items()))

    print(f"\nOutput: {out_dir}")

    return cls_df


def main():
    parser = argparse.ArgumentParser(
        description="Classify spectral transitions and compute Decoupling Index."
    )
    parser.add_argument(
        "--wide", type=str,
        default=str(OUTPUT_DIR / "anatomical_roi_parameters_wide.csv"),
    )
    parser.add_argument(
        "--stats", type=str,
        default=str(OUTPUT_DIR / "contrast_statistics_band_metrics.csv"),
    )
    parser.add_argument(
        "--out", type=str, default=str(OUTPUT_DIR),
    )
    args = parser.parse_args()
    run(Path(args.wide), Path(args.stats), Path(args.out))


if __name__ == "__main__":
    main()
