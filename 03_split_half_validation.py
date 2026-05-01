# -*- coding: utf-8 -*-
"""
Repeated split-half validation for sleep-onset spectral decomposition paper.

Input:
    anatomical_roi_subject_stage_parameters_wide.csv
Output:
    split_half_validation_summary.csv
    split_half_validation_all_iterations.csv

Purpose:
    Tests whether prespecified mechanistic signatures are internally reproducible
    across 1,000 random 30/31 subject split-half partitions.
"""

import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "anatomical_roi_subject_stage_parameters_wide.csv"
ALT_INPUT_CSV = SCRIPT_DIR / "anatomical_roi_subject_stage_parameters_wide.csv"
N_ITER = 1000
SEED = 20260426

def cohens_dz(delta):
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) < 2:
        return np.nan
    sd = np.nanstd(delta, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.nanmean(delta) / sd)

def effect_dz(data, unit, metric, a, b, subjects=None):
    sub = data[data["unit"].eq(unit)].copy()
    if subjects is not None:
        sub = sub[sub["subject"].isin(subjects)]
    x = sub[f"{metric}_{a}"].to_numpy(float)
    y = sub[f"{metric}_{b}"].to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    return cohens_dz(y[mask] - x[mask]), int(mask.sum())

def corr_delta(data, unit, metric1, a1, b1, metric2, a2, b2, subjects=None):
    sub = data[data["unit"].eq(unit)].copy()
    if subjects is not None:
        sub = sub[sub["subject"].isin(subjects)]
    d1 = (sub[f"{metric1}_{b1}"] - sub[f"{metric1}_{a1}"]).to_numpy(float)
    d2 = (sub[f"{metric2}_{b2}"] - sub[f"{metric2}_{a2}"]).to_numpy(float)
    mask = np.isfinite(d1) & np.isfinite(d2)
    if mask.sum() < 3:
        return np.nan, int(mask.sum())
    return float(stats.pearsonr(d1[mask], d2[mask]).statistic), int(mask.sum())

EFFECTS = [
    dict(label="Alpha-1 suppression WS→TS", kind="dz", unit="ACC", metric="alpha1_peak_height_log10", a="WS", b="TS", direction="negative", strong_abs=0.3),
    dict(label="Alpha-1 suppression TS→ESS1", kind="dz", unit="ACC", metric="alpha1_peak_height_log10", a="TS", b="ESS1", direction="negative", strong_abs=0.3),
    dict(label="Alpha-1 false-negative: height ESS1→LSS1", kind="dz", unit="ACC", metric="alpha1_peak_height_log10", a="ESS1", b="LSS1", direction="negative", strong_abs=0.3),
    dict(label="Alpha-1 false-negative: power ESS1→LSS1 near zero", kind="dz", unit="ACC", metric="power_alpha1_mean", a="ESS1", b="LSS1", direction="near_zero", near_abs=0.3),
    dict(label="Theta emergence TS→ESS1", kind="dz", unit="ACC", metric="theta_peak_height_log10", a="TS", b="ESS1", direction="positive", strong_abs=0.3),
    dict(label="Theta emergence ESS1→LSS1", kind="dz", unit="ACC", metric="theta_peak_height_log10", a="ESS1", b="LSS1", direction="positive", strong_abs=0.3),
    dict(label="Delta emergence TS→ESS1", kind="dz", unit="ACC", metric="delta_peak_height_log10", a="TS", b="ESS1", direction="positive", strong_abs=0.3),
    dict(label="Delta emergence ESS1→LSS1", kind="dz", unit="ACC", metric="delta_peak_height_log10", a="ESS1", b="LSS1", direction="positive", strong_abs=0.3),
    dict(label="Beta-3 aperiodic dissociation: power WS→TS", kind="dz", unit="ACC", metric="power_beta3_mean", a="WS", b="TS", direction="negative", strong_abs=0.3),
    dict(label="Beta-3 aperiodic dissociation: height WS→TS near zero/nondecrease", kind="dz", unit="ACC", metric="beta3_peak_height_log10", a="WS", b="TS", direction="not_negative", lower=-0.3),
    dict(label="Beta-3 aperiodic dissociation: power TS→ESS1", kind="dz", unit="ACC", metric="power_beta3_mean", a="TS", b="ESS1", direction="negative", strong_abs=0.3),
    dict(label="Beta-3 aperiodic dissociation: height TS→ESS1 near zero", kind="dz", unit="ACC", metric="beta3_peak_height_log10", a="TS", b="ESS1", direction="near_zero", near_abs=0.3),
    dict(label="Aperiodic exponent increase WS→TS", kind="dz", unit="ACC", metric="aperiodic_exponent", a="WS", b="TS", direction="positive", strong_abs=0.2),
    dict(label="Aperiodic exponent increase TS→ESS1", kind="dz", unit="ACC", metric="aperiodic_exponent", a="TS", b="ESS1", direction="positive", strong_abs=0.3),
    dict(label="Aperiodic exponent increase ESS1→LSS1", kind="dz", unit="ACC", metric="aperiodic_exponent", a="ESS1", b="LSS1", direction="positive", strong_abs=0.3),
    dict(label="Beta-1 late rebound: height ESS1→LSS1", kind="dz", unit="ACC", metric="beta1_peak_height_log10", a="ESS1", b="LSS1", direction="positive", strong_abs=0.3),
    dict(label="Beta-1 late rebound: power ESS1→LSS1", kind="dz", unit="ACC", metric="power_beta1_mean", a="ESS1", b="LSS1", direction="positive", strong_abs=0.3),
    dict(label="Exponent–beta3 power coupling WS→TS", kind="corr", unit="ACC", metric1="aperiodic_exponent", a1="WS", b1="TS", metric2="power_beta3_mean", a2="WS", b2="TS", direction="negative", strong_abs=0.2),
    dict(label="Alpha1–delta antagonism WS→TS", kind="corr", unit="ACC", metric1="alpha1_peak_height_log10", a1="WS", b1="TS", metric2="delta_peak_height_log10", a2="WS", b2="TS", direction="negative", strong_abs=0.2),
    dict(label="Theta–delta antagonism WS→TS", kind="corr", unit="ACC", metric1="theta_peak_height_log10", a1="WS", b1="TS", metric2="delta_peak_height_log10", a2="WS", b2="TS", direction="negative", strong_abs=0.2),
]

def direction_pass(value, rule):
    if np.isnan(value):
        return False
    direction = rule["direction"]
    if direction == "positive":
        return value > 0
    if direction == "negative":
        return value < 0
    if direction == "near_zero":
        return abs(value) < rule.get("near_abs", 0.3)
    if direction == "not_negative":
        return value > rule.get("lower", -0.3)
    raise ValueError(direction)

def criterion_pass(value, rule):
    if np.isnan(value):
        return False
    direction = rule["direction"]
    threshold = rule.get("strong_abs", 0.3)
    if direction == "positive":
        return value >= threshold
    if direction == "negative":
        return value <= -threshold
    if direction == "near_zero":
        return abs(value) < rule.get("near_abs", 0.3)
    if direction == "not_negative":
        return value > rule.get("lower", -0.3)
    raise ValueError(direction)

def main():
    input_csv = INPUT_CSV if INPUT_CSV.exists() else ALT_INPUT_CSV
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found. Expected one of:\n"
            f"  - {INPUT_CSV}\n"
            f"  - {ALT_INPUT_CSV}"
        )

    print("=" * 70, flush=True)
    print("Repeated split-half validation", flush=True)
    print("=" * 70, flush=True)
    print(f"Input CSV: {input_csv}", flush=True)
    print(f"Iterations: {N_ITER}", flush=True)
    print(f"Seed: {SEED}", flush=True)
    print("\nLoading input CSV...", flush=True)

    data = pd.read_csv(input_csv)
    print(f"Loaded rows: {len(data)}", flush=True)
    print(f"Loaded columns: {len(data.columns)}", flush=True)

    required_basic = {"subject", "unit"}
    missing_basic = sorted(required_basic - set(data.columns))
    if missing_basic:
        raise KeyError(f"Missing required columns in input CSV: {missing_basic}")

    subjects = np.array(sorted(data["subject"].unique()))
    print(f"Subjects detected: {len(subjects)}", flush=True)
    rng = np.random.default_rng(SEED)

    print("Computing full-sample effects...", flush=True)
    full_rows = []
    for rule in EFFECTS:
        if rule["kind"] == "dz":
            value, n = effect_dz(data, rule["unit"], rule["metric"], rule["a"], rule["b"])
        else:
            value, n = corr_delta(data, rule["unit"], rule["metric1"], rule["a1"], rule["b1"], rule["metric2"], rule["a2"], rule["b2"])
        full_rows.append({**rule, "full_value": value, "full_n": n})

    print("Running split-half iterations...", flush=True)
    split_rows = []
    for iteration in range(1, N_ITER + 1):
        if iteration == 1 or iteration % 50 == 0 or iteration == N_ITER:
            print(f"  Iteration {iteration}/{N_ITER}", flush=True)
        perm = rng.permutation(subjects)
        halves = {
            "A": set(perm[:len(subjects)//2]),
            "B": set(perm[len(subjects)//2:]),
        }
        for half, half_subjects in halves.items():
            for rule in EFFECTS:
                if rule["kind"] == "dz":
                    value, n = effect_dz(data, rule["unit"], rule["metric"], rule["a"], rule["b"], half_subjects)
                else:
                    value, n = corr_delta(data, rule["unit"], rule["metric1"], rule["a1"], rule["b1"], rule["metric2"], rule["a2"], rule["b2"], half_subjects)
                split_rows.append({
                    "iteration": iteration,
                    "half": half,
                    "label": rule["label"],
                    "kind": rule["kind"],
                    "value": value,
                    "n": n,
                    "direction_pass": direction_pass(value, rule),
                    "criterion_pass": criterion_pass(value, rule),
                })

    print("Summarising split-half results...", flush=True)
    split_df = pd.DataFrame(split_rows)
    summary_rows = []
    for rule in EFFECTS:
        sub = split_df[split_df["label"].eq(rule["label"])]
        direction_pivot = sub.pivot_table(index="iteration", columns="half", values="direction_pass", aggfunc="first")
        criterion_pivot = sub.pivot_table(index="iteration", columns="half", values="criterion_pass", aggfunc="first")
        values = sub["value"].to_numpy()
        full = next(row for row in full_rows if row["label"] == rule["label"])
        summary_rows.append({
            "finding": rule["label"],
            "statistic": "Cohen_dz" if rule["kind"] == "dz" else "Pearson_r",
            "full_sample_value": full["full_value"],
            "median_split_value": np.nanmedian(values),
            "IQR_split_value": f"{np.nanpercentile(values,25):.3f} to {np.nanpercentile(values,75):.3f}",
            "direction_reproducibility_all_halves_%": 100 * sub["direction_pass"].mean(),
            "direction_reproducibility_both_halves_%": 100 * direction_pivot.all(axis=1).mean(),
            "criterion_reproducibility_all_halves_%": 100 * sub["criterion_pass"].mean(),
            "criterion_reproducibility_both_halves_%": 100 * criterion_pivot.all(axis=1).mean(),
        })

    summary_path = SCRIPT_DIR / "split_half_validation_summary.csv"
    iterations_path = SCRIPT_DIR / "split_half_validation_all_iterations.csv"
    print("Writing output CSV files...", flush=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    split_df.to_csv(iterations_path, index=False)
    print("\nDone:", flush=True)
    print(f" - {summary_path}")
    print(f" - {iterations_path}")

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
