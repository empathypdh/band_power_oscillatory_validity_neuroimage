# -*- coding: utf-8 -*-
"""
06_sleep_edf_validation.py
==========================
Step 6 of the analysis pipeline.

External validation of the primary spectral decomposition findings using
the publicly available Sleep-EDF Expanded dataset (PhysioNet).

Dataset:
    Sleep-EDF Expanded (Cassette subset)
    Kemp et al. (2000); Goldberger et al. (2000)
    https://physionet.org/content/sleep-edfx/

    N = 61 subjects (first-night recordings, SC4xxx series)
    EEG channels: Fpz-Cz (fronto-central midline proxy)
                  Pz-Oz  (occipital proxy)
    Sampling frequency: 100 Hz
    Staging: Rechtschaffen & Kales (1968)

Contrasts:
    W → N1   Corresponds to WS → ESS1 (conventional stage-based)
    W → N2   Corresponds to WS → LSS1 (full span)

Primary validation criteria:
    1. Aperiodic exponent increases from W to N1 and N2
       (replicates main dataset finding)

    2. r(Δexponent, Δbeta-3 BLP) is strongly negative
       (replicates the FALSE_POSITIVE mechanism:
        aperiodic exponent preferentially predicts BLP over OPH)

    3. r(Δexponent, Δalpha-1 BLP) and r(Δexponent, Δalpha-1 OPH)
       are both negative and similar in magnitude
       (replicates the VALID pattern for alpha: both metrics track together)

Note on expected attenuation:
    The W→N1 contrast aggregates across the full R&K Stage 1,
    spanning what corresponds to TS through ESS1 in the primary dataset.
    The false-positive signature is maximal at the WS→TS boundary
    and becomes spatially heterogeneous by WS→ESS1.
    Accordingly, the W→N1 aperiodic exponent change may be attenuated
    relative to the primary dataset WS→ESS1 contrast.
    This attenuation is a principled prediction, not a replication failure.

Output
------
{OUTPUT_DIR}/
    sleep_edf_parameters.csv         per-subject stage-level parameters
    sleep_edf_contrast_stats.csv     paired statistics for W→N1, W→N2
    sleep_edf_validation_report.txt  key validation metrics

Usage
-----
    python 06_sleep_edf_validation.py
    python 06_sleep_edf_validation.py --n-subjects 61
    python 06_sleep_edf_validation.py --path "C:/sleep_edf"
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    OUTPUT_DIR, BANDS, BANDS_ANALYSIS, APERIODIC_FIT_RANGE,
    SLEEP_EDF_N_SUBJECTS, SLEEP_EDF_RECORDING, SLEEP_EDF_PATH,
    SLEEP_EDF_STAGE_MAP, SLEEP_EDF_CHANNELS, SLEEP_EDF_CONTRASTS,
)
from utils import (
    fit_aperiodic, band_power, band_peak_metrics,
    cohens_dz, pearson_r, ensure_dir, fmt, p_stars,
)

# MNE for Sleep-EDF downloading and loading
try:
    import mne
    from mne.datasets.sleep_physionet.age import fetch_data
    _MNE_OK = True
except ImportError:
    _MNE_OK = False
    warnings.warn("MNE not installed. Run: pip install mne")


# ── Signal processing for Sleep-EDF ───────────────────────────────────────────

SLEEP_EDF_SFREQ    = 100
SLEEP_EDF_NPERSEG  = int(4 * SLEEP_EDF_SFREQ)    # 4-sec windows
SLEEP_EDF_NOVERLAP = SLEEP_EDF_NPERSEG // 2

EPOCH_DURATION_SEC = 30


def load_sleep_edf_subject(
    psg_file: str,
    hyp_file: str,
    stage_map: dict,
) -> dict | None:
    """
    Load one Sleep-EDF recording and epoch by stage.

    Returns dict: stage_label -> {channel_label: concatenated signal array}
    or None on failure.

    Note: events_from_annotations requires integer event codes.
    We use a fixed integer mapping and convert back to stage labels.
    """
    if not _MNE_OK:
        return None
    try:
        raw   = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
        annot = mne.read_annotations(hyp_file)
        raw.set_annotations(annot, verbose=False)

        # Map annotation descriptions → integer codes (MNE requirement)
        # stage_map: {"Sleep stage W": "W", "Sleep stage 1": "N1", ...}
        desc_to_int  = {desc: i+1 for i, desc in enumerate(stage_map)}
        int_to_stage = {i+1: stage_map[desc] for i, desc in enumerate(stage_map)}

        events, found_id = mne.events_from_annotations(
            raw, event_id=desc_to_int, verbose=False
        )
        if events.size == 0:
            return None

        # Resample
        if raw.info["sfreq"] != SLEEP_EDF_SFREQ:
            raw.resample(SLEEP_EDF_SFREQ, verbose=False)

        # Bandpass filter
        raw.filter(0.5, 35.0, verbose=False)

        epoch_len = int(EPOCH_DURATION_SEC * SLEEP_EDF_SFREQ)

        # Collect epochs per stage per channel
        stage_signals: dict = {}

        for int_code, stage_label in int_to_stage.items():
            stage_events = events[events[:, 2] == int_code]
            if len(stage_events) == 0:
                continue

            ch_segs: dict = {ch: [] for ch in SLEEP_EDF_CHANNELS}
            for evt in stage_events:
                start = int(evt[0])
                end   = start + epoch_len
                if end > raw.n_times:
                    continue
                for ch_label, ch_name in SLEEP_EDF_CHANNELS.items():
                    if ch_name not in raw.ch_names:
                        continue
                    idx = raw.ch_names.index(ch_name)
                    seg = raw.get_data(picks=[idx], start=start, stop=end)[0]
                    ch_segs[ch_label].append(seg)

            stage_signals[stage_label] = {}
            for ch_label, segs in ch_segs.items():
                if segs:
                    stage_signals[stage_label][ch_label] = np.concatenate(segs)

        return stage_signals if stage_signals else None

    except Exception as e:
        warnings.warn(f"Failed to load {Path(psg_file).name}: {e}")
        return None


def extract_psd_metrics(
    signal: np.ndarray,
    sfreq: float = SLEEP_EDF_SFREQ,
) -> dict:
    """
    Compute PSD metrics from a concatenated stage signal.

    Returns dict of spectral parameters.
    """
    nperseg  = min(SLEEP_EDF_NPERSEG, len(signal) // 2)
    noverlap = nperseg // 2

    try:
        freqs, psd = welch(
            signal, fs=sfreq,
            nperseg=nperseg, noverlap=noverlap,
            scaling="density",
        )
    except Exception:
        return {}

    ap  = fit_aperiodic(freqs, psd, fit_range=APERIODIC_FIT_RANGE)
    rec = {
        "aperiodic_exponent": ap["aperiodic_exponent"],
        "aperiodic_offset":   ap["aperiodic_offset"],
    }

    for b in BANDS_ANALYSIS:
        lo, hi = BANDS[b]
        bp     = band_power(freqs, psd, (lo, hi))
        rec[f"log10_power_{b}"] = np.log10(bp) if bp > 0 else np.nan
        rec.update(band_peak_metrics(
            freqs, ap["residual_log10"], ap["log10_psd"], b
        ))

    return rec


# ── Main run ───────────────────────────────────────────────────────────────────

def run(
    data_path: Path,
    out_dir: Path,
    n_subjects: int = SLEEP_EDF_N_SUBJECTS,
) -> pd.DataFrame:
    ensure_dir(out_dir)

    if not _MNE_OK:
        raise RuntimeError("MNE is required. Install with: pip install mne")

    print(f"Sleep-EDF external validation")
    print(f"  N subjects: {n_subjects}")
    print(f"  Data path: {data_path}")
    print(f"  Contrasts: {SLEEP_EDF_CONTRASTS}")
    print(f"  Channels: {list(SLEEP_EDF_CHANNELS.values())}")

    # Scan data directory directly (files already downloaded)
    # Check both data_path and data_path/physionet-sleep-data/
    import re as _re
    scan_dirs = [
        data_path,
        data_path / "physionet-sleep-data",
    ]
    
    psg_files = []
    hyp_files = []
    for scan_dir in scan_dirs:
        if scan_dir.exists():
            psg_files.extend(sorted(scan_dir.glob("*PSG*.edf")))
            hyp_files.extend(sorted(scan_dir.glob("*Hypnogram*.edf")))
    
    # Match PSG + Hypnogram by subject code (SC4XXX)
    def get_subj_code(p):
        m = _re.search(r"(SC4\d+[A-Z]?)", str(p.name), _re.IGNORECASE)
        return m.group(1)[:6] if m else ""  # SC4XXX (6 chars)

    psg_dict = {}
    for f in psg_files:
        code = get_subj_code(f)
        if code:
            psg_dict[code] = str(f)

    hyp_dict = {}
    for f in hyp_files:
        code = get_subj_code(f)
        if code:
            hyp_dict[code] = str(f)

    common = sorted(set(psg_dict) & set(hyp_dict))
    all_pairs = [(psg_dict[k], hyp_dict[k]) for k in common]
    
    # Select exactly n_subjects pairs
    pairs = all_pairs[:n_subjects]
    print(f"  Available pairs: {len(all_pairs)}, using: {len(pairs)}")

    if len(pairs) == 0:
        print("  ERROR: No PSG-Hypnogram pairs found.")
        print(f"  Scanned: {[str(d) for d in scan_dirs]}")
        return pd.DataFrame()

    all_rows = []

    for subj_idx, (psg, hyp) in enumerate(pairs):
        print(f"  Subject {subj_idx+1:02d}/{len(pairs)}... ", end="", flush=True)

        stage_signals = load_sleep_edf_subject(psg, hyp, SLEEP_EDF_STAGE_MAP)
        if stage_signals is None:
            print("SKIP")
            continue

        for stage, ch_signals in stage_signals.items():
            for ch_label, signal in ch_signals.items():
                if len(signal) < SLEEP_EDF_NPERSEG * 2:
                    continue
                metrics = extract_psd_metrics(signal)
                if not metrics:
                    continue
                rec = {
                    "subject":  f"SEDF_{subj_idx:03d}",
                    "stage":    stage,
                    "channel":  ch_label,
                }
                rec.update(metrics)
                all_rows.append(rec)

        print("OK")

    params_df = pd.DataFrame(all_rows)
    params_df.to_csv(out_dir / "sleep_edf_parameters.csv", index=False)

    print(f"\nExtracted {len(params_df)} observations")
    if params_df.empty:
        print("  ERROR: No data extracted. Check file paths and stage annotations.")
        return params_df
    print(f"Subjects: {params_df['subject'].nunique()}")
    print(f"Stages:   {params_df['stage'].unique()}")

    # ── Paired statistics ──────────────────────────────────────────────────────
    stat_rows = []

    for ch_label in params_df["channel"].unique():
        ch_df = params_df[params_df["channel"] == ch_label]

        # Pivot to wide
        metric_cols = [c for c in ch_df.columns
                       if c not in ("subject", "stage", "channel")]
        wide = ch_df.pivot_table(
            index="subject", columns="stage",
            values=metric_cols, aggfunc="first",
        )
        wide.columns = [f"{m}_{s}" for m, s in wide.columns]
        wide = wide.reset_index()

        for s1, s2 in SLEEP_EDF_CONTRASTS:
            contrast_str = f"{s1}->{s2}"
            for b in BANDS_ANALYSIS:
                for metric_type, col_fn in [
                    ("band_power",  lambda b: f"log10_power_{b}"),
                    ("peak_height", lambda b: f"{b}_peak_height_log10"),
                ]:
                    col_base = col_fn(b)
                    ca = f"{col_base}_{s1}"
                    cb = f"{col_base}_{s2}"
                    if ca not in wide.columns or cb not in wide.columns:
                        continue
                    dz, p, n, md = cohens_dz(
                        wide[ca].values.astype(float),
                        wide[cb].values.astype(float),
                    )
                    stat_rows.append(dict(
                        channel=ch_label, band=b, metric_type=metric_type,
                        contrast=contrast_str, n=n,
                        dz=dz, p=p, stars=p_stars(p), mean_delta=md,
                    ))

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(out_dir / "sleep_edf_contrast_stats.csv", index=False)

    # ── Decoupling analysis (core validation) ──────────────────────────────────
    validation_lines = [
        "Sleep-EDF External Validation Report",
        "=" * 60,
        f"N subjects: {params_df['subject'].nunique()}",
        "",
    ]

    for ch_label in params_df["channel"].unique():
        ch_df = params_df[params_df["channel"] == ch_label]
        wide  = ch_df.pivot_table(
            index="subject", columns="stage",
            values=[c for c in ch_df.columns
                    if c not in ("subject","stage","channel")],
            aggfunc="first",
        )
        wide.columns = [f"{m}_{s}" for m, s in wide.columns]
        wide = wide.reset_index()

        validation_lines.append(f"\nChannel: {ch_label}")

        for s1, s2 in SLEEP_EDF_CONTRASTS:
            validation_lines.append(f"\n  Contrast: {s1} → {s2}")

            # Aperiodic exponent
            dz_e, p_e, n_e, _ = cohens_dz(
                wide.get(f"aperiodic_exponent_{s1}",
                         pd.Series(dtype=float)).values,
                wide.get(f"aperiodic_exponent_{s2}",
                         pd.Series(dtype=float)).values,
            )
            validation_lines.append(
                f"  Aperiodic exponent: dz={fmt(dz_e)} p={fmt(p_e,3)} {p_stars(p_e)}"
            )

            # Per-subject deltas
            def get_delta(metric_base: str) -> np.ndarray | None:
                ca = f"{metric_base}_{s1}"
                cb = f"{metric_base}_{s2}"
                if ca not in wide.columns or cb not in wide.columns:
                    return None
                b_ = pd.to_numeric(wide[ca], errors="coerce").values
                a_ = pd.to_numeric(wide[cb], errors="coerce").values
                mask = np.isfinite(b_) & np.isfinite(a_)
                delta = np.full(len(b_), np.nan)
                delta[mask] = a_[mask] - b_[mask]
                return delta

            d_exp = get_delta("aperiodic_exponent")

            for b in BANDS_ANALYSIS:
                d_blp = get_delta(f"log10_power_{b}")
                d_oph = get_delta(f"{b}_peak_height_log10")

                if d_exp is None or d_blp is None or d_oph is None:
                    continue

                mask = np.isfinite(d_exp) & np.isfinite(d_blp) & np.isfinite(d_oph)
                if mask.sum() < 5:
                    continue

                r_blp, p_blp, _ = pearson_r(d_exp[mask], d_blp[mask])
                r_oph, p_oph, _ = pearson_r(d_exp[mask], d_oph[mask])
                di = abs(r_blp) - abs(r_oph) if (np.isfinite(r_blp)
                                                  and np.isfinite(r_oph)) else np.nan

                validation_lines.append(
                    f"    {b:8s}: r(Δexp,ΔBLP)={fmt(r_blp)} {p_stars(p_blp)}"
                    f"  r(Δexp,ΔOPH)={fmt(r_oph)} {p_stars(p_oph)}"
                    f"  DI={fmt(di)}"
                )

    report_text = "\n".join(validation_lines)
    (out_dir / "sleep_edf_validation_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    print("\n" + report_text)
    print(f"\nOutput: {out_dir}")

    return stat_df


def main():
    parser = argparse.ArgumentParser(
        description="External validation using Sleep-EDF Expanded dataset."
    )
    parser.add_argument(
        "--path", type=str, default=str(SLEEP_EDF_PATH),
        help="Directory for Sleep-EDF data (downloads if not present)",
    )
    parser.add_argument(
        "--n-subjects", type=int, default=SLEEP_EDF_N_SUBJECTS,
        help="Number of subjects to include (max 78)",
    )
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    run(Path(args.path), Path(args.out), args.n_subjects)


if __name__ == "__main__":
    main()
