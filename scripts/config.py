# -*- coding: utf-8 -*-
"""
config.py
=========
Central configuration for the band-power sleep-onset analysis pipeline.

All constants used across scripts are defined here. To adapt this pipeline
to a new dataset, edit only this file.

Study:
    "When Does EEG Band Power Reflect Oscillatory Activity?
     Spectral Decomposition of Sleep-Onset EEG Reveals Systematic
     False-Positive, False-Negative, and Mixed Inference Errors
     in Band-Limited Power"

Authors: [authors]
Journal: NeuroImage (2026)
"""

from pathlib import Path
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

# Root folder containing per-subject .dat files
# Override with --root argument at runtime
DEFAULT_ROOT = Path(r"C:\SOP_data\analyzed data-pdh-eLoreta")

# Output directory (created automatically)
OUTPUT_DIR = Path(r"C:\SOP_data\results")

# ── Recording parameters ───────────────────────────────────────────────────────

SFREQ          = 500          # Sampling frequency (Hz)
N_CHANNELS     = 30           # Number of EEG channels
EXPECTED_SAMPLES = 15_000     # 30-second epoch at 500 Hz

# ── File naming conventions ────────────────────────────────────────────────────
# Stage is inferred from the filename of each .dat file

STAGE_PATTERNS = {
    "WS":   r"wake_30s_(\d+)\.dat$",    # Wakefulness Stable
    "TS":   r"trans_30s_(\d+)\.dat$",   # Transition State (operationally defined)
    "ESS1": r"early_30s_(\d+)\.dat$",   # Early Stable Stage 1
    "LSS1": r"late_30s_(\d+)\.dat$",    # Late Stable Stage 1
}

STAGE_ORDER = ["WS", "TS", "ESS1", "LSS1"]

# ── Channel montage (30-channel, original LC order) ───────────────────────────

MONTAGE = [
    "O2",  "O1",  "Oz",   "Pz",   "P4",   "CP4", "P8",  "C4",  "TP8", "T8",
    "P7",  "P3",  "CP3",  "CPz",  "Cz",   "FC4", "FT8", "TP7", "C3",  "FCz",
    "Fz",  "F4",  "F8",   "T7",   "FT7",  "FC3", "F3",  "Fp2", "F7",  "Fp1",
]

# Mapping from file column order to MONTAGE order
DAT_TO_LC = [
    29, 27, 28, 26, 20, 21, 22, 23, 18, 14,
     7,  9, 10, 11,  3,  4,  6,  1,  2,  0,
    25, 19, 15, 12, 13,  5, 24, 16, 17,  8,
]

# ── Frequency bands ────────────────────────────────────────────────────────────

BANDS = {
    "delta":     (1.0,  3.5),
    "theta":     (4.0,  7.5),
    "alpha1":    (8.0,  10.0),
    "alpha2":   (10.5,  12.0),
    "beta1":    (12.5,  18.0),
    "beta2":    (18.5,  21.0),
    "beta3":    (21.5,  30.0),
    "broadband": (1.0,  30.0),
}

BANDS_ANALYSIS = [b for b in BANDS if b != "broadband"]   # 7 bands for statistics

# ── Spatial levels ─────────────────────────────────────────────────────────────

# Level 1: individual electrodes
ELECTRODE_UNITS = {ch: [ch] for ch in MONTAGE}

# Level 2: five lobes
LOBE_UNITS = {
    "Frontal":   ["Fp1","Fp2","F7","F3","Fz","F4","F8","FC3","FCz","FC4"],
    "Central":   ["C3","Cz","C4","CP3","CPz","CP4"],
    "Parietal":  ["P3","Pz","P4"],
    "Temporal":  ["FT7","T7","TP7","P7","FT8","T8","TP8","P8"],
    "Occipital": ["O1","Oz","O2"],
}

# Level 3: nine anatomical ROIs (primary level for paper)
ANAT_UNITS = {
    "FC_midline": ["Fz","FCz","Cz","CPz"],          # Fronto-central midline
    "L_DLPFC":   ["Fp1","F3","FC3"],
    "R_DLPFC":   ["Fp2","F4","FC4"],
    "VmPFC":     ["Fp1","Fp2","Fz","FCz"],          # Ventromedial / orbitofrontal
    "IPG":       ["P7","P8","TP7","TP8"],            # Inferior parietal gyrus
    "SPG":       ["P3","Pz","P4"],                  # Superior parietal gyrus
    "L_STG":     ["F7","FT7","T7"],
    "R_STG":     ["F8","FT8","T8"],
    "Occipital": ["O1","Oz","O2"],
}

ROI_ORDER = list(ANAT_UNITS.keys())   # canonical display order

# ── Stage contrasts ────────────────────────────────────────────────────────────
#
# Sequential contrasts: track fine-grained temporal progression
#   WS → TS      Transition boundary (operationally defined)
#   TS → ESS1    Boundary to early Stage 1
#   ESS1 → LSS1  Stage 1 consolidation
#
# Conventional stage-based contrasts: clinically interpretable benchmarks
#   WS → ESS1    Stable wakefulness vs established Stage 1
#                (primary external comparison; corresponds to W→N1 in Sleep-EDF)
#   WS → LSS1    Full span (corresponds to W→N2 in Sleep-EDF)
#
# Excluded: TS → LSS1
#   The operationally defined TS starting point does not correspond to any
#   standard polysomnographic stage, making cumulative interpretation
#   ambiguous and cross-dataset comparison impossible.

CONTRASTS = [
    ("WS",   "TS"),     # Sequential 1
    ("TS",   "ESS1"),   # Sequential 2
    ("ESS1", "LSS1"),   # Sequential 3
    ("WS",   "ESS1"),   # Conventional 1  ← primary external comparison
    ("WS",   "LSS1"),   # Conventional 2
]

CONTRAST_LABELS = {
    "WS->TS":    "WS → TS",
    "TS->ESS1":  "TS → ESS1",
    "ESS1->LSS1":"ESS1 → LSS1",
    "WS->ESS1":  "WS → ESS1",
    "WS->LSS1":  "WS → LSS1",
}

# ── Aperiodic estimation ───────────────────────────────────────────────────────

APERIODIC_FIT_RANGE = (1.0, 30.0)   # Hz range for log-log linear regression

# ── PSD computation (Welch) ────────────────────────────────────────────────────

WELCH_NPERSEG  = int(4 * SFREQ)     # 4-second windows
WELCH_NOVERLAP = WELCH_NPERSEG // 2 # 50% overlap

# ── Classification thresholds ──────────────────────────────────────────────────

CLS_EFFECT_THRESHOLD = 0.30   # |dz| >= this → meaningful effect
CLS_ZERO_THRESHOLD   = 0.20   # |dz| <= this → effectively zero

# Classification labels
CLS_VALID    = "VALID_OSCILLATORY"       # Concordant band power & peak height
CLS_FP       = "FALSE_POSITIVE"          # Band power change; peak height ~0; exponent-driven
CLS_PWR_ONLY = "POWER_ONLY"             # Band power change; peak height ~0; no exponent link
CLS_FN       = "FALSE_NEGATIVE"          # Peak height change; band power ~0
CLS_OPP      = "OPPOSITE_DIRECTION"     # Both change but in opposing directions
CLS_NOREP    = "NO_CLEAR_PATTERN"       # Neither metric exceeds threshold

CLS_ORDER = [CLS_VALID, CLS_FP, CLS_PWR_ONLY, CLS_FN, CLS_OPP, CLS_NOREP]

CLS_COLORS = {
    CLS_VALID:    "#4CAF50",   # Green
    CLS_FP:       "#E53935",   # Red
    CLS_PWR_ONLY: "#FFC107",   # Amber
    CLS_FN:       "#81C784",   # Light green
    CLS_OPP:      "#1E88E5",   # Blue
    CLS_NOREP:    "#B0BEC5",   # Grey
}

# ── Validation methods ─────────────────────────────────────────────────────────

# specparam (fooof) parameters
SP_PARAMS = dict(
    peak_width_limits = [0.5, 12.0],
    max_n_peaks       = 8,
    min_peak_height   = 0.05,
    peak_threshold    = 2.0,
    aperiodic_mode    = "knee",   # appropriate for sleep EEG
    verbose           = False,
)
SP_FREQ_RANGE = [2, 30]

# IRASA parameters
IRASA_HSET    = np.arange(1.1, 1.95, 0.05)
IRASA_NPERSEG = int(4 * SFREQ)
IRASA_NOVERLAP= IRASA_NPERSEG // 2

# ── Split-half parameters ──────────────────────────────────────────────────────

SPLIT_HALF_N_ITER = 1000
SPLIT_HALF_SEED   = 20260426
SPLIT_HALF_RELIABILITY_THRESHOLD = 90.0   # % directional agreement

# ── Sleep-EDF external validation ─────────────────────────────────────────────

SLEEP_EDF_N_SUBJECTS = 61
SLEEP_EDF_RECORDING  = [1]       # First-night recordings only
SLEEP_EDF_PATH       = Path(r"C:\sleep_edf")

# Stage mapping: Sleep-EDF annotation → analysis stage
SLEEP_EDF_STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
}

# Channels: Sleep-EDF → proxy ROI
SLEEP_EDF_CHANNELS = {
    "fc_midline": "EEG Fpz-Cz",    # Fronto-central midline proxy
    "occipital":  "EEG Pz-Oz",     # Occipital proxy
}

SLEEP_EDF_CONTRASTS = [
    ("W", "N1"),   # Conventional stage-based (corresponds to WS→ESS1)
    ("W", "N2"),   # Full span (corresponds to WS→LSS1)
]

# ── Minimum N ─────────────────────────────────────────────────────────────────

MIN_N_SUBJECTS = 8   # Minimum subjects required for paired statistics
