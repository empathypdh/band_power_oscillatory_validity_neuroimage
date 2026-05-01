# Band Power vs Oscillatory Validity (NeuroImage)

This repository contains analysis code for the study:

"Band-limited power does not always reflect oscillatory activity: a mechanistic classification framework"

## Pipeline

### Step 1 — Main analysis
Run:
    python 01_main_analysis.py

Generates:
    anatomical_roi_subject_stage_parameters_wide.csv

---

### Step 2 — Validation methods
Run:
    python 02_validation_methods.py

Compares:
- residual-based method
- specparam
- IRASA
- Hilbert amplitude

---

### Step 3 — Split-half reproducibility
Run:
    python 03_split_half_validation.py

Input:
    data/anatomical_roi_subject_stage_parameters_wide.csv

Output:
    split_half_validation_summary.csv
    split_half_validation_all_iterations.csv

---

## Notes

- Raw EEG data are not included.
- The provided CSV file is a derived dataset required for validation.