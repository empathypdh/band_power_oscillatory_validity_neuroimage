# Band Power and Oscillatory Inference Framework

This repository contains analysis code and processed data for the manuscript:

**"When Does EEG Band Power Reflect Oscillatory Activity?
A Full-Spectrum Framework for Valid and Misleading Spectral Inference"**

---

## Overview

Band-limited power is widely used in EEG research but is often interpreted as reflecting oscillatory activity without validation.

This project provides an empirical framework identifying when band power yields:

* Valid oscillatory inference
* False-positive inference
* False-negative inference

---

## Repository Structure

* `scripts/`
  Analysis pipeline (parameter extraction, contrasts, classification, validation)

* `results/`
  Processed data, statistical outputs, and figures used in the manuscript

* `data/`
  Derived datasets used for analysis

---

## Reproducibility

Main analysis pipeline:

```bash
python scripts/01_extract_parameters.py
python scripts/02_stage_contrasts.py
python scripts/03_classify_transitions.py
python scripts/04_validation_methods.py
python scripts/05_split_half.py
python scripts/06_sleep_edf_validation.py
```

---

## Notes

* Raw EEG data is not publicly available due to ethical restrictions
* Processed data necessary for reproducing results is included

---

## Contact

Doo-Heum Park, MD, PhD
Konkuk University Medical Center
Seoul, Republic of Korea
