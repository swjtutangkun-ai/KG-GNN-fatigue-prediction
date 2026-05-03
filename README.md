# KG-GNN Fatigue Prediction

This repository contains the manuscript code for a Knowledge-Guided Graph Neural Network (KG-GNN) for fatigue life prediction of corroded welded joints.

The scripts are renamed according to the corresponding manuscript sections. See `CODE_FILE_MAP.md` for the section-by-section file map and `ENVIRONMENT.md` for runtime environment requirements.

## Quick Start

1. Create the Python environment described in `ENVIRONMENT.md`.
2. Check paths and constants in `config_single.py`.
3. Run the scripts following the manuscript section order.

## Repository Contents

- `02_02_*`: stress-field node extraction, graph construction, and preprocessing.
- `04_*`: model training, overall evaluation, and baseline comparison.
- `05_*`: hyperparameter sensitivity, ablation, and interpretability analysis.
- `06_*`: leave-one-out extrapolation and external BW dataset validation.

## Notes

The large raw/processed datasets and generated model outputs are not included here. Local paths are configured in `config_single.py`.
