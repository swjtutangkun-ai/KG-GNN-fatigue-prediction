# Runtime Environment

The code was prepared for Python 3.9+ and was developed in a Windows/Anaconda environment. Linux and macOS should also work if PyTorch and PyTorch Geometric are installed for the correct CPU/CUDA backend.

## Core Dependencies

- Python >= 3.9
- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib
- OpenPyXL
- PyTorch
- PyTorch Geometric
- Optuna
- XGBoost

## Recommended Installation

Create a clean environment first:

```bash
conda create -n kg-gnn-fatigue python=3.9
conda activate kg-gnn-fatigue
```

Install the common scientific stack:

```bash
pip install -r requirements.txt
```

If `torch-geometric` fails to install or import, install PyTorch and PyTorch Geometric manually according to your CUDA or CPU version:

- PyTorch: https://pytorch.org/get-started/locally/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

For example, a CPU-only setup can usually start with:

```bash
pip install torch
pip install torch-geometric
```

For GPU training, use the PyTorch command matching your installed CUDA version before installing `torch-geometric`.

## Notes

- The scripts read local data paths from `config_single.py`.
- Excel input files require `openpyxl` through Pandas.
- Baseline comparisons require `xgboost`.
- Hyperparameter search requires `optuna`.
- Training and ablation scripts require `torch` and `torch_geometric`.
