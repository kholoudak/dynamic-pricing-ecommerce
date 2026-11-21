"""
Step 2 - Project setup / Etape 2 - Creation de l'arborescence du projet.

Run it INSIDE the project folder (Windows 11, PowerShell):
    python setup_project.py
    python setup_project.py --source "C:\\Users\\le\\Downloads\\store-sales-time-series-forecasting"

--source (optional): folder holding the Kaggle CSV files; they are copied into data/raw/.
The script is safe to re-run: it never overwrites an existing file.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FOLDERS = [
    "data/raw", "data/processed",
    "notebooks", "models",
    "src/forecast",
    "outputs/plots",
    "tests",
]

# Kaggle files we want in data/raw (names confirmed by the inspection in Step 1)
RAW_FILES = ["train.csv", "stores.csv", "holidays_events.csv", "oil.csv", "transactions.csv"]

REQUIREMENTS = """\
# Agent 1 - LSTM demand forecasting (CPU is enough; GPU optional)
numpy>=1.26
pandas>=2.1
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
tensorflow>=2.16
pyyaml>=6.0
joblib>=1.3
jupyter>=1.0
pytest>=8.0
"""

CONFIG = """\
# Central configuration - no hardcoded paths in the code.
# All paths are relative to the project root.
seed: 42

paths:
  raw_dir: data/raw
  processed_dir: data/processed
  models_dir: models
  outputs_dir: outputs
  plots_dir: outputs/plots

# Column names CONFIRMED by the Step 1 inspection of train.csv
data:
  train_file: train.csv
  date_col: date
  product_col: family      # product_id = family (sales summed over all stores)
  target_col: sales
  promo_col: onpromotion

# Default values: they will be tuned in Steps 5-6
forecast:
  window_size: 12          # previous N weeks used as input
  horizon: 4               # forecast 1-4 weeks ahead
split:
  val_weeks: 26
  test_weeks: 26

model:
  lstm_units: [64, 32]
  dropout: 0.2
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  early_stopping_patience: 10
"""

GITIGNORE = """\
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
data/raw/*.csv
"""

# One docstring per module: filled in during the next steps.
MODULES = {
    "__init__.py": '"""Agent 1 - LSTM demand forecasting package."""\n\n__version__ = "0.1.0"\n',
    "data_loader.py": '"""Load raw CSV files (Step 3)."""\n',
    "preprocessing.py": '"""Cleaning and weekly aggregation (Step 4)."""\n',
    "feature_engineering.py": '"""Lag, rolling, calendar and holiday features (Step 5)."""\n',
    "sequence_generator.py": '"""Chronological split, scaling and sliding windows (Step 5)."""\n',
    "model.py": '"""Configurable LSTM architecture (Step 6)."""\n',
    "train.py": '"""Training with early stopping and checkpointing (Step 6)."""\n',
    "evaluate.py": '"""MAE / RMSE / MAPE and baseline comparison (Step 7)."""\n',
    "predict.py": '"""predict_demand(product_id, week) and CSV export (Steps 8-9)."""\n',
}

NOTEBOOKS = {
    "01_data_exploration.ipynb": "# 01 - Data exploration",
    "02_data_preprocessing.ipynb": "# 02 - Data preprocessing",
    "03_lstm_training.ipynb": "# 03 - LSTM training",
}


def write_if_missing(path: Path, content: str) -> None:
    """Create a text file only if it does not exist yet."""
    if path.exists():
        print(f"  = exists   {path.relative_to(ROOT)}")
        return
    path.write_text(content, encoding="utf-8")
    print(f"  + created  {path.relative_to(ROOT)}")


def make_notebook(title: str) -> str:
    """Return the JSON of an empty notebook with one markdown title cell."""
    nb = {
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": [title]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1)


def copy_raw_files(source: Path) -> None:
    """Copy the Kaggle CSV files from `source` into data/raw/."""
    if not source.is_dir():
        print(f"\n! Source folder not found: {source}")
        return
    print(f"\nCopying CSV files from {source}")
    for name in RAW_FILES:
        src, dst = source / name, ROOT / "data" / "raw" / name
        if not src.exists():
            print(f"  ! missing in source: {name}")
        elif dst.exists():
            print(f"  = exists   data/raw/{name}")
        else:
            shutil.copy2(src, dst)
            print(f"  + copied   data/raw/{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the Agent 1 project skeleton.")
    parser.add_argument("--source", type=Path, default=None, help="Folder with the Kaggle CSV files")
    args = parser.parse_args()

    print(f"Project root: {ROOT}\n")
    for folder in FOLDERS:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    print(f"Folders ready ({len(FOLDERS)})\n")

    write_if_missing(ROOT / "requirements.txt", REQUIREMENTS)
    write_if_missing(ROOT / "config.yaml", CONFIG)
    write_if_missing(ROOT / ".gitignore", GITIGNORE)
    write_if_missing(ROOT / "README.md", "# Dynamic Pricing & Competition in E-Commerce - Agent 1 (LSTM)\n\n(README completed in Step 10)\n")
    write_if_missing(ROOT / "tests" / ".gitkeep", "")
    for name, doc in MODULES.items():
        write_if_missing(ROOT / "src" / "forecast" / name, doc)
    for name, title in NOTEBOOKS.items():
        write_if_missing(ROOT / "notebooks" / name, make_notebook(title))

    if args.source:
        copy_raw_files(args.source)
    print("\nDone. Next: create the virtual environment and install requirements.txt.")


if __name__ == "__main__":
    main()
