# WiDS 2026 Wildfire Prediction

A professional Python project template for the WiDS 2026 wildfire danger prediction workflow originally developed in the notebook.

## Project overview

This repository focuses on predicting wildfire danger using a structured environmental dataset. The goal is to combine exploratory data analysis, preprocessing, model training, and evaluation in a clean and reusable Python project structure.

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── src/
│   └── wids_2026/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── models.py
│       ├── visualization.py
│       └── pipeline.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
└── notebooks/
    └── .gitkeep
```

## Features

- robust CSV loading utilities
- reusable preprocessing pipeline
- model training and evaluation helpers
- visualization functions for EDA and feature importance
- command-line training and evaluation entrypoints
- package-based structure for easier extension

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Atif-eng/WDIS-2026.git
cd WDIS-2026
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or .venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train a model

```bash
python scripts/train.py --data-dir /path/to/data --target event
```

### 5. Evaluate a trained model

```bash
python scripts/evaluate.py --test-data /path/to/test.csv --model-path models/best_model.pkl
```

## Expected data format

The project expects CSV files similar to the WiDS datathon structure:

- `train.csv`
- `test.csv`
- optionally `metadata` files or other supporting files

The target column is expected to be named `event` by default, but this can be overridden in training.

## Notes

This repo was turned into a cleaner and more maintainable Python project while preserving the notebook-based analysis concept. The original notebook remains a good reference for EDA and experimentation, while the Python modules make the workflow easier to extend, automate, and deploy.

## Author

Muhammad Atif
