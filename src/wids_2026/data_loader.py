"""Data loading utilities for WiDS 2026 project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import DATA_DIR


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def load_train_test_data(
    data_dir: str | Path = DATA_DIR,
    train_file: str = "train.csv",
    test_file: str = "test.csv",
    target_column: str = "event",
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load the training and testing datasets and return them with the target column name."""
    train_path = Path(data_dir) / train_file
    test_path = Path(data_dir) / test_file

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Testing dataset not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data.")

    return train_df, test_df, target_column


def list_csv_files(data_dir: str | Path) -> List[str]:
    """Return CSV files in the provided directory."""
    directory = Path(data_dir)
    if not directory.exists():
        return []
    return sorted(str(path.name) for path in directory.glob("*.csv"))
