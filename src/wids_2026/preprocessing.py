"""Preprocessing utilities for the WiDS 2026 wildfire pipeline."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def separate_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split feature matrix from target column."""
    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not found in dataframe.")
    target = df[target_column]
    features = df.drop(columns=[target_column])
    return features, target


def prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Align train and test columns for modeling and return feature lists."""
    train_features, _ = separate_features_target(train_df, target_column)
    test_features = test_df.copy()

    common_columns = [column for column in train_features.columns if column in test_features.columns]
    extra_train_cols = [column for column in train_features.columns if column not in common_columns]
    extra_test_cols = [column for column in test_features.columns if column not in common_columns]

    if extra_train_cols or extra_test_cols:
        # Align to the common set, ensuring both datasets have matching columns.
        train_features = train_features[common_columns]
        test_features = test_features[common_columns]

    return train_features, test_features, list(train_features.columns), list(test_features.columns)


def fill_missing_values(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing numeric values with the median from the training data."""
    train_filled = train_df.copy()
    test_filled = test_df.copy()

    for column in train_filled.columns:
        if train_filled[column].dtype.kind in "iufb":
            median_value = train_filled[column].median()
            train_filled[column] = train_filled[column].fillna(median_value)
            if column in test_filled.columns:
                test_filled[column] = test_filled[column].fillna(median_value)

    return train_filled, test_filled
