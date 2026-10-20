"""High-level pipeline orchestration for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import DEFAULT_TARGET_COLUMN, MODEL_OUTPUT_DIR
from .data_loader import load_train_test_data
from .models import build_model_catalog, cross_validate_models, evaluate_model
from .preprocessing import fill_missing_values, prepare_features, separate_features_target


def train_pipeline(
    data_dir: str | Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
    save_dir: str | Path = MODEL_OUTPUT_DIR,
) -> Tuple[Dict[str, float], object, pd.Series]:
    """Train a model using the available data and return evaluation metrics."""
    train_df, test_df, _ = load_train_test_data(data_dir, target_column=target_column)
    train_df, test_df = fill_missing_values(train_df, test_df)

    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
    X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]

    model_catalog = build_model_catalog()
    cv_results = cross_validate_models(X_train, y_train, model_catalog)

    best_model_name = max(cv_results, key=cv_results.get)
    best_model = model_catalog[best_model_name]
    best_model.fit(X_train, y_train)

    metrics = evaluate_model(best_model, X_test, y_test)

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "best_model.pkl"
    import joblib
    joblib.dump(best_model, model_path)

    return cv_results, best_model, pd.Series(y_test)


def run_experiment(data_dir: str | Path) -> Dict[str, float]:
    """Run the default training workflow and return validation scores."""
    cv_scores, _, _ = train_pipeline(data_dir)
    return cv_scores
