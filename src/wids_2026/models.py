"""Model definitions and evaluation utilities for the WiDS project."""

from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model_catalog() -> Dict[str, object]:
    """Create a dictionary of model definitions to compare."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1)),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(random_state=42)),
            ]
        ),
    }


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Evaluate a trained model using common classification metrics."""
    predictions = model.predict(X)
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
    }


def cross_validate_models(X: pd.DataFrame, y: pd.Series, model_catalog: Dict[str, object]) -> Dict[str, float]:
    """Run stratified cross-validation for each model and return mean scores."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: Dict[str, float] = {}

    for name, model in model_catalog.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
        results[name] = float(scores.mean())

    return results


def summarize_classification_report(model, X: pd.DataFrame, y: pd.Series) -> str:
    """Return a formatted classification report."""
    predictions = model.predict(X)
    return classification_report(y, predictions)
