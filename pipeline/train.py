"""
Parallel Training Module
=========================
Trains an XGBoost classifier and performs parallel hyperparameter search.

Logic:
  1. XGBoost natively uses all available CPU cores via its internal `nthread` parameter
     for tree-building parallelism. This gives a strong baseline.
  2. parallel_hyperparameter_search() uses RandomizedSearchCV (from sklearn) with
     `n_jobs=-1` to evaluate multiple hyperparameter combinations in parallel.
     Each combination trains a full XGBoost model on a different fold, so the outer
     parallelism comes from joblib's ProcessPoolExecutor wrapping the cross-validation loop.
  3. Both a serial (single-core XGBoost) and parallel (multi-core XGBoost + parallel CV)
     variant are provided so the benchmark module can measure the wall-clock difference.

Why this matters:
  - Hyperparameter search is trivially parallel: evaluating M parameter sets on K folds
    produces M*K independent train-evaluate jobs.
  - XGBoost's internal parallelism + joblib's outer parallelism together demonstrate
    both task-level and data-level parallelism concepts.
  - The speedup from n_jobs=1 to n_jobs=N is a direct demonstration of Amdahl's Law
    in a real ML setting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from typing import Dict, Any, Tuple

from pipeline.utils import timeit

# -- Written by qqdex --


DEFAULT_PARAM_GRID: Dict[str, Any] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3],
}


def evaluate_model(model: XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Compute standard classification metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


@timeit
def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """
    Train an XGBoost classifier with default hyperparameters.

    n_jobs=1 → single-core (for serial baseline).
    n_jobs=-1 → use all cores (for parallel benchmark).
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        n_jobs=n_jobs,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    return {"model": model, "metrics": metrics}


@timeit
def parallel_hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_iter: int = 20,
    n_jobs: int = 1,
    cv: int = 3,
) -> Dict[str, Any]:
    """
    Parallel randomized hyperparameter search using joblib + XGBoost.

    Pipeline:
      1. RandomizedSearchCV samples `n_iter` parameter combinations from DEFAULT_PARAM_GRID.
      2. For each combination, performs `cv`-fold cross-validation (each fold trains independently).
      3. With `n_jobs=N`, up to N fold-train jobs run concurrently via joblib.
      4. The best model is refit on the full training set and evaluated on the hold-out test set.
    """
    xgb = XGBClassifier(n_jobs=1, random_state=42, eval_metric="logloss")

    search = RandomizedSearchCV(
        xgb,
        param_distributions=DEFAULT_PARAM_GRID,
        n_iter=n_iter,
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)

    metrics = evaluate_model(search.best_estimator_, X_test, y_test)
    return {
        "model": search.best_estimator_,
        "metrics": metrics,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
    }


if __name__ == "__main__":
    from pipeline.data import generate_dataset
    from pipeline.features import serial_feature_engineering

    df = generate_dataset(n_samples=20_000)
    df, _ = serial_feature_engineering(df)

    X = df.drop(columns=["label"]).values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Serial training (1 core)...")
    result_s, t_s = train_xgboost(X_train, y_train, X_test, y_test, n_jobs=1)
    print(f"  Time: {t_s:.2f}s | Metrics: {result_s['metrics']}")

    print("Parallel training (all cores)...")
    result_p, t_p = train_xgboost(X_train, y_train, X_test, y_test, n_jobs=-1)
    print(f"  Time: {t_p:.2f}s | Metrics: {result_p['metrics']}")
    print(f"  Speedup: {t_s / t_p:.2f}x")
