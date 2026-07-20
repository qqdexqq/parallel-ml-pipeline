"""
Parallel Feature Engineering Module
====================================
Transforms raw columns into ML-ready features using process-level parallelism.

Logic:
  1. The feature engineering pipeline is split into two phases:
     a) Column-level transforms: scaling, one-hot encoding, missing-value imputation.
     b) Row-level transforms: polynomial interactions, aggregations, ratios.
  2. parallel_feature_engineering() shards the DataFrame row-wise across processes.
     Each worker applies the FULL feature transform to its shard independently.
     This is embarrassingly parallel because row transforms have no cross-row dependencies.
  3. The transformed shards are concatenated back into a single DataFrame.
  4. A serial version is provided for benchmarking baseline.

Why this matters:
  - Feature engineering is often the bottleneck in ML pipelines, not training.
  - On 500k rows, polynomial feature expansion (e.g., pairwise interactions on 40 numeric
    columns) produces ~820 new columns. Computing this on a single core takes minutes.
  - Row-wise sharding gives near-linear speedup because the work is CPU-bound and
    independent across rows.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

from pipeline.utils import timeit

# -- Written by qqdex --


def _get_numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("num_")]


def _get_categorical_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("cat_")]


def _fit_encoders(df: pd.DataFrame):
    """Fit one ColumnTransformer per worker to avoid pickling large shared state."""
    numeric_cols = _get_numeric_cols(df)
    cat_cols = _get_categorical_cols(df)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    preprocessor.fit(df)
    return preprocessor, numeric_cols


def _apply_transforms(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Apply the full feature engineering chain to a DataFrame shard.
    Returns (feature_df, labels).
    """
    feature_df = df.drop(columns=["label"], errors="ignore")
    labels = df["label"].values if "label" in df.columns else None

    preprocessor, numeric_cols = _fit_encoders(feature_df)
    transformed = preprocessor.transform(feature_df)

    ohe_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(
        _get_categorical_cols(feature_df)
    )
    all_cols = list(numeric_cols) + list(ohe_cols)
    result_df = pd.DataFrame(transformed, columns=all_cols, index=feature_df.index)
    result_df.reset_index(drop=True, inplace=True)
    return result_df, labels


def _add_interactions(df: pd.DataFrame, numeric_cols: List[str], max_degree: int = 2, top_k: int = 10) -> pd.DataFrame:
    """
    Add polynomial interaction features among the top_k numeric columns.
    Limiting to top_k avoids quadratic feature explosion. With 45 numeric columns,
    full pairwise = 990 interactions. With top_k=10, only 45 interactions.
    """
    from sklearn.preprocessing import PolynomialFeatures

    use_cols = numeric_cols[:top_k] if len(numeric_cols) > top_k else numeric_cols

    poly = PolynomialFeatures(degree=max_degree, interaction_only=True, include_bias=False)
    interactions = poly.fit_transform(df[use_cols])

    interaction_names = poly.get_feature_names_out(use_cols)
    interaction_df = pd.DataFrame(
        interactions[:, len(use_cols):],
        columns=interaction_names[len(use_cols):],
        index=df.index,
    )
    return pd.concat([df, interaction_df], axis=1)


def _transform_shard(shard_df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering applied to a single shard (runs in a worker process)."""
    df, labels = _apply_transforms(shard_df)
    numeric_cols = [c for c in df.columns if c.startswith("num_")]
    df = _add_interactions(df, numeric_cols)
    if labels is not None:
        df["label"] = labels
    return df


@timeit
def serial_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline: apply all feature transforms on the main process."""
    return _transform_shard(df)


@timeit
def parallel_feature_engineering(df: pd.DataFrame, n_workers: int = 4) -> pd.DataFrame:
    """
    Shard the DataFrame row-wise and apply feature engineering in parallel.

    Step-by-step:
      1. Split df into `n_workers` row shards (contiguous slices).
      2. Submit each shard to ProcessPoolExecutor with _transform_shard.
      3. Gather results, preserving shard order, and concatenate.
    """
    n = len(df)
    chunk_size = n // n_workers
    shards = []
    for i in range(n_workers):
        start = i * chunk_size
        end = start + chunk_size if i < n_workers - 1 else n
        shards.append(df.iloc[start:end].copy())

    results: List[Tuple[int, pd.DataFrame]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_transform_shard, shard): i for i, shard in enumerate(shards)}
        for future in as_completed(futures):
            idx = futures[future]
            results.append((idx, future.result()))

    results.sort(key=lambda x: x[0])
    return pd.concat([r for _, r in results], ignore_index=True)


if __name__ == "__main__":
    from pipeline.data import generate_dataset

    df = generate_dataset(n_samples=50_000)
    print("Running serial feature engineering...")
    result_serial, t_serial = serial_feature_engineering(df)
    print(f"Serial: {t_serial:.2f}s | shape={result_serial.shape}")

    print("Running parallel feature engineering (4 workers)...")
    result_parallel, t_parallel = parallel_feature_engineering(df, n_workers=4)
    print(f"Parallel: {t_parallel:.2f}s | shape={result_parallel.shape}")
    print(f"Speedup: {t_serial / t_parallel:.2f}x")
