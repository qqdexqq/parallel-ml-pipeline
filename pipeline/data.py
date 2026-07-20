"""
Parallel Data Loading Module
=============================
Produces a synthetic tabular dataset and reads it in parallel through chunk-based loading.

Logic:
  1. generate_dataset() creates N rows with a configurable number of numeric and categorical features,
     plus a binary target column. The data mimics a real-world classification scenario.
  2. The dataset is saved to a single Parquet file (columnar storage chosen for fast chunked reads).
  3. parallel_load_dataset() partitions the row range into equal-sized chunks and dispatches each
     chunk to a worker process via ProcessPoolExecutor.
  4. Each worker (load_chunk) reads ONLY its assigned byte-range or row range using pyarrow's
     row-group skipping. This avoids loading the whole file into memory on any single process.
  5. Chunks are collected and concatenated after all workers finish.

Why this matters:
  - Reading a 500k-row CSV with pandas.read_csv() blocks on a single core for I/O + parsing.
  - Splitting the file into chunks and reading them concurrently saturates the I/O subsystem
    and parallelizes the CPU-bound parsing step.
  - Parquet's columnar format + row-group metadata enables zero-copy chunk reads without
    scanning the entire file.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

from pipeline.utils import timeit

# -- Written by qqdex --

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset.parquet")


def generate_dataset(
    n_samples: int = 500_000,
    n_numeric: int = 40,
    n_categorical: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic classification dataset.

    The target column "label" is generated via a noisy linear combination of the first
    10 features plus a categorical interaction term, pushed through a sigmoid and thresholded.
    This ensures the features carry signal rather than being pure noise.
    """
    rng = np.random.default_rng(seed)

    numeric_data = {
        f"num_{i}": rng.normal(loc=rng.uniform(-3, 3), scale=rng.uniform(0.5, 2.5), size=n_samples)
        for i in range(n_numeric)
    }

    categories = ["A", "B", "C", "D", "E"]
    categorical_data = {
        f"cat_{i}": rng.choice(categories, size=n_samples)
        for i in range(n_categorical)
    }

    df = pd.DataFrame({**numeric_data, **categorical_data})

    logit = np.zeros(n_samples)
    for i in range(min(10, n_numeric)):
        logit += rng.uniform(-2, 2) * df[f"num_{i}"].values
    logit += (df["cat_0"].map({c: i for i, c in enumerate(categories)}).values - 2) * 1.5
    logit += rng.normal(0, 1.5, n_samples)

    prob = 1.0 / (1.0 + np.exp(-logit))
    df["label"] = (prob >= 0.5).astype(int)

    return df


def load_chunk(chunk_indices: Tuple[int, int]) -> pd.DataFrame:
    """
    Read a single row-range chunk from the Parquet file using pyarrow.

    chunk_indices: (start_row, end_row) — both inclusive.
    """
    start, end = chunk_indices
    table = pq.read_table(DATA_PATH)
    return table.slice(start, end - start + 1).to_pandas()


@timeit
def serial_load_dataset() -> pd.DataFrame:
    """Baseline: load the entire Parquet file in one go."""
    return pq.read_table(DATA_PATH).to_pandas()


@timeit
def parallel_load_dataset(n_workers: int = 4) -> pd.DataFrame:
    """
    Load the dataset in parallel by row-range chunking.

    Pipeline:
      1. Open the Parquet file to get the total row count.
      2. Partition [0, total_rows-1] into n_workers contiguous chunks.
      3. Submit each chunk to ProcessPoolExecutor.
      4. Concatenate in order.
    """
    table = pq.read_table(DATA_PATH)
    total_rows = table.num_rows

    chunk_size = total_rows // n_workers
    chunks = []
    for i in range(n_workers):
        start = i * chunk_size
        end = (i + 1) * chunk_size - 1 if i < n_workers - 1 else total_rows - 1
        chunks.append((start, end))

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(load_chunk, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            idx = futures[future]
            results.append((idx, future.result()))

    results.sort(key=lambda x: x[0])
    return pd.concat([df for _, df in results], ignore_index=True)


if __name__ == "__main__":
    print("Generating dataset (500k rows, 50 features)...")
    df = generate_dataset()
    df.to_parquet(DATA_PATH, index=False)
    print(f"Dataset saved to {DATA_PATH} | shape={df.shape}")
