# Parallel ML Pipeline Benchmark

> **Author:** qqdex

End-to-end machine learning pipeline demonstrating **CPU-bound parallelism** across every stage: data loading, feature engineering, model training, and hyperparameter search.

---

## Project Overview

This project is designed to answer one question:

> *"If I throw more CPU cores at my ML pipeline, how much faster does it actually get?"*

It simulates a real-world classification task on a **500,000-row synthetic dataset** and systematically benchmarks the serial (single-core) vs parallel (multi-core) execution of each pipeline stage. The result is a quantitative demonstration of **Amdahl's Law** in a practical ML context — a portfolio-worthy artifact that shows you understand not just *how* to use `multiprocessing`, but *when* and *why* it works.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│              PIPELINE FLOW  —  designed by qqdex                 │
│                                                                  │
│  ┌──────────┐    ┌────────────────┐    ┌──────────┐    ┌───────┐│
│  │  Data    │───▶│    Feature     │───▶│ Training │───▶│  HP   ││
│  │ Loading  │    │  Engineering   │    │ (XGBoost)│    │ Search││
│  └──────────┘    └────────────────┘    └──────────┘    └───────┘│
│       │                │                    │               │    │
│       ▼                ▼                    ▼               ▼    │
│  Chunk-based      Row-sharded         Internal + joblib   joblib│
│  ProcessPool      ProcessPool         tree parallelism   CV loop│
└──────────────────────────────────────────────────────────────────┘
```

### File Map

| File | Responsibility |
|---|---|
| `pipeline/utils.py` | Timers, speedup/efficiency formulas, CPU detection |
| `pipeline/data.py` | Synthetic dataset generation + parallel chunk-based loading |
| `pipeline/features.py` | Row-sharded parallel feature engineering (scaling, encoding, polynomial interactions) |
| `pipeline/train.py` | XGBoost training + RandomizedSearchCV with configurable parallelism |
| `pipeline/benchmarks.py` | Orchestrates full pipeline, collects timing data, generates report + chart |
| `main.py` | CLI entry point with `--samples` and `--workers` flags |

---

## The Logic — In Plain Detail

### Stage 1: Parallel Data Loading

**Problem**: `pd.read_csv("big_file.csv")` is single-threaded. Python reads the file sequentially, parses the CSV row by row, and builds the DataFrame all on one core. For 500k rows with 50 columns, this can take 10–30 seconds of wall-clock time.

**Parallel Strategy**:

1. The dataset is stored in **Parquet** format — a columnar binary format that supports row-group skipping. Unlike CSV, Parquet can be read in chunks without scanning the entire file.
2. The row range `[0, N-1]` is split into `W` contiguous chunks (e.g., 4 workers → 4 chunks of 125k rows each).
3. Each worker process calls `pq.read_table(path).slice(start, end)` to read *only its assigned rows*.
4. Chunks are gathered and `pd.concat()`'d in order.

**Why this works**: Reading is I/O + CPU (parsing). When multiple processes read different byte ranges simultaneously, the OS disk scheduler can reorder requests to minimize seek time. The CPU parsing happens in parallel across cores.

**Key concept**: **Data parallelism** — same operation on different data partitions.

### Stage 2: Parallel Feature Engineering

**Problem**: Feature transformation (scaling, one-hot encoding, polynomial interaction generation) is purely CPU-bound. With 40 numeric features, `PolynomialFeatures(degree=2, interaction_only=True)` generates C(40,2) = 780 new interaction columns. Computing these for 500k rows is O(N × D²).

**Parallel Strategy**:

1. The DataFrame is **row-sharded** into `W` equal slices.
2. Each worker receives a complete copy of its shard via `ProcessPoolExecutor.submit()`.
3. Each worker independently:
   - Fits a `StandardScaler` and `OneHotEncoder` on its *own* shard
   - Applies the transforms
   - Computes polynomial interactions
4. Results are collected and concatenated.

**Why this works**: Feature engineering on row `i` has **zero dependency** on row `j`. The work is *embarrassingly parallel* — each row can be transformed independently. The shards are small enough to fit in each worker's memory, so there's no shared-state contention.

**Nuance**: Each worker fits its own scaler/encoder on its local shard — not the global distribution. This is an approximation that's valid when shards are large enough to be statistically representative. For production, you'd fit once on a sample and broadcast the fitted transformers. This is noted in the code comments as a deliberate simplification for the benchmark.

**Key concept**: **Embarrassingly parallel** decomposition — zero communication, zero synchronization.

### Stage 3: XGBoost Training

**Problem**: Gradient-boosted tree training builds trees sequentially (each tree corrects the residuals of the previous one), so you can't parallelize *across* trees. But you *can* parallelize *within* each tree.

**Parallel Strategy**:

XGBoost uses two levels of parallelism internally:

1. **Feature-level parallelism (column block)**: When finding the best split at each node, the feature space is partitioned across threads. Each thread evaluates candidate splits on a subset of features, then the best split is chosen via a reduction.
2. **Data-level parallelism (row block)**: The gradient histogram used for split finding can be computed in parallel by partitioning rows across threads.

We benchmark `n_jobs=1` vs `n_jobs=-1` to isolate the effect.

**Key concept**: **Task parallelism** — the split-finding sub-problem is decomposed across threads, with a synchronization barrier at each node.

### Stage 4: Parallel Hyperparameter Search

**Problem**: `RandomizedSearchCV` evaluates `M` parameter combinations, each with `K`-fold cross-validation, producing `M × K` independent train-evaluate jobs.

**Parallel Strategy**:

1. `RandomizedSearchCV(n_jobs=-1)` uses `joblib.Parallel` under the hood.
2. Each `(param_set, fold)` pair becomes a unit of work dispatched to the worker pool.
3. Workers train and evaluate independently — no communication needed.
4. Results are collected and the best parameters are selected by `roc_auc`.

**Why this works**: Each fold evaluation is completely independent. With `M=10` combinations and `K=3` folds, there are **30 independent jobs** that can run in parallel. On an 8-core machine, this gives near-8× speedup (minus overhead).

**Key concept**: **Job-level parallelism** via task queue — batch of independent tasks consumed by a worker pool.

---

## Amdahl's Law in This Pipeline

The overall speedup is bounded by Amdahl's Law:

$$S = \frac{1}{(1 - P) + \frac{P}{N}}$$

Where `P` is the parallelizable fraction and `N` is the number of cores.

| Stage | Parallelizable fraction `P` | Expected speedup (8 cores) |
|---|---|---|
| Data Loading | ~0.85 (I/O bound component) | ~5× |
| Feature Engineering | ~0.99 (trivially parallel) | ~7.5× |
| XGBoost Training | ~0.80 (sequential tree building) | ~3.5× |
| HP Search | ~0.99 (independent folds) | ~7.5× |

The pipeline's overall `P` is the weighted average by stage duration. Feature engineering and HP search dominate → high overall speedup.

---

## Setup & Usage

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run full benchmark (500k samples, 4 workers)
python main.py

# Smaller run for quick testing
python main.py --samples 100000 --workers 2

# More aggressive parallelism
python main.py --samples 1000000 --workers 8
```

---

## Expected Output

```
===========================================================================
 BENCHMARK REPORT
===========================================================================
  Samples: 500,000 | Workers: 4 | Physical cores: 8
---------------------------------------------------------------------------
  Stage                        Serial(s)  Parallel(s)  Speedup  Efficiency
---------------------------------------------------------------------------
  Data Loading                     12.50        3.80     3.29x       0.82
  Feature Engineering              45.20        6.10     7.41x       0.93
  XGBoost Training                  8.30        2.90     2.86x       0.36
  Hyperparameter Search            62.00        9.50     6.53x       0.82
---------------------------------------------------------------------------
  TOTAL                           128.00       22.30     5.74x
---------------------------------------------------------------------------
  Final Model ROC-AUC: 0.8732
  Best CV ROC-AUC:     0.8815
===========================================================================
```

Two files are generated:
- `benchmark_report.json` — machine-readable results
- `benchmark_plot.png` — grouped bar chart with speedup overlay

---

## Skills Demonstrated

- **Python multiprocessing**: `ProcessPoolExecutor`, `concurrent.futures`, chunk-based task dispatch
- **Parallel algorithm design**: identifying data-level vs task-level parallelism opportunities
- **ML engineering**: `scikit-learn` pipelines, `XGBoost`, `RandomizedSearchCV`
- **Performance measurement**: wall-clock timing, speedup ratio, parallel efficiency
- **Data formats**: Parquet for chunked I/O, columnar storage tradeoffs
- **Scientific Python**: `numpy`, `pandas`, `matplotlib`, `seaborn`

---

## Next Steps (Week 3–4)

After completing this project, the natural progression is:
1. **GPU-Accelerated Training**: Port the training loop to PyTorch + CUDA
2. **Distributed Training**: Scale beyond one machine with PyTorch DDP
3. **Stream Processing**: Replace batch feature engineering with an online feature store

---

*Built by **qqdex**.*

