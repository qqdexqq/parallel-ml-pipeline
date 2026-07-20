"""
Benchmark Framework
====================
Runs the full pipeline end-to-end, comparing serial vs parallel execution at each stage.

Stages benchmarked:
  1. Data loading        — serial load vs parallel chunk-based load
  2. Feature engineering — serial transform vs row-sharded parallel transform
  3. Training             — single-core XGBoost vs multi-core XGBoost
  4. Hyperparameter search — n_jobs=1 vs n_jobs=-1 RandomizedSearchCV

For each stage, record:
  - Wall-clock time (seconds)
  - Speedup ratio (t_serial / t_parallel)
  - Efficiency (speedup / n_workers)

The output is a structured report and a matplotlib bar chart.
"""

import json
import os
import time
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from pipeline.data import generate_dataset, serial_load_dataset, parallel_load_dataset
from pipeline.features import serial_feature_engineering, parallel_feature_engineering
from pipeline.train import train_xgboost, parallel_hyperparameter_search
from pipeline.utils import cpu_count_physical, speedup as calc_speedup, efficiency as calc_efficiency

# -- Written by qqdex --


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset.parquet")
BENCHMARK_REPORT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark_report.json")
BENCHMARK_PLOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark_plot.png")


def run_pipeline_benchmark(
    n_samples: int = 500_000,
    n_workers: int = 4,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Execute the full pipeline and collect timing data for every stage.

    Returns a structured dictionary with all metrics.
    """
    n_physical = cpu_count_physical()
    workers = min(n_workers, n_physical)
    print(f"CPU cores available: {n_physical}, using {workers} workers for parallel stages.\n")

    # ---- Generate and save dataset once ----
    print("=" * 60)
    print("STAGE 0: Dataset Generation")
    print("=" * 60)
    t0 = time.perf_counter()
    df_full = generate_dataset(n_samples=n_samples)
    df_full.to_parquet(DATA_PATH, index=False)
    t_gen = time.perf_counter() - t0
    print(f"  Generated {n_samples:,} rows in {t_gen:.2f}s\n")

    # ---- STAGE 1: Data Loading ----
    print("=" * 60)
    print("STAGE 1: Data Loading")
    print("=" * 60)
    _, t_load_serial = serial_load_dataset()
    _, t_load_parallel = parallel_load_dataset(n_workers=workers)
    su_load = calc_speedup(t_load_serial, t_load_parallel)
    ef_load = calc_efficiency(su_load, workers)
    print(f"  Serial:   {t_load_serial:.2f}s")
    print(f"  Parallel: {t_load_parallel:.2f}s ({workers} workers)")
    print(f"  Speedup:  {su_load:.2f}x | Efficiency: {ef_load:.2f}\n")

    # ---- STAGE 2: Feature Engineering ----
    print("=" * 60)
    print("STAGE 2: Feature Engineering")
    print("=" * 60)
    df_for_fe = df_full.copy()

    result_ser, t_fe_serial = serial_feature_engineering(df_for_fe)
    _, t_fe_parallel = parallel_feature_engineering(df_for_fe, n_workers=workers)
    su_fe = calc_speedup(t_fe_serial, t_fe_parallel)
    ef_fe = calc_efficiency(su_fe, workers)
    print(f"  Serial:   {t_fe_serial:.2f}s")
    print(f"  Parallel: {t_fe_parallel:.2f}s ({workers} workers)")
    print(f"  Speedup:  {su_fe:.2f}x | Efficiency: {ef_fe:.2f}")
    print(f"  Output shape: {result_ser.shape}\n")

    # ---- Prepare train/test split for Stage 3 & 4 ----
    X = result_ser.drop(columns=["label"]).values
    y = result_ser["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # ---- STAGE 3: Training ----
    print("=" * 60)
    print("STAGE 3: XGBoost Training")
    print("=" * 60)
    result_tr_s, t_tr_serial = train_xgboost(X_train, y_train, X_test, y_test, n_jobs=1)
    result_tr_p, t_tr_parallel = train_xgboost(X_train, y_train, X_test, y_test, n_jobs=-1)
    su_tr = calc_speedup(t_tr_serial, t_tr_parallel)
    ef_tr = calc_efficiency(su_tr, n_physical)
    print(f"  Serial (1 core):   {t_tr_serial:.2f}s | AUC: {result_tr_s['metrics']['roc_auc']:.4f}")
    print(f"  Parallel (all):    {t_tr_parallel:.2f}s | AUC: {result_tr_p['metrics']['roc_auc']:.4f}")
    print(f"  Speedup:  {su_tr:.2f}x | Efficiency: {ef_tr:.2f}\n")

    # ---- STAGE 4: Hyperparameter Search ----
    print("=" * 60)
    print("STAGE 4: Hyperparameter Search (RandomizedSearchCV)")
    print("=" * 60)
    _, t_hp_serial = parallel_hyperparameter_search(X_train, y_train, X_test, y_test, n_iter=8, n_jobs=1, cv=2)
    result_hp, t_hp_parallel = parallel_hyperparameter_search(
        X_train, y_train, X_test, y_test, n_iter=8, n_jobs=-1, cv=2
    )
    su_hp = calc_speedup(t_hp_serial, t_hp_parallel)
    ef_hp = calc_efficiency(su_hp, n_physical)
    print(f"  Serial (n_jobs=1):  {t_hp_serial:.2f}s")
    print(f"  Parallel (n_jobs=-1): {t_hp_parallel:.2f}s")
    print(f"  Speedup:  {su_hp:.2f}x | Efficiency: {ef_hp:.2f}")
    print(f"  Best params: {result_hp['best_params']}")
    print(f"  Best CV AUC: {result_hp['best_cv_score']:.4f}\n")

    total_serial = t_load_serial + t_fe_serial + t_tr_serial + t_hp_serial
    total_parallel = t_load_parallel + t_fe_parallel + t_tr_parallel + t_hp_parallel
    total_su = calc_speedup(total_serial, total_parallel)

    report = {
        "config": {"n_samples": n_samples, "n_workers": workers, "n_physical_cores": n_physical},
        "stages": [
            {
                "name": "Data Loading",
                "serial_time_s": round(t_load_serial, 2),
                "parallel_time_s": round(t_load_parallel, 2),
                "speedup": round(su_load, 2),
                "efficiency": round(ef_load, 2),
            },
            {
                "name": "Feature Engineering",
                "serial_time_s": round(t_fe_serial, 2),
                "parallel_time_s": round(t_fe_parallel, 2),
                "speedup": round(su_fe, 2),
                "efficiency": round(ef_fe, 2),
            },
            {
                "name": "XGBoost Training",
                "serial_time_s": round(t_tr_serial, 2),
                "parallel_time_s": round(t_tr_parallel, 2),
                "speedup": round(su_tr, 2),
                "efficiency": round(ef_tr, 2),
                "roc_auc": round(result_tr_p["metrics"]["roc_auc"], 4),
            },
            {
                "name": "Hyperparameter Search",
                "serial_time_s": round(t_hp_serial, 2),
                "parallel_time_s": round(t_hp_parallel, 2),
                "speedup": round(su_hp, 2),
                "efficiency": round(ef_hp, 2),
                "best_cv_auc": round(result_hp["best_cv_score"], 4),
            },
        ],
        "totals": {
            "serial_total_s": round(total_serial, 2),
            "parallel_total_s": round(total_parallel, 2),
            "overall_speedup": round(total_su, 2),
            "pipeline_roc_auc": round(result_hp["metrics"]["roc_auc"], 4),
        },
    }

    with open(BENCHMARK_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Benchmark report saved to {BENCHMARK_REPORT_PATH}")

    return report


def print_benchmark_report(report: Dict[str, Any]) -> None:
    """Pretty-print the benchmark report as a formatted table."""
    cfg = report["config"]
    print("\n" + "=" * 75)
    print(" BENCHMARK REPORT")
    print("=" * 75)
    print(f"  Samples: {cfg['n_samples']:,} | Workers: {cfg['n_workers']} | Physical cores: {cfg['n_physical_cores']}")
    print("-" * 75)
    print(f"  {'Stage':<28} {'Serial(s)':>10} {'Parallel(s)':>10} {'Speedup':>8} {'Efficiency':>10}")
    print("-" * 75)
    for st in report["stages"]:
        print(
            f"  {st['name']:<28} {st['serial_time_s']:>10.2f} {st['parallel_time_s']:>10.2f} "
            f"{st['speedup']:>7.2f}x {st['efficiency']:>9.2f}"
        )
    print("-" * 75)
    t = report["totals"]
    print(
        f"  {'TOTAL':<28} {t['serial_total_s']:>10.2f} {t['parallel_total_s']:>10.2f} "
        f"{t['overall_speedup']:>7.2f}x"
    )
    print("-" * 75)
    hp = report["stages"][-1]
    tr = report["stages"][-2]
    print(f"  Final Model ROC-AUC: {tr.get('roc_auc', 'N/A')}")
    print(f"  Best CV ROC-AUC:     {hp.get('best_cv_auc', 'N/A')}")
    print("=" * 75)


def plot_speedup(report: Dict[str, Any]) -> str:
    """
    Generate a grouped bar chart comparing serial vs parallel times across stages,
    plus a speedup line overlay. Saves to BENCHMARK_PLOT_PATH.
    """
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    stages = [s["name"] for s in report["stages"]]
    x = np.arange(len(stages))
    width = 0.35

    serial_times = [s["serial_time_s"] for s in report["stages"]]
    parallel_times = [s["parallel_time_s"] for s in report["stages"]]
    speedups = [s["speedup"] for s in report["stages"]]

    bars1 = ax1.bar(x - width / 2, serial_times, width, label="Serial", color="#e74c3c", alpha=0.85)
    bars2 = ax1.bar(x + width / 2, parallel_times, width, label="Parallel", color="#2ecc71", alpha=0.85)

    ax1.set_ylabel("Wall-clock Time (seconds)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Parallel ML Pipeline Benchmark\n"
        f"({report['config']['n_samples']:,} samples, {report['config']['n_workers']} workers)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=10)
    ax1.legend(loc="upper left", fontsize=10)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.3, f"{height:.1f}s", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.3, f"{height:.1f}s", ha="center", va="bottom", fontsize=8)

    ax2 = ax1.twinx()
    line = ax2.plot(x, speedups, "D-", color="#3498db", linewidth=2.5, markersize=10, label="Speedup")
    ax2.set_ylabel("Speedup Ratio (×)", fontsize=12, fontweight="bold", color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")
    ax2.set_ylim(bottom=0)

    for i, su in enumerate(speedups):
        ax2.annotate(
            f"{su:.1f}×",
            (i, su),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#2980b9",
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(BENCHMARK_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Speedup plot saved to {BENCHMARK_PLOT_PATH}")
    return BENCHMARK_PLOT_PATH
