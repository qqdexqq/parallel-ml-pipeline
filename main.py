"""
Parallel ML Pipeline — Entry Point
====================================
Runs the full benchmark and prints a structured report.

Usage:
    python main.py                       # default: 500k samples, 4 workers
    python main.py --samples 200000      # custom sample size
    python main.py --workers 8           # custom worker count
    python main.py --samples 100000 --workers 2

Written by qqdex
"""

import argparse
from pipeline.benchmarks import run_pipeline_benchmark, print_benchmark_report, plot_speedup


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel ML Pipeline Benchmark")
    parser.add_argument("--samples", type=int, default=500_000, help="Number of synthetic samples (default: 500000)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    args = parser.parse_args()

    report = run_pipeline_benchmark(n_samples=args.samples, n_workers=args.workers)
    print_benchmark_report(report)
    plot_speedup(report)


if __name__ == "__main__":
    main()
