"""Latency benchmark: research model vs deployment XGBoost JSON model.

Measures p50 / p95 / p99 latency (ms) for single-sample and batch inference.
1 000 warm-up calls precede the timed runs to mitigate JIT / cache effects.
"""

from __future__ import annotations

import time

import numpy as np


def _percentiles(times_sec: list[float]) -> dict[str, float]:
    arr = np.array(times_sec) * 1e3
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def benchmark(
    research_fn,
    deploy_fn,
    X_research: np.ndarray,
    X_deploy: np.ndarray | None = None,
    warmup: int = 1_000,
    n_runs: int = 1_000,
    batch_size: int = 1,
) -> dict:
    """Benchmark research model vs deployment model.

    X_research : feature array for the research model (engineered features).
    X_deploy   : feature array for the deployment model (raw features).
                 Falls back to X_research if not provided.
    """
    if X_deploy is None:
        X_deploy = X_research
    rs = X_research[:batch_size].astype(np.float32)
    ds = X_deploy[:batch_size].astype(np.float32)

    for i in range(warmup):
        research_fn(X_research[i % len(X_research): i % len(X_research) + 1].astype(np.float32))
        deploy_fn(X_deploy[i % len(X_deploy): i % len(X_deploy) + 1].astype(np.float32))

    research_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        research_fn(rs)
        research_times.append(time.perf_counter() - t0)

    deploy_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        deploy_fn(ds)
        deploy_times.append(time.perf_counter() - t0)

    research_stats = _percentiles(research_times)
    deploy_stats = _percentiles(deploy_times)
    speedup = research_stats["p50_ms"] / (deploy_stats["p50_ms"] + 1e-9)

    return {
        "batch_size": batch_size,
        "n_runs": n_runs,
        "research": research_stats,
        "deploy": deploy_stats,
        "speedup_p50": round(speedup, 2),
    }


def print_report(result: dict) -> None:
    print(f"\n{chr(9472) * 56}")
    print(f"  Latency Benchmark  (batch_size={result['batch_size']}, n_runs={result['n_runs']})")
    print(f"{chr(9472) * 56}")
    print(f"  {'Metric':<18} {'Research':>12} {'Deployment':>12}")
    print(f"{chr(9472) * 56}")
    for key in ("p50_ms", "p95_ms", "p99_ms"):
        label = key.replace("_ms", "").upper()
        print(f"  {label:<18} {result['research'][key]:>11.3f}ms {result['deploy'][key]:>11.3f}ms")
    print(f"{chr(9472) * 56}")
    print(f"  Speedup (p50): {result['speedup_p50']:.2f}x  (deployment vs research)")
    print(f"{chr(9472) * 56}\n")
