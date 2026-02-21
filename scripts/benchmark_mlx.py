#!/usr/bin/env python3
"""Benchmark MLX inswapper vs ONNX Runtime CoreML EP.

Runs N inference calls with random inputs and reports mean/median/p95
latency and effective FPS for each backend.

Usage:
    uv run scripts/benchmark_mlx.py [--runs 100] [--warmup 10]
    uv run scripts/benchmark_mlx.py --correctness   # also check output accuracy

Requirements (macOS ARM only):
    uv pip install mlx onnx
"""
import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np


def _stats(times: List[float]) -> dict:
    arr = np.array(times)
    return {
        "mean_ms":   float(np.mean(arr) * 1000),
        "median_ms": float(np.median(arr) * 1000),
        "p95_ms":    float(np.percentile(arr, 95) * 1000),
        "fps":       float(1.0 / np.mean(arr)),
    }


def _print_stats(label: str, stats: dict) -> None:
    print(
        f"  {label:30s}  mean={stats['mean_ms']:6.1f}ms  "
        f"median={stats['median_ms']:6.1f}ms  "
        f"p95={stats['p95_ms']:6.1f}ms  "
        f"FPS={stats['fps']:5.1f}"
    )


def benchmark_onnx_coreml(onnx_path: str, runs: int, warmup: int) -> Optional[dict]:
    """Benchmark ONNX Runtime with CoreML EP."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [skip] onnxruntime not available")
        return None

    providers = [
        ("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram",
            "MLComputeUnits": "ALL",
            "RequireStaticShapes": 1,
            "SpecializationStrategy": "FastPrediction",
        }),
        "CPUExecutionProvider",
    ]
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as exc:
        print(f"  [skip] session creation failed: {exc}")
        return None

    target = np.random.randn(1, 3, 128, 128).astype(np.float32)
    source = np.random.randn(1, 512).astype(np.float32)
    feed = {session.get_inputs()[0].name: target,
            session.get_inputs()[1].name: source}

    for _ in range(warmup):
        session.run(None, feed)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append(time.perf_counter() - t0)

    return _stats(times)


def benchmark_mlx(onnx_path: str, runs: int, warmup: int) -> Optional[dict]:
    """Benchmark MLX native inference."""
    try:
        import mlx.core as mx
        from modules.mlx_inswapper import MLXSessionWrapper
    except ImportError as exc:
        print(f"  [skip] {exc}")
        return None

    wrapper = MLXSessionWrapper.load(onnx_path)
    if wrapper is None:
        print("  [skip] MLXSessionWrapper.load() returned None")
        return None

    target = np.random.randn(1, 3, 128, 128).astype(np.float32)
    source = np.random.randn(1, 512).astype(np.float32)
    feed = {"target": target, "source": source}

    for _ in range(warmup):
        wrapper.run(None, feed)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        wrapper.run(None, feed)
        times.append(time.perf_counter() - t0)

    return _stats(times)


def correctness_check(onnx_path: str) -> None:
    """Compare MLX and ONNX Runtime outputs on the same random input."""
    print("\nCorrectness check (MLX vs ONNX Runtime CPUExecutionProvider):")
    try:
        import onnxruntime as ort
        from modules.mlx_inswapper import MLXSessionWrapper

        np.random.seed(42)
        target = np.random.randn(1, 3, 128, 128).astype(np.float32)
        source = np.random.randn(1, 512).astype(np.float32)

        # Use CPU provider for reproducibility (CoreML may differ in FP16 rounding)
        ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_feed = {ort_session.get_inputs()[0].name: target,
                    ort_session.get_inputs()[1].name: source}
        ort_out = ort_session.run(None, ort_feed)[0]  # (1, 3, 128, 128)

        mlx_wrapper = MLXSessionWrapper.load(onnx_path)
        mlx_out = mlx_wrapper.run(None, {"target": target, "source": source})[0]

        mae = np.mean(np.abs(ort_out - mlx_out))
        max_err = np.max(np.abs(ort_out - mlx_out))
        print(f"  MAE={mae:.6f}  MaxErr={max_err:.6f}")
        if mae < 0.05:
            print("  PASS — MAE < 0.05 (acceptable for FP16 model)")
        else:
            print("  WARN — MAE ≥ 0.05; check forward pass implementation")

    except Exception as exc:
        print(f"  [error] {exc}")
        import traceback
        traceback.print_exc()


def main() -> int:
    if sys.platform != "darwin":
        print("This benchmark is macOS-only.")
        return 1

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of timed inference calls (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup runs before timing (default: 10)")
    parser.add_argument("--models-dir",
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "models"
                        ),
                        help="Path to models directory")
    parser.add_argument("--correctness", action="store_true",
                        help="Run correctness check comparing MLX vs ONNX Runtime CPU")
    args = parser.parse_args()

    onnx_path = os.path.join(args.models_dir, "inswapper_128_fp16.onnx")
    if not os.path.exists(onnx_path):
        print(f"ERROR: model not found: {onnx_path}")
        print("Run `just setup` to download models.")
        return 1

    # Ensure repo root is importable when running from scripts/
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    print(f"Benchmark: {args.runs} runs, {args.warmup} warmup")
    print(f"Model: {onnx_path}\n")

    results = {}

    print("ONNX Runtime (CoreML EP):")
    stats = benchmark_onnx_coreml(onnx_path, args.runs, args.warmup)
    if stats:
        _print_stats("CoreML EP", stats)
        results["onnx_coreml"] = stats

    print("\nMLX (native Apple Silicon):")
    stats = benchmark_mlx(onnx_path, args.runs, args.warmup)
    if stats:
        _print_stats("MLX", stats)
        results["mlx"] = stats

    if "onnx_coreml" in results and "mlx" in results:
        speedup = results["onnx_coreml"]["mean_ms"] / results["mlx"]["mean_ms"]
        print(
            f"\nSpeedup: MLX is {speedup:.2f}× "
            f"{'faster' if speedup > 1 else 'slower'} than ONNX Runtime CoreML EP"
        )

    if args.correctness:
        correctness_check(onnx_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
