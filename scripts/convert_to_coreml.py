#!/usr/bin/env python3
"""Convert inswapper_128_fp16.onnx to CoreML .mlpackage for native ANE dispatch.

Requires: coremltools, onnx2torch (both in pyproject.toml as macOS-only deps)

Usage:
    uv run scripts/convert_to_coreml.py [--models-dir models/]

On success, writes models/inswapper_128.mlpackage which face_swapper.py will
use automatically on the next run (bypassing ONNX Runtime's CoreML EP).

Conversion path: ONNX → onnx2torch → TorchScript trace → coremltools → .mlpackage
"""
import argparse
import os
import sys
import time


def convert(models_dir: str) -> int:
    onnx_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
    mlpackage_path = os.path.join(models_dir, "inswapper_128.mlpackage")

    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX model not found: {onnx_path}")
        print("Run `just setup` or download the model first.")
        return 1

    if os.path.exists(mlpackage_path):
        print(f"Already exists: {mlpackage_path}")
        print("Delete it first to force re-conversion.")
        return 0

    try:
        import onnx
        import onnx2torch
        import torch
        import coremltools as ct
    except ImportError as exc:
        print(f"ERROR: Missing dependency: {exc}")
        print("Run: uv sync")
        return 1

    print(f"Loading {onnx_path}...")
    t0 = time.perf_counter()
    onnx_model = onnx.load(onnx_path)

    print("Converting ONNX → PyTorch (onnx2torch)...")
    torch_model = onnx2torch.convert(onnx_model)
    torch_model.eval()
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    print("Tracing to TorchScript...")
    t1 = time.perf_counter()
    # inswapper inputs: target (1,3,128,128) float32, source embedding (1,512) float32
    example_target = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
    example_source = torch.zeros(1, 512, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, (example_target, example_source))
    print(f"  done in {time.perf_counter() - t1:.1f}s")

    print("Converting TorchScript → CoreML (large model — expect 20-40 minutes)...")
    t2 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="target", shape=(1, 3, 128, 128)),
            ct.TensorType(name="source", shape=(1, 512)),
        ],
        outputs=[
            ct.TensorType(name="output"),
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.ALL,
    )
    print(f"  done in {time.perf_counter() - t2:.1f}s")

    print(f"Saving {mlpackage_path}...")
    mlmodel.save(mlpackage_path)
    print(f"Conversion complete in {time.perf_counter() - t0:.1f}s total.")
    print(f"Saved: {mlpackage_path}")
    print()
    print("On next run, face_swapper will automatically use the CoreML model")
    print("when execution provider is coreml.")
    return 0


def main() -> int:
    if sys.platform != "darwin":
        print("CoreML is only available on macOS.")
        return 1

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"),
        help="Path to models directory (default: ../models relative to this script)",
    )
    args = parser.parse_args()

    try:
        return convert(args.models_dir)
    except Exception as exc:
        print(f"ERROR: Conversion failed: {exc}")
        import traceback
        traceback.print_exc()
        print()
        print("Common causes:")
        print("  - Unsupported ONNX operators in inswapper (check traceback above)")
        print("  - Incompatible coremltools version")
        print("  - Insufficient memory during conversion")
        return 1


if __name__ == "__main__":
    sys.exit(main())
