"""Tests for NumPy BLAS configuration on Apple Silicon."""
import io
import contextlib
import sys
import platform
import pytest


def _get_numpy_config_string():
    """Capture np.show_config() stdout output (it returns None)."""
    import numpy as np

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        np.show_config()
    return f.getvalue()


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Apple Accelerate BLAS only on Apple Silicon (macOS ARM64)"
)
def test_numpy_uses_accelerate_blas():
    """Verify NumPy is using Apple Accelerate BLAS on macOS ARM."""
    config_str = _get_numpy_config_string().lower()

    assert "accelerate" in config_str or "veclib" in config_str, \
        "NumPy should use Apple Accelerate BLAS on macOS ARM64. " \
        "To fix: set no-binary-package = ['numpy'] in pyproject.toml " \
        "and rebuild with: uv sync"


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Apple Accelerate BLAS only on Apple Silicon (macOS ARM64)"
)
def test_numpy_blas_not_openblas():
    """Verify NumPy BLAS name is not OpenBLAS on Apple Silicon."""
    config_str = _get_numpy_config_string().lower()

    # Check the "name:" field specifically, not the entire output
    # (the output always contains "openblas configuration:" as a field label)
    import re
    blas_name_match = re.search(r"blas:.*?name:\s*(\S+)", config_str, re.DOTALL)
    assert blas_name_match, "Could not find BLAS name in numpy config"
    blas_name = blas_name_match.group(1)

    assert blas_name != "openblas64" and blas_name != "openblas", \
        f"NumPy BLAS is '{blas_name}' — should be 'accelerate' on Apple Silicon. " \
        "Rebuild from source to use Apple Accelerate instead."


def test_numpy_linear_algebra_performance():
    """Benchmark NumPy linear algebra operations (informal performance check)."""
    import numpy as np
    import time

    matrix_size = 512
    iterations = 100

    A = np.random.randn(matrix_size, matrix_size).astype(np.float32)
    B = np.random.randn(matrix_size, matrix_size).astype(np.float32)

    start = time.time()
    for _ in range(iterations):
        np.dot(A, B)
    elapsed = time.time() - start

    assert elapsed > 0, "Matrix multiplication should complete"
