"""Detect and report NumPy BLAS configuration on Apple Silicon.

This module checks if NumPy is using Apple Accelerate BLAS on macOS ARM64.
If not, it provides guidance on how to build NumPy with Accelerate BLAS.
"""
import sys
import platform
import logging

logger = logging.getLogger(__name__)


def get_numpy_blas_info():
    """Get BLAS configuration information from NumPy.

    Returns:
        dict: Contains 'uses_accelerate' (bool), 'blas_type' (str), 'config' (str)
    """
    import numpy as np

    # Get full config output (not just string representation)
    import io
    import contextlib

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        np.show_config()
    config_str = f.getvalue().lower()

    blas_type = "unknown"
    uses_accelerate = False

    if "accelerate" in config_str or "veclib" in config_str:
        blas_type = "accelerate"
        uses_accelerate = True
    elif "openblas" in config_str:
        blas_type = "openblas"
        uses_accelerate = False
    elif "mkl" in config_str:
        blas_type = "mkl"
        uses_accelerate = False

    return {
        "uses_accelerate": uses_accelerate,
        "blas_type": blas_type,
        "config": config_str,
    }


def check_apple_silicon_blas():
    """Check if running on Apple Silicon and validate BLAS configuration.

    On Apple Silicon (macOS ARM64):
    - Logs a warning if NumPy is not using Accelerate BLAS
    - Returns True if Accelerate is in use, False otherwise

    On other platforms:
    - Returns None (not applicable)

    Returns:
        bool or None: True/False on Apple Silicon, None on other platforms
    """
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return None

    info = get_numpy_blas_info()

    if not info["uses_accelerate"]:
        logger.warning(
            "NumPy on Apple Silicon is using %s instead of Apple Accelerate BLAS. "
            "To improve performance, rebuild NumPy with Accelerate support. "
            "See docs/guides/numpy-accelerate-build.md",
            info["blas_type"],
        )
        return False

    logger.info("NumPy is using Apple Accelerate BLAS on Apple Silicon ✓")
    return True


def log_blas_config():
    """Log full NumPy BLAS configuration for debugging."""
    info = get_numpy_blas_info()
    logger.debug("NumPy BLAS Type: %s", info["blas_type"])
    logger.debug("Uses Accelerate: %s", info["uses_accelerate"])
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Full config:\n%s", info["config"])
