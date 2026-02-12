# --- START OF FILE gpu_processing.py ---
"""
GPU-accelerated image processing using OpenCV CUDA (cv2.cuda.GpuMat).

Provides drop-in replacements for common cv2 functions.  When OpenCV is built
with CUDA support the functions transparently upload → process → download via
GpuMat; otherwise they fall back to the regular CPU path so the rest of the
codebase never has to care whether CUDA is available.

Usage
-----
    from modules.gpu_processing import (
        gpu_gaussian_blur, gpu_sharpen, gpu_add_weighted,
        gpu_resize, gpu_cvt_color, gpu_flip,
        is_gpu_accelerated,
    )
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# CUDA availability detection (evaluated once at import time)
# ---------------------------------------------------------------------------
CUDA_AVAILABLE: bool = False

try:
    # cv2.cuda.GpuMat is only present when OpenCV is compiled with CUDA
    _test_mat = cv2.cuda.GpuMat()
    # Verify we have the required filter / image-processing functions
    _has_gauss = hasattr(cv2.cuda, "createGaussianFilter")
    _has_resize = hasattr(cv2.cuda, "resize")
    _has_cvt = hasattr(cv2.cuda, "cvtColor")
    if _has_gauss and _has_resize and _has_cvt:
        CUDA_AVAILABLE = True
        print("[gpu_processing] OpenCV CUDA support detected – GPU-accelerated processing enabled.")
    else:
        missing = []
        if not _has_gauss:
            missing.append("createGaussianFilter")
        if not _has_resize:
            missing.append("resize")
        if not _has_cvt:
            missing.append("cvtColor")
        print(f"[gpu_processing] cv2.cuda.GpuMat exists but missing: {', '.join(missing)} – falling back to CPU.")
except Exception:
    print("[gpu_processing] OpenCV CUDA not available – using CPU fallback for all operations.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Clip and convert to uint8 if necessary."""
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


def _ksize_odd(ksize: Tuple[int, int]) -> Tuple[int, int]:
    """Ensure kernel dimensions are positive and odd (required by GaussianBlur)."""
    kw = max(1, ksize[0] // 2 * 2 + 1) if ksize[0] > 0 else 0
    kh = max(1, ksize[1] // 2 * 2 + 1) if ksize[1] > 0 else 0
    return (kw, kh)


def _cv_type_for(img: np.ndarray) -> int:
    """Return the OpenCV type constant matching *img* (uint8 only)."""
    channels = 1 if img.ndim == 2 else img.shape[2]
    if channels == 1:
        return cv2.CV_8UC1
    elif channels == 3:
        return cv2.CV_8UC3
    elif channels == 4:
        return cv2.CV_8UC4
    return cv2.CV_8UC3  # fallback


# ---------------------------------------------------------------------------
# Public API – Gaussian Blur
# ---------------------------------------------------------------------------

def gpu_gaussian_blur(
    src: np.ndarray,
    ksize: Tuple[int, int],
    sigma_x: float,
    sigma_y: float = 0,
) -> np.ndarray:
    """Drop-in replacement for ``cv2.GaussianBlur`` with CUDA acceleration.

    Parameters match ``cv2.GaussianBlur(src, ksize, sigmaX, sigmaY)``.
    When *ksize* is ``(0, 0)`` OpenCV computes the kernel size from *sigma_x*.
    """
    if CUDA_AVAILABLE:
        try:
            src_u8 = _ensure_uint8(src)
            cv_type = _cv_type_for(src_u8)
            ks = _ksize_odd(ksize) if ksize != (0, 0) else ksize

            gauss = cv2.cuda.createGaussianFilter(cv_type, cv_type, ks, sigma_x, sigma_y)
            gpu_src = cv2.cuda.GpuMat()
            gpu_src.upload(src_u8)
            gpu_dst = gauss.apply(gpu_src)
            return gpu_dst.download()
        except cv2.error:
            pass

    return cv2.GaussianBlur(src, ksize, sigma_x, sigmaY=sigma_y)


# ---------------------------------------------------------------------------
# Public API – addWeighted
# ---------------------------------------------------------------------------

def gpu_add_weighted(
    src1: np.ndarray,
    alpha: float,
    src2: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Drop-in replacement for ``cv2.addWeighted`` with CUDA acceleration."""
    if CUDA_AVAILABLE:
        try:
            s1 = _ensure_uint8(src1)
            s2 = _ensure_uint8(src2)
            g1 = cv2.cuda.GpuMat()
            g2 = cv2.cuda.GpuMat()
            g1.upload(s1)
            g2.upload(s2)
            gpu_dst = cv2.cuda.addWeighted(g1, alpha, g2, beta, gamma)
            return gpu_dst.download()
        except cv2.error:
            pass

    return cv2.addWeighted(src1, alpha, src2, beta, gamma)


# ---------------------------------------------------------------------------
# Public API – Unsharp-mask sharpening
# ---------------------------------------------------------------------------

def gpu_sharpen(
    src: np.ndarray,
    strength: float,
    sigma: float = 3,
) -> np.ndarray:
    """Unsharp-mask sharpening, optionally GPU-accelerated.

    Equivalent to::

        blurred = GaussianBlur(src, (0,0), sigma)
        result  = addWeighted(src, 1+strength, blurred, -strength, 0)
    """
    if strength <= 0:
        return src

    if CUDA_AVAILABLE:
        try:
            src_u8 = _ensure_uint8(src)
            cv_type = _cv_type_for(src_u8)

            gauss = cv2.cuda.createGaussianFilter(cv_type, cv_type, (0, 0), sigma)
            gpu_src = cv2.cuda.GpuMat()
            gpu_src.upload(src_u8)
            gpu_blurred = gauss.apply(gpu_src)
            gpu_sharp = cv2.cuda.addWeighted(gpu_src, 1.0 + strength, gpu_blurred, -strength, 0)
            result = gpu_sharp.download()
            return np.clip(result, 0, 255).astype(np.uint8)
        except cv2.error:
            pass

    blurred = cv2.GaussianBlur(src, (0, 0), sigma)
    sharpened = cv2.addWeighted(src, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API – Resize
# ---------------------------------------------------------------------------

# Map common cv2 interpolation flags to their CUDA equivalents
_INTERP_MAP = {
    cv2.INTER_NEAREST: cv2.INTER_NEAREST,
    cv2.INTER_LINEAR: cv2.INTER_LINEAR,
    cv2.INTER_CUBIC: cv2.INTER_CUBIC,
    cv2.INTER_AREA: cv2.INTER_AREA,
    cv2.INTER_LANCZOS4: cv2.INTER_LANCZOS4,
}


def gpu_resize(
    src: np.ndarray,
    dsize: Tuple[int, int],
    fx: float = 0,
    fy: float = 0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Drop-in replacement for ``cv2.resize`` with CUDA acceleration.

    Parameters match ``cv2.resize(src, dsize, fx=fx, fy=fy, interpolation=...)``.
    """
    if CUDA_AVAILABLE:
        try:
            src_u8 = _ensure_uint8(src)
            gpu_src = cv2.cuda.GpuMat()
            gpu_src.upload(src_u8)

            interp = _INTERP_MAP.get(interpolation, cv2.INTER_LINEAR)

            if dsize and dsize[0] > 0 and dsize[1] > 0:
                gpu_dst = cv2.cuda.resize(gpu_src, dsize, interpolation=interp)
            else:
                gpu_dst = cv2.cuda.resize(gpu_src, (0, 0), fx=fx, fy=fy, interpolation=interp)

            return gpu_dst.download()
        except cv2.error:
            pass

    return cv2.resize(src, dsize, fx=fx, fy=fy, interpolation=interpolation)


# ---------------------------------------------------------------------------
# Public API – Color conversion
# ---------------------------------------------------------------------------

def gpu_cvt_color(
    src: np.ndarray,
    code: int,
) -> np.ndarray:
    """Drop-in replacement for ``cv2.cvtColor`` with CUDA acceleration.

    Parameters match ``cv2.cvtColor(src, code)``.
    """
    if CUDA_AVAILABLE:
        try:
            src_u8 = _ensure_uint8(src)
            gpu_src = cv2.cuda.GpuMat()
            gpu_src.upload(src_u8)
            gpu_dst = cv2.cuda.cvtColor(gpu_src, code)
            return gpu_dst.download()
        except cv2.error:
            pass

    return cv2.cvtColor(src, code)


# ---------------------------------------------------------------------------
# Public API – Flip
# ---------------------------------------------------------------------------

def gpu_flip(
    src: np.ndarray,
    flip_code: int,
) -> np.ndarray:
    """Drop-in replacement for ``cv2.flip`` with CUDA acceleration.

    Parameters match ``cv2.flip(src, flipCode)``.
    *flip_code*: 0 = vertical, 1 = horizontal, -1 = both.
    """
    if CUDA_AVAILABLE:
        try:
            src_u8 = _ensure_uint8(src)
            gpu_src = cv2.cuda.GpuMat()
            gpu_src.upload(src_u8)
            gpu_dst = cv2.cuda.flip(gpu_src, flip_code)
            return gpu_dst.download()
        except cv2.error:
            pass

    return cv2.flip(src, flip_code)


# ---------------------------------------------------------------------------
# Convenience: check at runtime whether GPU path is active
# ---------------------------------------------------------------------------

def is_gpu_accelerated() -> bool:
    """Return ``True`` when the CUDA path will be used."""
    return CUDA_AVAILABLE

# --- END OF FILE gpu_processing.py ---
