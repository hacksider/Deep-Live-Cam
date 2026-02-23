"""RIFE frame interpolation using rife-ncnn-vulkan.

Supports two backends (checked in order):
  1. Native Python binding: ``rife-ncnn-vulkan-python-tntwise`` (preferred)
     - Zero disk I/O overhead, works directly with NumPy arrays
     - Pre-built wheels for Python 3.9-3.13 on Linux/macOS/Windows
     - Install: ``pip install rife-ncnn-vulkan-python-tntwise``
  2. CLI binary fallback: ``rife-ncnn-vulkan``
     - Requires binary on PATH or in models/rife-ncnn-vulkan/
     - Requires model directories (bundled with binary releases)
     - See: https://github.com/TNTwise/rife-ncnn-vulkan/releases

Supports Practical-RIFE v4.25 and v4.25.lite models.
"""

import glob
import os
import shutil
import subprocess
import sys
from typing import List, Optional

import modules.globals
from modules.paths import MODELS_DIR

NAME = "DLC.RIFE-INTERPOLATION"

RIFE_DIR = os.path.join(MODELS_DIR, "rife-ncnn-vulkan")

AVAILABLE_MODELS = {
    "rife-v4.25": "Practical-RIFE v4.25 (higher quality)",
    "rife-v4.25-lite": "Practical-RIFE v4.25 lite (faster)",
}

DEFAULT_MODEL = "rife-v4.25-lite"

# Cached native Rife instance (avoids repeated model loading)
_NATIVE_RIFE = None
_NATIVE_RIFE_MODEL = None


def _update_status(message: str) -> None:
    """Print status and forward to UI if available."""
    print(f"[{NAME}] {message}")
    try:
        from modules.core import update_status

        update_status(message, NAME)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def has_native_binding() -> bool:
    """Check if the native Python binding is available."""
    try:
        from rife_ncnn_vulkan_python import Rife  # noqa: F401

        return True
    except ImportError:
        return False


def _binary_name() -> str:
    """Return platform-appropriate binary name."""
    if sys.platform == "win32":
        return "rife-ncnn-vulkan.exe"
    return "rife-ncnn-vulkan"


def find_binary() -> Optional[str]:
    """Find rife-ncnn-vulkan binary on PATH or in models directory."""
    path_binary = shutil.which("rife-ncnn-vulkan")
    if path_binary:
        return path_binary

    local_binary = os.path.join(RIFE_DIR, _binary_name())
    if os.path.isfile(local_binary) and os.access(local_binary, os.X_OK):
        return local_binary

    return None


def find_model_dir(model_name: str) -> Optional[str]:
    """Find model directory for the given model name.

    Searches in order:
      1. models/rife-ncnn-vulkan/<model_name>/
      2. models/<model_name>/
    """
    model_path = os.path.join(RIFE_DIR, model_name)
    if os.path.isdir(model_path):
        return model_path

    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.isdir(model_path):
        return model_path

    return None


def _get_backend() -> str:
    """Determine which backend to use: 'native' or 'cli' or 'none'."""
    if has_native_binding():
        return "native"
    if find_binary():
        return "cli"
    return "none"


def pre_check() -> bool:
    """Verify RIFE interpolation requirements.

    Returns True if RIFE is ready to use, or if RIFE is disabled.
    """
    if not getattr(modules.globals, "rife_enabled", False):
        return True

    backend = _get_backend()

    if backend == "native":
        _update_status("RIFE: using native Python binding (rife-ncnn-vulkan-python)")
        return True

    if backend == "cli":
        model_name = getattr(modules.globals, "rife_model", DEFAULT_MODEL)
        model_dir = find_model_dir(model_name)
        if not model_dir:
            _update_status(
                f"RIFE model '{model_name}' not found. Place model files in "
                f"models/rife-ncnn-vulkan/{model_name}/ or models/{model_name}/"
            )
            return False
        _update_status("RIFE: using CLI binary fallback")
        return True

    _update_status(
        "RIFE interpolation requires either:\n"
        "  1. pip install rife-ncnn-vulkan-python-tntwise  (recommended)\n"
        "  2. rife-ncnn-vulkan binary on PATH or in models/rife-ncnn-vulkan/\n"
        "See: https://github.com/TNTwise/rife-ncnn-vulkan"
    )
    return False


# ---------------------------------------------------------------------------
# Frame counting
# ---------------------------------------------------------------------------

def _count_frames(directory: str) -> int:
    """Count image files in a directory."""
    count = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        count += len(glob.glob(os.path.join(directory, ext)))
    return count


# ---------------------------------------------------------------------------
# Native Python binding backend
# ---------------------------------------------------------------------------

def _get_native_rife():
    """Get or create a cached native Rife instance."""
    global _NATIVE_RIFE, _NATIVE_RIFE_MODEL

    model_name = getattr(modules.globals, "rife_model", DEFAULT_MODEL)

    if _NATIVE_RIFE is not None and _NATIVE_RIFE_MODEL == model_name:
        return _NATIVE_RIFE

    from rife_ncnn_vulkan_python import Rife

    # The native binding resolves bundled model names automatically.
    # For custom model dirs, we'd pass the path instead.
    model_dir = find_model_dir(model_name)
    model_arg = model_dir if model_dir else model_name

    _NATIVE_RIFE = Rife(gpuid=0, model=model_arg, uhd_mode=False)
    _NATIVE_RIFE_MODEL = model_name
    _update_status(f"RIFE native engine loaded: {model_name}")
    return _NATIVE_RIFE


def _interpolate_native(temp_directory_path: str) -> Optional[int]:
    """Interpolate frames using the native Python binding."""
    import cv2

    rife = _get_native_rife()
    multiplier = getattr(modules.globals, "rife_multiplier", 2)

    # Read all frames sorted
    frame_paths = sorted(glob.glob(os.path.join(temp_directory_path, "*.jpg")))
    if len(frame_paths) < 2:
        _update_status("Not enough frames for interpolation (need at least 2)")
        return None

    input_count = len(frame_paths)
    _update_status(
        f"Running RIFE interpolation (native, "
        f"{getattr(modules.globals, 'rife_model', DEFAULT_MODEL)}, {multiplier}x) "
        f"on {input_count} frames..."
    )

    # Read all frames into memory
    frames = []
    for path in frame_paths:
        img = cv2.imread(path)
        if img is not None:
            frames.append(img)
        else:
            _update_status(f"Warning: could not read {path}, skipping")

    if len(frames) < 2:
        _update_status("Not enough readable frames for interpolation")
        return None

    # Generate interpolated sequence
    output_frames = []
    # Number of intermediate frames between each pair
    n_intermediate = multiplier - 1

    for i in range(len(frames) - 1):
        output_frames.append(frames[i])

        for step_idx in range(1, n_intermediate + 1):
            timestep = step_idx / multiplier
            try:
                interpolated = rife.process_cv2(frames[i], frames[i + 1], timestep=timestep)
                output_frames.append(interpolated)
            except Exception as e:
                _update_status(f"Warning: interpolation failed at frame {i}, step {step_idx}: {e}")
                # Skip this intermediate frame on failure

    # Add the last frame
    output_frames.append(frames[-1])

    # Clear original frames
    for f in frame_paths:
        os.remove(f)

    # Write output frames with sequential numbering
    for i, frame in enumerate(output_frames, start=1):
        out_path = os.path.join(temp_directory_path, f"{i:04d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    final_count = len(output_frames)
    _update_status(
        f"RIFE interpolation complete: {input_count} -> {final_count} frames"
    )
    return final_count


# ---------------------------------------------------------------------------
# CLI binary backend
# ---------------------------------------------------------------------------

def _build_command(
    binary: str,
    input_dir: str,
    output_dir: str,
    model_dir: str,
    input_frame_count: int,
    multiplier: int,
) -> List[str]:
    """Build the rife-ncnn-vulkan command line."""
    target_frames = input_frame_count * multiplier

    cmd = [
        binary,
        "-i",
        input_dir,
        "-o",
        output_dir,
        "-m",
        model_dir,
        "-n",
        str(target_frames),
        "-f",
        "%04d.jpg",
        "-g",
        "auto",
    ]
    return cmd


def _interpolate_cli(temp_directory_path: str) -> Optional[int]:
    """Interpolate frames using the CLI binary."""
    binary = find_binary()
    if not binary:
        _update_status("rife-ncnn-vulkan binary not found, skipping interpolation")
        return None

    model_name = getattr(modules.globals, "rife_model", DEFAULT_MODEL)
    model_dir = find_model_dir(model_name)
    if not model_dir:
        _update_status(f"RIFE model '{model_name}' not found, skipping interpolation")
        return None

    multiplier = getattr(modules.globals, "rife_multiplier", 2)
    if multiplier < 2:
        multiplier = 2

    input_frame_count = _count_frames(temp_directory_path)
    if input_frame_count < 2:
        _update_status("Not enough frames for interpolation (need at least 2)")
        return None

    rife_output_dir = temp_directory_path + "_rife"
    os.makedirs(rife_output_dir, exist_ok=True)

    cmd = _build_command(
        binary, temp_directory_path, rife_output_dir, model_dir,
        input_frame_count, multiplier,
    )

    _update_status(
        f"Running RIFE interpolation (CLI, {model_name}, {multiplier}x) "
        f"on {input_frame_count} frames..."
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() if result.stderr else "unknown error"
            _update_status(f"RIFE interpolation failed: {stderr}")
            shutil.rmtree(rife_output_dir, ignore_errors=True)
            return None
    except subprocess.TimeoutExpired:
        _update_status("RIFE interpolation timed out")
        shutil.rmtree(rife_output_dir, ignore_errors=True)
        return None
    except FileNotFoundError:
        _update_status(f"rife-ncnn-vulkan binary not found at {binary}")
        shutil.rmtree(rife_output_dir, ignore_errors=True)
        return None
    except Exception as e:
        _update_status(f"RIFE interpolation error: {e}")
        shutil.rmtree(rife_output_dir, ignore_errors=True)
        return None

    output_frame_count = _count_frames(rife_output_dir)
    if output_frame_count == 0:
        _update_status("RIFE produced no output frames")
        shutil.rmtree(rife_output_dir, ignore_errors=True)
        return None

    # Rename output frames to sequential %04d.jpg format
    output_files = sorted(glob.glob(os.path.join(rife_output_dir, "*")))
    for i, src in enumerate(output_files, start=1):
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(rife_output_dir, f"{i:04d}.jpg")
        if src != dst:
            if ext == ".png":
                import cv2

                img = cv2.imread(src)
                if img is not None:
                    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    os.remove(src)
            else:
                os.rename(src, dst)

    # Replace original frames with interpolated ones
    for f in glob.glob(os.path.join(temp_directory_path, "*.jpg")):
        os.remove(f)
    for f in glob.glob(os.path.join(temp_directory_path, "*.png")):
        os.remove(f)

    for f in glob.glob(os.path.join(rife_output_dir, "*.jpg")):
        shutil.move(f, os.path.join(temp_directory_path, os.path.basename(f)))

    shutil.rmtree(rife_output_dir, ignore_errors=True)

    final_count = _count_frames(temp_directory_path)
    _update_status(
        f"RIFE interpolation complete: {input_frame_count} -> {final_count} frames"
    )
    return final_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def interpolate_frames(temp_directory_path: str) -> Optional[int]:
    """Run RIFE frame interpolation on extracted video frames.

    Automatically selects the best available backend (native > CLI).
    Interpolates frames in-place: the original frames in temp_directory_path
    are replaced with the interpolated output.

    Args:
        temp_directory_path: Directory containing numbered frame images
            (e.g., 0001.jpg, 0002.jpg, ...)

    Returns:
        The new frame count after interpolation, or None if interpolation
        failed or was skipped.
    """
    backend = _get_backend()

    if backend == "native":
        return _interpolate_native(temp_directory_path)
    elif backend == "cli":
        return _interpolate_cli(temp_directory_path)
    else:
        _update_status("No RIFE backend available, skipping interpolation")
        return None
