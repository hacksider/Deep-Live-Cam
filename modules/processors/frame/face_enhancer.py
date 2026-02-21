# --- START OF FILE face_enhancer.py ---

from typing import Any, List
import cv2
import threading
import numpy as np
import os
import platform
import torch

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.model_loader import ModelHolder
from modules.paths import MODELS_DIR
from modules.typing import Frame, Face
from modules.utilities import (
    conditional_download,
    is_image,
    is_video,
)

# Allow up to min(cpu_count, 8) concurrent GFPGAN calls to better utilise multi-core hardware.
_SEMAPHORE_COUNT = min(max(1, (os.cpu_count() or 1)), 8)
THREAD_SEMAPHORE = threading.Semaphore(_SEMAPHORE_COUNT)
NAME = "DLC.FACE-ENHANCER"

_model = ModelHolder()


def _select_device() -> torch.device:
    """Select the best available torch device, respecting execution_providers."""
    providers = getattr(modules.globals, "execution_providers", [])

    # If user explicitly requested CPU-only, honour that
    if providers and all(p == "CPUExecutionProvider" for p in providers):
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_gfpgan() -> Any:
    """Load GFPGAN model with device selection and CPU fallback."""
    import gfpgan

    model_path = os.path.join(MODELS_DIR, "GFPGANv1.4.pth")
    device = _select_device()
    print(f"{NAME}: Using {device} device.")

    try:
        enhancer = gfpgan.GFPGANer(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
        print(f"{NAME}: GFPGANer initialized successfully on {device}.")
        return enhancer
    except Exception as e:
        if device.type != "cpu":
            print(f"{NAME}: Error initializing on {device}: {e}. Falling back to CPU.")
            cpu = torch.device("cpu")
            enhancer = gfpgan.GFPGANer(
                model_path=model_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=cpu,
            )
            print(f"{NAME}: GFPGANer initialized on CPU after fallback.")
            return enhancer
        raise


def _warmup_gfpgan(enhancer: Any) -> None:
    """Run a dummy enhancement pass to trigger JIT / compute-plan caching."""
    try:
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        with torch.inference_mode():
            enhancer.enhance(dummy, has_aligned=False, only_center_face=False, paste_back=True)
        print(f"{NAME}: Warmup inference complete.")
    except Exception as e:
        print(f"{NAME}: Warmup skipped (non-fatal): {e}")


def pre_check() -> bool:
    conditional_download(
        MODELS_DIR,
        [
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
        ],
    )
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_face_enhancer() -> Any:
    """Return the GFPGAN singleton, loading on first access with warmup."""
    return _model.get(loader_fn=_load_gfpgan, warmup_fn=_warmup_gfpgan)


def enhance_face(temp_frame: Frame) -> Frame:
    """Enhances faces in a single frame using the global GFPGANer instance."""
    try:
        enhancer = get_face_enhancer()
    except Exception as e:
        print(f"{NAME}: {e}")
        return temp_frame
    try:
        with THREAD_SEMAPHORE:
            with torch.inference_mode():
                _, _, restored_img = enhancer.enhance(
                    temp_frame,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True,
                )
        if restored_img is None:
            return temp_frame
        return restored_img
    except Exception as e:
        print(f"{NAME}: Error during face enhancement: {e}")
        return temp_frame


def process_frame(source_face: Face | None, temp_frame: Frame) -> Frame:
    """Processes a frame: enhances face if detected.

    We run a lightweight InsightFace detection before calling GFPGAN to avoid
    paying the full GFPGAN inference cost on frames that contain no face.
    """
    target_face = get_one_face(temp_frame)
    if target_face is None:
        return temp_frame
    return enhance_face(temp_frame)


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """Processes multiple frames from file paths."""
    modules.processors.frame.core.process_frames_io(
        temp_frame_paths,
        process_fn=lambda frame: process_frame(None, frame),
        progress=progress,
    )


def process_image(
    source_path: str | None, target_path: str, output_path: str
) -> None:
    """Processes a single image file."""
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    cv2.imwrite(output_path, result_frame)
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(
    source_path: str | None, temp_frame_paths: List[str]
) -> None:
    """Processes video frames using the frame processor core."""
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)


def process_frame_v2(temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame)
    return temp_frame

# --- END OF FILE face_enhancer.py ---
