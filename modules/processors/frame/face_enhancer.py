import os
import cv2
import threading
import platform
import torch
import modules
import numpy as np
from typing import Any, List
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
from modules.utilities import (
    conditional_download,
    is_image,
    is_video,
)

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-ENHANCER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def pre_check() -> bool:
    """Ensure required model is downloaded."""
    download_directory_path = models_dir
    conditional_download(
        download_directory_path,
        [
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
        ],
    )
    return True


def pre_start() -> bool:
    """Check if target path is valid before starting."""
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


TENSORRT_AVAILABLE = False
try:
    import tensorrt
    TENSORRT_AVAILABLE = True
except ImportError as im:
    print(f"TensorRT is not available: {im}")
except Exception as e:
    print(f"TensorRT is not available: {e}")


def get_face_enhancer() -> Any:
    """Thread-safe singleton loader for the face enhancer model."""
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
            selected_device = "cpu"
            if TENSORRT_AVAILABLE and torch.cuda.is_available():
                selected_device = "cuda"
            elif torch.cuda.is_available():
                selected_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and platform.system() == "Darwin":
                selected_device = "mps"
            # Import GFPGAN only when needed
            try:
                import gfpgan

                FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1, device=selected_device)
            except Exception as e:
                print(f"Failed to load GFPGAN: {e}")
                FACE_ENHANCER = None
    return FACE_ENHANCER


def enhance_face(temp_frame: Any) -> Any:
    """Enhance a face in the given frame using GFPGAN."""
    with THREAD_SEMAPHORE:
        enhancer = get_face_enhancer()
        if enhancer is None:
            print("Face enhancer model not loaded.")
            return temp_frame
        try:
            _, _, temp_frame = enhancer.enhance(temp_frame, paste_back=True)
        except Exception as e:
            print(f"Face enhancement failed: {e}")
    return temp_frame


def process_frame(source_face: Any, temp_frame: Any) -> Any:
    """Process a single frame for face enhancement."""
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame)
    return temp_frame


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """Process a list of frames for face enhancement, updating progress and handling errors."""
    for temp_frame_path in temp_frame_paths:
        if not os.path.exists(temp_frame_path):
            print(f"{NAME}: Warning: Frame path not found {temp_frame_path}, skipping.")
            if progress:
                progress.update(1)
            continue

        temp_frame = cv2.imread(temp_frame_path)
        try:
            result = process_frame(None, temp_frame)
            cv2.imwrite(temp_frame_path, result)
        except Exception as e:
            print(f"Frame enhancement failed: {e}")
        finally:
            if progress:
                progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Process a single image for face enhancement."""
    target_frame = cv2.imread(target_path)
    result = process_frame(None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Process a video for face enhancement."""
    modules.processors.frame.core.process_video(None, temp_frame_paths, process_frames)


def process_frame_v2(temp_frame: Any) -> Any:
    """Alternative frame processing for face enhancement (for mapped faces, if needed)."""
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame)
    return temp_frame
