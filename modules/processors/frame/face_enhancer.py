# --- START OF FILE face_enhancer.py ---

from typing import Any, List
import cv2
import threading
import gfpgan
import os
import platform
import torch # Make sure torch is imported

import modules.globals
import modules.processors.frame.core
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
    download_directory_path = models_dir
    conditional_download(
        download_directory_path,
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
    """
    Initializes and returns the GFPGAN face enhancer instance,
    prioritizing CUDA, then MPS (Mac), then CPU.
    """
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
            device = None
            try:
                # Priority 1: CUDA
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    print(f"{NAME}: Using CUDA device.")
                # Priority 2: MPS (Mac Silicon)
                elif platform.system() == "Darwin" and torch.backends.mps.is_available():
                    device = torch.device("mps")
                    print(f"{NAME}: Using MPS device.")
                # Priority 3: CPU
                else:
                    device = torch.device("cpu")
                    print(f"{NAME}: Using CPU device.")

                FACE_ENHANCER = gfpgan.GFPGANer(
                    model_path=model_path,
                    upscale=1,  # upscale=1 means enhancement only, no resizing
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=device
                )
                print(f"{NAME}: GFPGANer initialized successfully on {device}.")

            except Exception as e:
                print(f"{NAME}: Error initializing GFPGANer: {e}")
                # Fallback to CPU if initialization with GPU fails for some reason
                if device is not None and device.type != 'cpu':
                    print(f"{NAME}: Falling back to CPU due to error.")
                    try:
                        device = torch.device("cpu")
                        FACE_ENHANCER = gfpgan.GFPGANer(
                            model_path=model_path,
                            upscale=1,
                            arch='clean',
                            channel_multiplier=2,
                            bg_upsampler=None,
                            device=device
                        )
                        print(f"{NAME}: GFPGANer initialized successfully on CPU after fallback.")
                    except Exception as fallback_e:
                         print(f"{NAME}: FATAL: Could not initialize GFPGANer even on CPU: {fallback_e}")
                         FACE_ENHANCER = None # Ensure it's None if totally failed
                else:
                    # If it failed even on the first CPU attempt or device was already CPU
                     print(f"{NAME}: FATAL: Could not initialize GFPGANer on CPU: {e}")
                     FACE_ENHANCER = None # Ensure it's None if totally failed


    # Check if enhancer is still None after attempting initialization
    if FACE_ENHANCER is None:
        raise RuntimeError(f"{NAME}: Failed to initialize GFPGANer. Check logs for errors.")

    return FACE_ENHANCER


def enhance_face(temp_frame: Frame) -> Frame:
    """Enhances faces in a single frame using the global GFPGANer instance."""
    # Ensure enhancer is ready
    enhancer = get_face_enhancer()
    try:
        with THREAD_SEMAPHORE:
            # The enhance method returns: _, restored_faces, restored_img
            _, _, restored_img = enhancer.enhance(
                temp_frame,
                has_aligned=False, # Assume faces are not pre-aligned
                only_center_face=False, # Enhance all detected faces
                paste_back=True # Paste enhanced faces back onto the original image
            )
        # GFPGAN might return None if no face is detected or an error occurs
        if restored_img is None:
            # print(f"{NAME}: Warning: GFPGAN enhancement returned None. Returning original frame.")
            return temp_frame
        return restored_img
    except Exception as e:
        print(f"{NAME}: Error during face enhancement: {e}")
        # Return the original frame in case of error during enhancement
        return temp_frame


def process_frame(source_face: Face | None, temp_frame: Frame) -> Frame:
    """Processes a frame: enhances face if detected."""
    # We don't strictly need source_face for enhancement only
    # Check if any face exists to potentially save processing time, though GFPGAN also does detection.
    # For simplicity and ensuring enhancement is attempted if possible, we can rely on enhance_face.
    # target_face = get_one_face(temp_frame) # This gets only ONE face
    # If you want to enhance ONLY if a face is detected by your *own* analyser first:
    # has_face = get_one_face(temp_frame) is not None # Or use get_many_faces
    # if has_face:
    #     temp_frame = enhance_face(temp_frame)
    # else: # Enhance regardless, let GFPGAN handle detection
    temp_frame = enhance_face(temp_frame)
    return temp_frame


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """Processes multiple frames from file paths."""
    for temp_frame_path in temp_frame_paths:
        if not os.path.exists(temp_frame_path):
            print(f"{NAME}: Warning: Frame path not found {temp_frame_path}, skipping.")
            if progress:
                progress.update(1)
            continue

        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            print(f"{NAME}: Warning: Failed to read frame {temp_frame_path}, skipping.")
            if progress:
                progress.update(1)
            continue

        result_frame = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result_frame)
        if progress:
            progress.update(1)


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    """Processes a single image file."""
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    cv2.imwrite(output_path, result_frame)
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    """Processes video frames using the frame processor core."""
    # source_path might be optional depending on how process_video is called
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

# Optional: Keep process_frame_v2 if it's used elsewhere, otherwise it's redundant
# def process_frame_v2(temp_frame: Frame) -> Frame:
#     target_face = get_one_face(temp_frame)
#     if target_face:
#         temp_frame = enhance_face(temp_frame)
#     return temp_frame

# --- END OF FILE face_enhancer.py ---