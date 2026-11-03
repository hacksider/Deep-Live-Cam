from typing import Any, List, Optional
import os
import cv2
import threading
import numpy as np
import onnxruntime

import modules.globals
import modules.processors.frame.core
from modules import imread_unicode, imwrite_unicode
from modules.core import update_status
from modules.typing import Frame, Face
from modules.utilities import is_image, is_video
from modules.gpu_processing import gpu_gaussian_blur, gpu_resize

BACKGROUND_SWAPPER: Optional[onnxruntime.InferenceSession] = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.BACKGROUND-SWAPPER"
MODEL_FILE = "modnet_photographic_portrait_matting.onnx"

_cached_bg_path: Optional[str] = None
_cached_bg_image: Optional[np.ndarray] = None


def pre_check() -> bool:
    from modules.model_downloader import ensure_model
    if ensure_model(MODEL_FILE) is None:
        update_status(
            f"Could not obtain {MODEL_FILE}. Place it in the models folder "
            "manually or check your internet connection.",
            NAME,
        )
        return False
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_background_swapper() -> Optional[onnxruntime.InferenceSession]:
    """Initializes and returns the ONNX session for MODNet portrait matting."""
    global BACKGROUND_SWAPPER

    with THREAD_LOCK:
        if BACKGROUND_SWAPPER is None:
            from modules.model_downloader import ensure_model
            model_path = ensure_model(MODEL_FILE, quiet=True)
            if model_path is None or not os.path.exists(model_path):
                return None

            try:
                providers = []
                for p in modules.globals.execution_providers:
                    if p in onnxruntime.get_available_providers():
                        providers.append(p)
                if not providers:
                    providers = ["CPUExecutionProvider"]

                BACKGROUND_SWAPPER = onnxruntime.InferenceSession(
                    model_path,
                    providers=providers
                )
                print(f"[{NAME}] MODNet background matting model loaded successfully.")
            except Exception as e:
                print(f"[{NAME}] Failed to load background matting model: {e}")
                BACKGROUND_SWAPPER = None
                return None

    return BACKGROUND_SWAPPER


def _get_background_image(target_shape: tuple) -> Optional[np.ndarray]:
    """Retrieves and caches the custom background image resized to match target_shape (H, W)."""
    global _cached_bg_path, _cached_bg_image

    bg_path = getattr(modules.globals, "background_path", None)
    if not bg_path or not os.path.isfile(bg_path):
        _cached_bg_path = None
        _cached_bg_image = None
        return None

    target_h, target_w = target_shape[:2]
    if _cached_bg_path != bg_path or _cached_bg_image is None or _cached_bg_image.shape[:2] != (target_h, target_w):
        loaded = imread_unicode(bg_path)
        if loaded is None:
            return None
        _cached_bg_path = bg_path
        _cached_bg_image = gpu_resize(loaded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return _cached_bg_image


def _predict_matte(frame: Frame, session: onnxruntime.InferenceSession) -> np.ndarray:
    """Predicts a float32 matte [0.0, 1.0] representing person foreground probability.

    MODNet standard input: 1x3x512x512, normalized RGB to [-1.0, 1.0].
    """
    orig_h, orig_w = frame.shape[:2]
    # MODNet typically takes 512x512 (multiples of 32)
    in_w, in_h = 512, 512
    resized = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_AREA)
    # BGR -> RGB
    rgb = resized[:, :, ::-1].astype(np.float32)
    # Normalize: (rgb / 127.5) - 1.0
    norm = (rgb - 127.5) / 127.5
    tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})
    matte_raw = outputs[0][0, 0]  # shape (512, 512), values roughly 0 to 1

    matte_resized = cv2.resize(matte_raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    np.clip(matte_resized, 0.0, 1.0, out=matte_resized)
    return matte_resized


def swap_background(temp_frame: Frame) -> Frame:
    """Replaces or blurs the background of temp_frame using portrait segmentation."""
    if temp_frame is None or temp_frame.size == 0:
        return temp_frame

    bg_path = getattr(modules.globals, "background_path", None)
    blur_amount = max(0.0, min(100.0, float(getattr(modules.globals, "background_blur", 0.0))))
    feather_amount = max(0.0, min(50.0, float(getattr(modules.globals, "background_feather", 0.0))))

    has_custom_bg = bg_path is not None and os.path.isfile(bg_path)
    if not has_custom_bg and blur_amount <= 0:
        return temp_frame

    h, w = temp_frame.shape[:2]
    session = get_background_swapper()

    if session is not None:
        try:
            matte = _predict_matte(temp_frame, session)
        except Exception as e:
            print(f"[{NAME}] Segmentation inference failed: {e}")
            matte = None
    else:
        matte = None

    # Fallback heuristic mask if session is not available: center vignette / fallback
    if matte is None:
        matte = np.ones((h, w), dtype=np.float32)
        # Create an elliptical center-focused mask as a graceful fallback
        center = (w // 2, h // 2)
        axes = (int(w * 0.35), int(h * 0.45))
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask_u8, center, axes, 0, 0, 360, 255, -1)
        matte = cv2.GaussianBlur(mask_u8, (51, 51), 20).astype(np.float32) / 255.0

    # Optional feathering on the matte
    if feather_amount > 0:
        k_size = int(feather_amount * 2) | 1
        k_size = max(3, min(k_size, 51))
        matte = cv2.GaussianBlur(matte, (k_size, k_size), 0)

    # Determine destination background
    if has_custom_bg:
        new_bg = _get_background_image(temp_frame.shape)
        if new_bg is None:
            new_bg = temp_frame.copy()
    else:
        new_bg = temp_frame.copy()

    # Apply background blur if requested
    if blur_amount > 0:
        # Scale blur radius based on blur_amount (0.0 - 100.0)
        k = int((blur_amount / 100.0) * 45) | 1
        k = max(3, k)
        new_bg = gpu_gaussian_blur(new_bg, (k, k), 0)

    # Composite: foreground * matte + background * (1 - matte)
    matte_3c = matte[:, :, np.newaxis]
    blended = (temp_frame.astype(np.float32) * matte_3c +
               new_bg.astype(np.float32) * (1.0 - matte_3c))
    return np.clip(blended, 0, 255).astype(np.uint8)


def process_frame(source_face: Optional[Face], temp_frame: Frame, **kwargs: Any) -> Frame:
    """Frame processor standard interface function."""
    return swap_background(temp_frame)


def process_frame_v2(temp_frame: Frame, **kwargs: Any) -> Frame:
    """Live streaming interface function."""
    return swap_background(temp_frame)


def process_frames(source_path: Optional[str], temp_frame_paths: List[str], progress: Any = None) -> None:
    for path in temp_frame_paths:
        if not os.path.exists(path):
            if progress:
                progress.update(1)
            continue
        frame = imread_unicode(path)
        if frame is not None:
            result = swap_background(frame)
            imwrite_unicode(path, result)
        if progress:
            progress.update(1)


def process_image(source_path: Optional[str], target_path: str, output_path: str) -> None:
    target_frame = imread_unicode(target_path)
    if target_frame is None:
        print(f"[{NAME}] Failed to read image: {target_path}")
        return
    result_frame = swap_background(target_frame)
    imwrite_unicode(output_path, result_frame)
    print(f"[{NAME}] Saved processed background image to: {output_path}")


def process_video(source_path: Optional[str], temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames
    )
