"""Background Swapper frame processor with 3D perspective / parallax warping.

Segments the foreground person using portrait matting (MODNet ONNX) and
blends with a user-selected background image. Dynamically adjusts the
background perspective according to the viewer's head pose / point of view
(yaw, pitch, and x/y displacement) to create a realistic 3D parallax effect.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

import modules.globals
from modules import imread_unicode, imwrite_unicode
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.gpu_processing import gpu_resize
from modules.paths import MODELS_DIR
from modules.platform_info import OPENVINO_PROVIDER_CONFIG
from modules.typing import Face, Frame

NAME = "DLC.BACKGROUND-SWAPPER"
MODEL_NAME = "modnet_photographic_portrait_matting.onnx"

SEGMENTATION_SESSION: Optional[ort.InferenceSession] = None
THREAD_LOCK = threading.Lock()

# Cache for loaded background image
_CACHED_BG_PATH: Optional[str] = None
_CACHED_BG_IMAGE: Optional[np.ndarray] = None

# Temporal smoothing for viewer's point of view (x, y, yaw, pitch, roll)
_SMOOTHED_POV: Optional[Tuple[float, float, float, float, float]] = None


def pre_check() -> bool:
    """Ensure the segmentation model is present or can be downloaded."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    from modules.model_downloader import ensure_model

    model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    if not os.path.isfile(model_path) or os.path.getsize(model_path) == 0:
        downloaded = ensure_model(MODEL_NAME)
        if downloaded is None or not os.path.isfile(downloaded):
            update_status(
                f"Could not obtain {MODEL_NAME}. Place it in the models folder manually.",
                NAME,
            )
            return False
    return True


def pre_start() -> bool:
    """Check readiness before starting processing."""
    bg_path = getattr(modules.globals, "background_path", None)
    if not bg_path or not os.path.isfile(bg_path):
        update_status("No background image selected or file does not exist.", NAME)
        return False
    return get_segmentation_session() is not None


def get_segmentation_session() -> Optional[ort.InferenceSession]:
    """Load or retrieve the ONNX inference session for portrait matting."""
    global SEGMENTATION_SESSION
    with THREAD_LOCK:
        if SEGMENTATION_SESSION is None:
            model_path = os.path.join(MODELS_DIR, MODEL_NAME)
            if not os.path.isfile(model_path):
                if not pre_check():
                    return None

            providers_config = []
            for p in modules.globals.execution_providers:
                if p == "OpenVINOExecutionProvider":
                    providers_config.append(OPENVINO_PROVIDER_CONFIG)
                else:
                    providers_config.append(p)

            try:
                SEGMENTATION_SESSION = ort.InferenceSession(
                    model_path,
                    providers=providers_config,
                )
            except Exception as exc:
                update_status(f"Error loading segmentation model: {exc}", NAME)
                SEGMENTATION_SESSION = None
    return SEGMENTATION_SESSION


def get_background_image() -> Optional[np.ndarray]:
    """Retrieve and cache the user's selected background image."""
    global _CACHED_BG_PATH, _CACHED_BG_IMAGE
    bg_path = getattr(modules.globals, "background_path", None)
    if not bg_path:
        return None
    if _CACHED_BG_PATH != bg_path:
        _CACHED_BG_PATH = bg_path
        _CACHED_BG_IMAGE = imread_unicode(bg_path)
    return _CACHED_BG_IMAGE


def compute_viewer_pov(
    face: Optional[Face], frame_shape: Tuple[int, int]
) -> Tuple[float, float, float, float, float]:
    """Estimate viewer point of view (dx, dy, yaw, pitch, roll) from face.

    dx, dy are normalized offsets in [-1.0, 1.0] relative to frame center.
    yaw, pitch, roll are head rotation angles in degrees.
    """
    global _SMOOTHED_POV
    h, w = frame_shape[:2]

    if face is None:
        raw_pov = (0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        # 1. Position displacement relative to frame center
        center_x, center_y = w * 0.5, h * 0.5
        if hasattr(face, "bbox") and face.bbox is not None:
            bx1, by1, bx2, by2 = face.bbox[:4]
            fc_x = (bx1 + bx2) * 0.5
            fc_y = (by1 + by2) * 0.5
        elif hasattr(face, "kps") and face.kps is not None and len(face.kps) > 0:
            fc_x = float(np.mean(face.kps[:, 0]))
            fc_y = float(np.mean(face.kps[:, 1]))
        else:
            fc_x, fc_y = center_x, center_y

        dx = float(np.clip((fc_x - center_x) / (center_x or 1.0), -1.0, 1.0))
        dy = float(np.clip((fc_y - center_y) / (center_y or 1.0), -1.0, 1.0))

        # 2. Head pose angles
        yaw, pitch, roll = 0.0, 0.0, 0.0
        if hasattr(face, "pose") and face.pose is not None:
            try:
                # insightface pose is [pitch, yaw, roll] in degrees
                pitch = float(face.pose[0])
                yaw = float(face.pose[1])
                roll = float(face.pose[2])
            except (IndexError, TypeError):
                pass
        elif hasattr(face, "kps") and face.kps is not None and len(face.kps) >= 5:
            # Estimate from standard 5 facial landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
            kps = face.kps
            left_eye = kps[0]
            right_eye = kps[1]
            nose = kps[2]
            eye_mid = (left_eye + right_eye) * 0.5
            eye_dist = max(1.0, np.linalg.norm(right_eye - left_eye))
            yaw = float(np.clip((nose[0] - eye_mid[0]) / (eye_dist * 0.5) * 35.0, -45.0, 45.0))
            pitch = float(np.clip((nose[1] - eye_mid[1]) / (eye_dist * 0.5) * 30.0, -35.0, 35.0))

        raw_pov = (dx, dy, yaw, pitch, roll)

    # Temporal exponential moving average for smooth transitions
    if _SMOOTHED_POV is None:
        _SMOOTHED_POV = raw_pov
    else:
        alpha = 0.35  # responsiveness vs smoothness
        _SMOOTHED_POV = tuple(
            float(alpha * curr + (1.0 - alpha) * prev)
            for curr, prev in zip(raw_pov, _SMOOTHED_POV)
        )
    return _SMOOTHED_POV


def render_3d_perspective_background(
    bg_image: np.ndarray,
    target_shape: Tuple[int, int],
    dx: float,
    dy: float,
    yaw: float,
    pitch: float,
    roll: float = 0.0,
) -> np.ndarray:
    """Warp background image using a 3D perspective homography.

    Adapts the background to the viewer's point of view:
    - Translation parallax: background moves opposite to viewer motion.
    - Perspective tilt/keystone: background plane rotates with viewer yaw & pitch.
    """
    out_h, out_w = target_shape[:2]
    if bg_image is None or bg_image.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Scale background to safely cover target with room for parallax padding
    pad_factor = 1.25
    scaled_w = int(out_w * pad_factor)
    scaled_h = int(out_h * pad_factor)
    bg_resized = cv2.resize(bg_image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    # Compute 4 source corner coordinates on the background
    # Parallax translation shifts the crop window
    max_shift_x = (scaled_w - out_w) * 0.45
    max_shift_y = (scaled_h - out_h) * 0.45
    shift_x = -dx * max_shift_x
    shift_y = -dy * max_shift_y

    base_x = (scaled_w - out_w) * 0.5 + shift_x
    base_y = (scaled_h - out_h) * 0.5 + shift_y

    src_corners = np.array(
        [
            [base_x, base_y],
            [base_x + out_w, base_y],
            [base_x + out_w, base_y + out_h],
            [base_x, base_y + out_h],
        ],
        dtype=np.float32,
    )

    # Compute 4 destination corners with 3D perspective keystone / tilt
    # Yaw causes horizontal trapezoid (near side larger)
    # Pitch causes vertical trapezoid
    perspective_strength_x = np.radians(np.clip(yaw, -40.0, 40.0)) * 0.35
    perspective_strength_y = np.radians(np.clip(pitch, -35.0, 35.0)) * 0.35

    delta_top = out_w * float(perspective_strength_x) * 0.5
    delta_side = out_h * float(perspective_strength_y) * 0.5

    dst_corners = np.array(
        [
            [delta_top, delta_side],
            [out_w + delta_top, -delta_side],
            [out_w - delta_top, out_h - delta_side],
            [-delta_top, out_h + delta_side],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
    warped_bg = cv2.warpPerspective(
        bg_resized,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped_bg


def predict_matte(frame: Frame) -> np.ndarray:
    """Predict soft person segmentation matte in range [0.0, 1.0]."""
    sess = get_segmentation_session()
    h, w = frame.shape[:2]
    if sess is None:
        # Fallback heuristic: ellipse centered on frame
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (w // 2, int(h * 0.55)), (int(w * 0.35), int(h * 0.45)), 0, 0, 360, 1.0, -1)
        return cv2.GaussianBlur(mask, (31, 31), 11)

    # MODNet accepts inputs divisible by 32
    infer_size = 512
    img = cv2.resize(frame, (infer_size, infer_size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

    try:
        input_name = sess.get_inputs()[0].name
        matte = sess.run(None, {input_name: img})[0]
        matte = np.squeeze(matte)
        matte = np.clip(matte, 0.0, 1.0)
        # Resize matte back to original dimensions
        matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)
        # Slight blur to avoid harsh aliasing around hair/clothing
        matte = cv2.GaussianBlur(matte, (5, 5), 0)
        return matte
    except Exception as exc:
        print(f"[{NAME}] Segmentation error: {exc}")
        mask = np.ones((h, w), dtype=np.float32)
        return mask


def swap_background(frame: Frame, target_face: Optional[Face] = None) -> Frame:
    """Perform background swap with 3D perspective warping on a frame."""
    bg_image = get_background_image()
    if bg_image is None or bg_image.size == 0:
        return frame

    h, w = frame.shape[:2]

    # 1. Calculate viewer's point of view
    if target_face is None:
        target_face = get_one_face(frame)
    dx, dy, yaw, pitch, roll = compute_viewer_pov(target_face, (h, w))

    # 2. Render 3D perspective background
    warped_bg = render_3d_perspective_background(
        bg_image, (h, w), dx, dy, yaw, pitch, roll
    )

    # 3. Predict foreground person matte
    matte = predict_matte(frame)
    alpha = matte[:, :, np.newaxis]

    # 4. Alpha composite: Foreground * Alpha + Background * (1 - Alpha)
    blended = frame.astype(np.float32) * alpha + warped_bg.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def process_frame(
    source_face: Optional[Face], temp_frame: Frame, target_face: Optional[Face] = None
) -> Frame:
    """Process a single frame for standard pipeline."""
    if not getattr(modules.globals, "background_swap", False):
        return temp_frame
    return swap_background(temp_frame, target_face=target_face)


def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    """Process a single frame for v2/live pipeline."""
    if not getattr(modules.globals, "background_swap", False):
        return temp_frame
    return swap_background(temp_frame)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Process a single target image file."""
    frame = imread_unicode(target_path)
    if frame is None:
        update_status(f"Could not read image: {target_path}", NAME)
        return
    result = swap_background(frame)
    imwrite_unicode(output_path, result)


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """Process frame sequence for video."""
    for path in temp_frame_paths:
        frame = imread_unicode(path)
        if frame is None:
            if progress:
                progress.update(1)
            continue
        res = swap_background(frame)
        imwrite_unicode(path, res)
        if progress:
            progress.update(1)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Process video frames using core framework."""
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames
    )
