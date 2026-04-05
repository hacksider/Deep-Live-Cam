# --- START OF FILE face_enhancer.py ---
# Uses ONNX Runtime for GFPGAN face enhancement (no torch/gfpgan dependency)

from typing import Any, List
import cv2
import threading
import numpy as np
import os

import onnxruntime

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces
from modules.typing import Frame, Face
from modules.utilities import (
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

# Standard FFHQ 5-point face template for 512x512 resolution
# Points: left_eye, right_eye, nose, left_mouth, right_mouth
FFHQ_TEMPLATE_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19366],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


def pre_check() -> bool:
    model_path = os.path.join(models_dir, "gfpgan-1024.onnx")
    if not os.path.exists(model_path):
        update_status(
            f"GFPGAN ONNX model not found at {model_path}. "
            "Please place gfpgan-1024.onnx in the models folder.",
            NAME,
        )
        return False
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_face_enhancer() -> onnxruntime.InferenceSession:
    """
    Initializes and returns the GFPGAN ONNX Runtime inference session,
    using the execution providers configured in modules.globals.
    """
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = os.path.join(models_dir, "gfpgan-1024.onnx")

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"{NAME}: Model not found at {model_path}"
                )

            try:
                providers = modules.globals.execution_providers

                session_options = onnxruntime.SessionOptions()
                session_options.graph_optimization_level = (
                    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                )

                FACE_ENHANCER = onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_options,
                    providers=providers,
                )

                input_info = FACE_ENHANCER.get_inputs()[0]
                output_info = FACE_ENHANCER.get_outputs()[0]
                active_providers = FACE_ENHANCER.get_providers()
                print(
                    f"{NAME}: GFPGAN ONNX model loaded successfully."
                )
                print(
                    f"{NAME}: Input: {input_info.name}, "
                    f"shape: {input_info.shape}, type: {input_info.type}"
                )
                print(
                    f"{NAME}: Output: {output_info.name}, "
                    f"shape: {output_info.shape}, type: {output_info.type}"
                )
                print(f"{NAME}: Active providers: {active_providers}")

            except Exception as e:
                print(f"{NAME}: Error loading GFPGAN ONNX model: {e}")
                FACE_ENHANCER = None
                raise RuntimeError(
                    f"{NAME}: Failed to load GFPGAN ONNX model: {e}"
                )

    if FACE_ENHANCER is None:
        raise RuntimeError(
            f"{NAME}: Failed to initialize GFPGAN ONNX session. Check logs."
        )

    return FACE_ENHANCER


def _align_face(
    frame: Frame, landmarks_5: np.ndarray, output_size: int
) -> tuple:
    """
    Align and crop a face from the frame using 5-point landmarks and the
    standard FFHQ template.

    Returns:
        (aligned_face, affine_matrix) or (None, None) on failure.
    """
    # Scale the 512-base template to the desired output size
    scale = output_size / 512.0
    template = FFHQ_TEMPLATE_512 * scale

    # Estimate a similarity transform (4 DOF: rotation, scale, tx, ty)
    affine_matrix, _ = cv2.estimateAffinePartial2D(
        landmarks_5, template, method=cv2.LMEDS
    )
    if affine_matrix is None:
        return None, None

    # Warp the face to the aligned position
    aligned_face = cv2.warpAffine(
        frame,
        affine_matrix,
        (output_size, output_size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(135, 133, 132),
    )

    return aligned_face, affine_matrix


def _paste_back(
    frame: Frame,
    enhanced_face: np.ndarray,
    affine_matrix: np.ndarray,
    output_size: int,
) -> Frame:
    """
    Paste an enhanced (aligned) face back onto the original frame using the
    inverse affine transform with feathered-edge blending.
    """
    h, w = frame.shape[:2]

    # Inverse the affine warp
    inv_matrix = cv2.invertAffineTransform(affine_matrix)
    inv_restored = cv2.warpAffine(
        enhanced_face,
        inv_matrix,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Build a soft feathered mask in aligned space for edge blending
    face_mask = np.ones((output_size, output_size), dtype=np.float32)

    # Feather the border (5 % of the size on each edge)
    border = max(1, int(output_size * 0.05))
    ramp_up = np.linspace(0.0, 1.0, border, dtype=np.float32)
    ramp_down = np.linspace(1.0, 0.0, border, dtype=np.float32)

    # Top / bottom rows
    face_mask[:border, :] *= ramp_up[:, None]
    face_mask[-border:, :] *= ramp_down[:, None]
    # Left / right columns
    face_mask[:, :border] *= ramp_up[None, :]
    face_mask[:, -border:] *= ramp_down[None, :]

    # Expand to 3-channel
    face_mask_3c = np.stack([face_mask] * 3, axis=-1)

    # Warp mask back to original frame space
    inv_mask = cv2.warpAffine(
        face_mask_3c,
        inv_matrix,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    inv_mask = np.clip(inv_mask, 0.0, 1.0)

    # Alpha-blend
    result = (
        frame.astype(np.float32) * (1.0 - inv_mask)
        + inv_restored.astype(np.float32) * inv_mask
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def _preprocess_face(aligned_face: np.ndarray) -> np.ndarray:
    """
    Convert an aligned BGR uint8 face image to the ONNX model input tensor.
    Format: NCHW float32, normalised to [-1, 1].
    """
    # BGR -> RGB
    rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32)
    # [0, 255] -> [0, 1] -> [-1, 1]
    rgb = rgb / 255.0
    rgb = (rgb - 0.5) / 0.5
    # HWC -> CHW, add batch dim
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0)  # shape: (1, 3, H, W)


def _postprocess_face(output: np.ndarray) -> np.ndarray:
    """
    Convert the ONNX model output tensor back to a BGR uint8 image.
    Expects input in NCHW format with values in [-1, 1].
    """
    face = np.squeeze(output)  # remove batch dim -> (3, H, W)
    face = np.transpose(face, (1, 2, 0))  # CHW -> HWC
    # [-1, 1] -> [0, 1] -> [0, 255]
    face = (face + 1.0) / 2.0
    face = np.clip(face * 255.0, 0, 255).astype(np.uint8)
    # RGB -> BGR
    return cv2.cvtColor(face, cv2.COLOR_RGB2BGR)


def enhance_face(temp_frame: Frame) -> Frame:
    """Enhances all faces in a frame using the GFPGAN ONNX model."""
    session = get_face_enhancer()

    # Determine model input resolution from the session metadata
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape  # e.g. [1, 3, 512, 512]
    # Safely extract input size (handle dynamic / symbolic dimensions)
    try:
        align_size = int(input_shape[2])
        if align_size <= 0:
            align_size = 512
    except (ValueError, TypeError, IndexError):
        align_size = 512

    # Detect faces using InsightFace (already a project dependency)
    faces = get_many_faces(temp_frame)
    if not faces:
        return temp_frame

    result_frame = temp_frame.copy()

    for face in faces:
        # Need the 5-point key-points for alignment
        if not hasattr(face, "kps") or face.kps is None:
            continue

        landmarks_5 = face.kps.astype(np.float32)
        if landmarks_5.shape[0] < 5:
            continue

        # Align / crop the face at the model's INPUT resolution
        aligned_face, affine_matrix = _align_face(
            temp_frame, landmarks_5, output_size=align_size
        )
        if aligned_face is None or affine_matrix is None:
            continue

        try:
            with THREAD_SEMAPHORE:
                input_tensor = _preprocess_face(aligned_face)
                output_tensor = session.run(None, {input_name: input_tensor})[0]
                enhanced_bgr = _postprocess_face(output_tensor)

            # The model may output at a different resolution than its input
            # (e.g. input 512x512 → output 1024x1024).  Resize the enhanced
            # face back to the alignment size so the inverse affine maps
            # correctly.
            eh, ew = enhanced_bgr.shape[:2]
            if eh != align_size or ew != align_size:
                enhanced_bgr = cv2.resize(
                    enhanced_bgr,
                    (align_size, align_size),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            # Paste enhanced face back onto the frame
            result_frame = _paste_back(
                result_frame, enhanced_bgr, affine_matrix, output_size=align_size
            )
        except Exception as e:
            print(f"{NAME}: Error enhancing a face: {e}")
            continue

    return result_frame


def process_frame(source_face: Face | None, temp_frame: Frame) -> Frame:
    """Processes a frame: enhances face if detected."""
    temp_frame = enhance_face(temp_frame)
    return temp_frame


def process_frame_v2(temp_frame: Frame) -> Frame:
    """Processes a frame without source face (used by live webcam preview)."""
    return enhance_face(temp_frame)


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """Processes multiple frames from file paths."""
    for temp_frame_path in temp_frame_paths:
        if not os.path.exists(temp_frame_path):
            print(
                f"{NAME}: Warning: Frame path not found {temp_frame_path}, skipping."
            )
            if progress:
                progress.update(1)
            continue

        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            print(
                f"{NAME}: Warning: Failed to read frame {temp_frame_path}, skipping."
            )
            if progress:
                progress.update(1)
            continue

        result_frame = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result_frame)
        if progress:
            progress.update(1)


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
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames
    )


# --- END OF FILE face_enhancer.py ---
