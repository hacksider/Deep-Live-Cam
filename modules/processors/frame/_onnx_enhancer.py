"""Shared ONNX-based face enhancement utilities for GPEN-BFR models.

Provides session creation (reusing CoreML EP config from face_swapper),
pre/post processing, and the core enhance-face-via-ONNX pipeline.
"""

import os
import platform
import threading
from typing import Any, Optional

import cv2
import numpy as np
import onnxruntime

import modules.globals

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Limit concurrent ONNX calls to avoid VRAM exhaustion on multi-face frames
_SEMAPHORE_COUNT = min(max(1, (os.cpu_count() or 1)), 8)
THREAD_SEMAPHORE = threading.Semaphore(_SEMAPHORE_COUNT)


def create_onnx_session(model_path: str) -> onnxruntime.InferenceSession:
    """Create an ONNX Runtime session with provider config matching face_swapper."""
    providers_config = []
    for p in modules.globals.execution_providers:
        if p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
            coreml_cache_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "deep-live-cam", "coreml"
            )
            os.makedirs(coreml_cache_dir, exist_ok=True)
            providers_config.append((
                "CoreMLExecutionProvider",
                {
                    "ModelFormat": "MLProgram",
                    "MLComputeUnits": "ALL",
                    "SpecializationStrategy": "FastPrediction",
                    "AllowLowPrecisionAccumulationOnGPU": 1,
                    "EnableOnSubgraphs": 1,
                    "RequireStaticShapes": 1,
                    "MaximumCacheSize": 1024 * 1024 * 512,
                    "ModelCacheDirectory": coreml_cache_dir,
                },
            ))
        else:
            providers_config.append(p)

    session = onnxruntime.InferenceSession(model_path, providers=providers_config)
    return session


def warmup_session(session: onnxruntime.InferenceSession) -> None:
    """Run a dummy inference pass to trigger JIT / CoreML compile caching."""
    try:
        input_feed = {
            inp.name: np.zeros(
                [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape],
                dtype=np.float32,
            )
            for inp in session.get_inputs()
        }
        session.run(None, input_feed)
    except Exception as e:
        print(f"ONNX enhancer warmup skipped (non-fatal): {e}")


def preprocess_face(face_img: np.ndarray, input_size: int) -> np.ndarray:
    """Resize, normalize, and convert a BGR face crop to ONNX input blob.

    GPEN-BFR expects [1, 3, H, W] float32 in RGB, normalized to [-1, 1].
    """
    resized = cv2.resize(face_img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    # BGR -> RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize to [-1, 1]
    blob = rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
    # HWC -> CHW -> NCHW
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob


def postprocess_face(output: np.ndarray) -> np.ndarray:
    """Convert ONNX output [1, 3, H, W] float32 back to BGR uint8 image."""
    # NCHW -> CHW -> HWC
    img = output[0].transpose(1, 2, 0)
    # [-1, 1] -> [0, 255]
    img = ((img + 1.0) / 2.0 * 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # RGB -> BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _get_face_affine(face: Any, input_size: int):
    """Compute affine transform to align a face to GPEN input space.

    Uses the 5-point landmarks (eyes, nose, mouth corners) from InsightFace
    to warp the face into a canonical position expected by restoration models.

    Returns (M, inv_M) — forward and inverse affine matrices.
    """
    # Standard 5-point template for face restoration (FFHQ-aligned)
    # Scaled to input_size
    template = np.array([
        [0.31556875, 0.4615741],
        [0.68262291, 0.4615741],
        [0.50009375, 0.6405054],
        [0.34947187, 0.8246919],
        [0.65343645, 0.8246919],
    ], dtype=np.float32) * input_size

    landmarks = None
    if hasattr(face, "kps") and face.kps is not None:
        landmarks = face.kps.astype(np.float32)
    elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
        # Extract 5 key points from 106 landmarks
        lm106 = face.landmark_2d_106
        landmarks = np.array([
            lm106[38],  # left eye
            lm106[88],  # right eye
            lm106[86],  # nose tip
            lm106[52],  # left mouth
            lm106[61],  # right mouth
        ], dtype=np.float32)

    if landmarks is None or len(landmarks) < 5:
        return None, None

    M = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)[0]
    if M is None:
        return None, None
    inv_M = cv2.invertAffineTransform(M)
    return M, inv_M


def enhance_face_onnx(
    frame: np.ndarray,
    face: Any,
    session: onnxruntime.InferenceSession,
    input_size: int,
) -> np.ndarray:
    """Enhance a single face in the frame using an ONNX face restoration model.

    1. Warp face to canonical position using affine transform
    2. Run ONNX inference
    3. Warp enhanced face back and blend into original frame
    """
    M, inv_M = _get_face_affine(face, input_size)
    if M is None:
        return frame

    # Warp face out of the frame
    face_crop = cv2.warpAffine(
        frame, M, (input_size, input_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    # Preprocess and run inference
    blob = preprocess_face(face_crop, input_size)
    with THREAD_SEMAPHORE:
        output = session.run(None, {session.get_inputs()[0].name: blob})[0]
    enhanced = postprocess_face(output)

    # Create mask for blending (feathered edges)
    mask = np.ones((input_size, input_size), dtype=np.float32)
    border = max(1, input_size // 16)
    mask[:border, :] = np.linspace(0, 1, border)[:, np.newaxis]
    mask[-border:, :] = np.linspace(1, 0, border)[:, np.newaxis]
    mask[:, :border] = np.minimum(mask[:, :border], np.linspace(0, 1, border)[np.newaxis, :])
    mask[:, -border:] = np.minimum(mask[:, -border:], np.linspace(1, 0, border)[np.newaxis, :])

    # Warp enhanced face and mask back to original frame space
    h, w = frame.shape[:2]
    warped_enhanced = cv2.warpAffine(
        enhanced, inv_M, (w, h),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask, inv_M, (w, h),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    # Blend: enhanced face where mask > 0, original elsewhere
    mask_3ch = warped_mask[:, :, np.newaxis]
    result = (warped_enhanced.astype(np.float32) * mask_3ch +
              frame.astype(np.float32) * (1.0 - mask_3ch))
    return np.clip(result, 0, 255).astype(np.uint8)
