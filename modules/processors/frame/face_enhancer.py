"""ONNX Runtime GFPGAN face enhancer."""

from typing import Any, List
import os
import threading

import cv2
import numpy as np
import onnxruntime

import modules.globals
import modules.processors.frame.core
from modules import imread_unicode, imwrite_unicode
from modules.core import update_status
from modules.face_analyser import get_many_faces
from modules.typing import Frame, Face
from modules.utilities import conditional_download, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-ENHANCER"
MODEL_URL = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/gfpgan-1024.onnx"
MODEL_FILE = "gfpgan-1024.onnx"
MIN_MODEL_BYTES = 100 * 1024 * 1024

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

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


def model_path() -> str:
    return os.path.join(models_dir, MODEL_FILE)


def model_ready(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) >= MIN_MODEL_BYTES


def ensure_model() -> bool:
    path = model_path()
    if model_ready(path):
        return True
    if os.path.exists(path):
        os.remove(path)
    update_status(f"Downloading {MODEL_FILE}...", NAME)
    conditional_download(models_dir, [MODEL_URL])
    if not model_ready(path):
        update_status(f"{MODEL_FILE} download failed or is incomplete at {path}", NAME)
        return False
    return True


def pre_check() -> bool:
    return ensure_model()


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_face_enhancer() -> onnxruntime.InferenceSession:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            if not ensure_model():
                raise FileNotFoundError(f"{NAME}: Model not ready at {model_path()}")
            try:
                from modules.processors.frame._onnx_enhancer import create_onnx_session

                FACE_ENHANCER = create_onnx_session(model_path())
                input_info = FACE_ENHANCER.get_inputs()[0]
                output_info = FACE_ENHANCER.get_outputs()[0]
                print(f"{NAME}: GFPGAN ONNX model loaded successfully.")
                print(
                    f"{NAME}: Input: {input_info.name}, "
                    f"shape: {input_info.shape}, type: {input_info.type}"
                )
                print(
                    f"{NAME}: Output: {output_info.name}, "
                    f"shape: {output_info.shape}, type: {output_info.type}"
                )
                print(f"{NAME}: Active providers: {FACE_ENHANCER.get_providers()}")
            except Exception as exc:
                FACE_ENHANCER = None
                raise RuntimeError(f"{NAME}: Failed to load GFPGAN ONNX model: {exc}") from exc
    return FACE_ENHANCER


def _align_face(frame: Frame, landmarks_5: np.ndarray, output_size: int) -> tuple[Any, Any]:
    template = FFHQ_TEMPLATE_512 * (output_size / 512.0)
    affine_matrix, _ = cv2.estimateAffinePartial2D(
        landmarks_5, template, method=cv2.LMEDS
    )
    if affine_matrix is None:
        return None, None
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
    h, w = frame.shape[:2]
    inv_matrix = cv2.invertAffineTransform(affine_matrix)
    restored = cv2.warpAffine(
        enhanced_face, inv_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT
    )

    mask = np.ones((output_size, output_size), dtype=np.float32)
    border = max(1, int(output_size * 0.05))
    ramp_up = np.linspace(0.0, 1.0, border, dtype=np.float32)
    ramp_down = np.linspace(1.0, 0.0, border, dtype=np.float32)
    mask[:border, :] *= ramp_up[:, None]
    mask[-border:, :] *= ramp_down[:, None]
    mask[:, :border] *= ramp_up[None, :]
    mask[:, -border:] *= ramp_down[None, :]
    warped_mask = cv2.warpAffine(mask, inv_matrix, (w, h), borderValue=0)
    alpha = warped_mask[:, :, None]
    blended = restored.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _preprocess_face(aligned_face: np.ndarray) -> np.ndarray:
    rgb = aligned_face[:, :, ::-1]
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    chw *= 1.0 / 127.5
    chw -= 1.0
    return chw[np.newaxis, ...]


def _postprocess_face(output: np.ndarray) -> np.ndarray:
    face = output[0]
    face = (face + 1.0) * 127.5
    np.clip(face, 0, 255, out=face)
    face = face.astype(np.uint8).transpose(1, 2, 0)
    return face[:, :, ::-1].copy()


def enhance_face(temp_frame: Frame, detected_faces=None) -> Frame:
    session = get_face_enhancer()
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    try:
        align_size = int(input_info.shape[2])
        if align_size <= 0:
            align_size = 1024
    except (ValueError, TypeError, IndexError):
        align_size = 1024

    faces = detected_faces if detected_faces is not None else get_many_faces(temp_frame)
    if not faces:
        return temp_frame

    many_faces_mode = getattr(modules.globals, "many_faces", False)
    for face in faces:
        if not hasattr(face, "kps") or face.kps is None:
            continue
        landmarks_5 = face.kps.astype(np.float32)
        if landmarks_5.shape[0] < 5:
            continue
        aligned_face, affine_matrix = _align_face(temp_frame, landmarks_5, align_size)
        if aligned_face is None or affine_matrix is None:
            continue
        try:
            with THREAD_SEMAPHORE:
                from modules.processors.frame._onnx_enhancer import run_inference

                input_tensor = _preprocess_face(aligned_face)
                output_tensor = run_inference(session, input_name, input_tensor)
                enhanced_bgr = _postprocess_face(output_tensor)
            if enhanced_bgr.shape[:2] != (align_size, align_size):
                enhanced_bgr = cv2.resize(
                    enhanced_bgr, (align_size, align_size), interpolation=cv2.INTER_LANCZOS4
                )
            temp_frame = _paste_back(temp_frame, enhanced_bgr, affine_matrix, align_size)
        except Exception as exc:
            print(f"{NAME}: Error enhancing a face: {exc}")
            continue
        if not many_faces_mode:
            break
    return temp_frame


def process_frame(source_face: Face | None, temp_frame: Frame, detected_faces=None) -> Frame:
    return enhance_face(temp_frame, detected_faces=detected_faces)


def process_frame_v2(temp_frame: Frame, detected_faces=None) -> Frame:
    return enhance_face(temp_frame, detected_faces=detected_faces)


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    for temp_frame_path in temp_frame_paths:
        if not os.path.exists(temp_frame_path):
            if progress:
                progress.update(1)
            continue
        temp_frame = imread_unicode(temp_frame_path)
        if temp_frame is None:
            if progress:
                progress.update(1)
            continue
        result_frame = process_frame(None, temp_frame)
        imwrite_unicode(temp_frame_path, result_frame)
        if progress:
            progress.update(1)


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    target_frame = imread_unicode(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    imwrite_unicode(output_path, result_frame)
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
