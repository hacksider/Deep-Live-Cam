"""GPEN-BFR-512 face enhancer - ONNX-based face restoration at 512x512."""

from typing import Any, List
import os
import threading

import modules.globals
import modules.processors.frame.core
from modules import imread_unicode, imwrite_unicode
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
from modules.utilities import conditional_download, is_image, is_video
from modules.processors.frame._onnx_enhancer import (
    create_onnx_session,
    warmup_session,
    enhance_face_onnx,
)

NAME = "DLC.FACE-ENHANCER-GPEN512"
INPUT_SIZE = 512
MODEL_URL = "https://huggingface.co/martintomov/comfy/resolve/6644701b147beb68645be82ff78e4fd0eddb3927/facerestore_models/GPEN-BFR-512.onnx"
MODEL_FILE = "GPEN-BFR-512.onnx"
MIN_MODEL_BYTES = 100 * 1024 * 1024

ENHANCER = None
THREAD_LOCK = threading.Lock()

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
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
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_enhancer() -> Any:
    global ENHANCER
    with THREAD_LOCK:
        if ENHANCER is None:
            if not ensure_model():
                raise FileNotFoundError(f"Model file not found or incomplete: {model_path()}")
            print(f"{NAME}: Loading ONNX model from {model_path()}")
            ENHANCER = create_onnx_session(model_path())
            warmup_session(ENHANCER)
            print(f"{NAME}: Model loaded successfully.")
    return ENHANCER


def enhance_face(temp_frame: Frame, face: Face) -> Frame:
    try:
        session = get_enhancer()
    except Exception as e:
        print(f"{NAME}: {e}")
        return temp_frame
    try:
        return enhance_face_onnx(temp_frame, face, session, INPUT_SIZE)
    except Exception as e:
        print(f"{NAME}: Error during face enhancement: {e}")
        return temp_frame


def process_frame(source_face: Face | None, temp_frame: Frame, detected_faces=None) -> Frame:
    if detected_faces:
        target_face = detected_faces[0]
    else:
        target_face = get_one_face(temp_frame)
    if target_face is None:
        return temp_frame
    return enhance_face(temp_frame, target_face)


def process_frame_v2(temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame, target_face)
    return temp_frame


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = imread_unicode(temp_frame_path)
        if temp_frame is None:
            if progress:
                progress.update(1)
            continue
        result = process_frame(None, temp_frame)
        imwrite_unicode(temp_frame_path, result)
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
