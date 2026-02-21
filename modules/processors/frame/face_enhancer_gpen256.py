"""GPEN-BFR-256 face enhancer — ONNX-based face restoration at 256×256."""

from typing import Any, List
import os

import cv2
import numpy as np

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.model_loader import ModelHolder
from modules.paths import MODELS_DIR
from modules.typing import Frame, Face
from modules.utilities import conditional_download, is_image, is_video
from modules.processors.frame._onnx_enhancer import (
    create_onnx_session,
    warmup_session,
    enhance_face_onnx,
)

NAME = "DLC.FACE-ENHANCER-GPEN256"
INPUT_SIZE = 256
MODEL_URL = "https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/GPEN-BFR/GPEN-BFR-256.onnx"
MODEL_FILE = "GPEN-BFR-256.onnx"

_model = ModelHolder()


def _load_model() -> Any:
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    print(f"{NAME}: Loading ONNX model from {model_path}")
    session = create_onnx_session(model_path)
    print(f"{NAME}: Model loaded successfully.")
    return session


def _warmup(session: Any) -> None:
    warmup_session(session)
    print(f"{NAME}: Warmup inference complete.")


def get_enhancer() -> Any:
    return _model.get(loader_fn=_load_model, warmup_fn=_warmup)


def pre_check() -> bool:
    conditional_download(MODELS_DIR, [MODEL_URL])
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


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


def process_frame(source_face: Face | None, temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face is None:
        return temp_frame
    return enhance_face(temp_frame, target_face)


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    modules.processors.frame.core.process_frames_io(
        temp_frame_paths,
        process_fn=lambda frame: process_frame(None, frame),
        progress=progress,
    )


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    cv2.imwrite(output_path, result_frame)
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)


def process_frame_v2(temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame, target_face)
    return temp_frame
