# modules/processors/frame/face_sorter.py
from typing import Any, List
import cv2
import os
import shutil
import threading

import modules.globals
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
from modules.utilities import is_image, is_video

NAME = "DLC.FACE-SORTER"
THREAD_LOCK = threading.Lock()


def pre_check() -> bool:
    # No external models needed
    return True


def pre_start() -> bool:
    # Accept an image or a video as target (we'll intentionally ignore videos later).
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def _ensure_dest_dirs(base_dir: str):
    with_faces = os.path.join(base_dir, "with_faces")
    without_faces = os.path.join(base_dir, "without_faces")
    os.makedirs(with_faces, exist_ok=True)
    os.makedirs(without_faces, exist_ok=True)
    return with_faces, without_faces


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    if not modules.globals.fp_ui.get("face_sorter", False):
        return temp_frame
    # Add your face sorting logic here if desired
    return temp_frame

def _ensure_unprocessed_dir(base_dir: str):
    unprocessed_dir = os.path.join(base_dir, "unprocessed")
    os.makedirs(unprocessed_dir, exist_ok=True)
    return unprocessed_dir


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    if not modules.globals.fp_ui.get("face_sorter", False):
        return

    if not temp_frame_paths:
        return

    base_dir = os.path.dirname(temp_frame_paths[0])
    unprocessed_dir = _ensure_unprocessed_dir(base_dir)

    for p in temp_frame_paths:
        try:
            img = cv2.imread(p)
            if not img.any():
                print(f"[{NAME}] Skipping empty frame: {p}")
                continue

            has_face = bool(get_one_face(img))
            if not has_face:
                shutil.move(p, os.path.join(unprocessed_dir, os.path.basename(p)))
                print(f"[{NAME}] No face found — moved to unprocessed: {p}")
                continue  # Skip further processing

            print(f"[{NAME}] Face found: {p}")

        except Exception as e:
            print(f"[{NAME}] error processing {p}: {e}")

        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> bool:
    if not modules.globals.fp_ui.get("face_sorter", False):
        return True  # Keep processing

    target_frame = cv2.imread(target_path)
    if target_frame is None:
        print(f"[{NAME}] Failed to read {target_path}")
        return False  # Stop processing

    has_face = bool(get_one_face(target_frame))

    # Use original target path location for unprocessed
    base_dir = os.path.dirname(modules.globals.target_path) or os.getcwd()
    unprocessed_dir = os.path.join(base_dir, "unprocessed")
    os.makedirs(unprocessed_dir, exist_ok=True)

    if not has_face:
        shutil.move(modules.globals.target_path, os.path.join(unprocessed_dir, os.path.basename(modules.globals.target_path)))
        print(f"[{NAME}] No face detected. Moved original file to {unprocessed_dir}")
        return False  # Stop further processing

    return True  # Continue processing

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not modules.globals.fp_ui.get("face_sorter", False):
        return

    update_status("Face sorter ignores videos (skipping).", NAME)


def process_frame_v2(temp_frame: Frame) -> Frame:
    return temp_frame

