from typing import Any, List, Tuple, Optional
import cv2
import insightface
import threading
import numpy as np

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, default_source_face
from modules.typing import Face, Frame
from modules.utilities import (
    conditional_download,
    resolve_relative_path,
    is_image,
    is_video,
)
from modules.cluster_analysis import find_closest_centroid

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-SWAPPER"
BLUR_AMOUNT = 12


def pre_check() -> bool:
    download_directory_path = resolve_relative_path("../models")
    conditional_download(
        download_directory_path,
        [
            "https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx"
        ],
    )
    return True


def pre_start() -> bool:
    if not modules.globals.map_faces and not is_image(modules.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False
    elif not modules.globals.map_faces and not get_one_face(
        cv2.imread(modules.globals.source_path)
    ):
        update_status("No face in source path detected.", NAME)
        return False
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path("../models/inswapper_128_fp16.onnx")
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path, providers=modules.globals.execution_providers
            )
    return FACE_SWAPPER


def create_face_masks(
    face: Face, frame: Frame
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], np.ndarray]:
    """Create both face and mouth masks in one pass."""
    face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Create full face mask
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.fillConvexPoly(face_mask, hull, 255)

        # Create mouth mask
        lower_lip = landmarks[90:96]  # Lower lip points
        lower_lip_polygon = cv2.convexHull(lower_lip.astype(np.int32))
        cv2.fillConvexPoly(mouth_mask, lower_lip_polygon, 255)

        # Get mouth bounding box
        x, y, w, h = cv2.boundingRect(lower_lip_polygon)
        mouth_cutout = frame[y : y + h, x : x + w].copy()

        return face_mask, mouth_cutout, (x, y, w, h), lower_lip_polygon

    return None, None, None, None


def apply_mouth_area(
    frame: Frame,
    mouth_cutout: np.ndarray,
    mouth_box: Tuple[int, int, int, int],
    face_mask: np.ndarray,
    lower_lip_polygon: Optional[np.ndarray],
) -> Frame:
    """Apply the original mouth area back to the face-swapped frame."""
    if mouth_cutout is None or mouth_box is None:
        return frame

    x, y, w, h = mouth_box

    # Create a blurred version of the mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if lower_lip_polygon is not None:
        cv2.fillConvexPoly(mask, lower_lip_polygon, 255)
    else:
        mask[y : y + h, x : x + w] = 255

    # Blur the mask
    blurred_mask = cv2.GaussianBlur(mask, (BLUR_AMOUNT * 2 + 1, BLUR_AMOUNT * 2 + 1), 0)
    blurred_mask = blurred_mask / 255.0

    # Create 3-channel mask
    blurred_mask_3channel = np.repeat(blurred_mask[:, :, np.newaxis], 3, axis=2)

    # Blend the original mouth area with the swapped face
    frame_copy = frame.copy()
    frame_copy[y : y + h, x : x + w] = mouth_cutout

    # Combine using the blurred mask
    result = (
        frame_copy * blurred_mask_3channel + frame * (1 - blurred_mask_3channel)
    ).astype(np.uint8)

    return result


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    face_swapper = get_face_swapper()
    # Apply the face swap
    swapped_frame = face_swapper.get(
        temp_frame, target_face, source_face, paste_back=True
    )

    if modules.globals.mouth_mask:
        # Create masks
        face_mask = create_face_mask(target_face, temp_frame)
        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
            create_lower_mouth_mask(target_face, temp_frame)
        )

        if mouth_mask is not None:
            # Apply the mouth area preservation
            swapped_frame = apply_mouth_area(
                swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
            )

    return swapped_frame


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    # Ensure the frame is in RGB format if color correction is enabled
    if modules.globals.color_correction:
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)

    if modules.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)

    return temp_frame


def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    if is_image(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map in modules.globals.souce_target_map:
                target_face = map["target"]["face"]
                temp_frame = swap_face(source_face, target_face, temp_frame)
        elif not modules.globals.many_faces:
            for map in modules.globals.souce_target_map:
                if "source" in map:
                    source_face = map["source"]["face"]
                    target_face = map["target"]["face"]
                    temp_frame = swap_face(source_face, target_face, temp_frame)

    elif is_video(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map in modules.globals.souce_target_map:
                target_frame = [
                    f
                    for f in map["target_faces_in_frame"]
                    if f["location"] == temp_frame_path
                ]
                for frame in target_frame:
                    for target_face in frame["faces"]:
                        temp_frame = swap_face(source_face, target_face, temp_frame)
        elif not modules.globals.many_faces:
            for map in modules.globals.souce_target_map:
                if "source" in map:
                    target_frame = [
                        f
                        for f in map["target_faces_in_frame"]
                        if f["location"] == temp_frame_path
                    ]
                    source_face = map["source"]["face"]
                    for frame in target_frame:
                        for target_face in frame["faces"]:
                            temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        detected_faces = get_many_faces(temp_frame)
        if modules.globals.many_faces:
            if detected_faces:
                source_face = default_source_face()
                for target_face in detected_faces:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
        elif not modules.globals.many_faces:
            if detected_faces:
                if len(detected_faces) <= len(
                    modules.globals.simple_map["target_embeddings"]
                ):
                    for detected_face in detected_faces:
                        closest_centroid_index, _ = find_closest_centroid(
                            modules.globals.simple_map["target_embeddings"],
                            detected_face.normed_embedding,
                        )
                        temp_frame = swap_face(
                            modules.globals.simple_map["source_faces"][
                                closest_centroid_index
                            ],
                            detected_face,
                            temp_frame,
                        )
                else:
                    detected_faces_centroids = []
                    for face in detected_faces:
                        detected_faces_centroids.append(face.normed_embedding)
                    i = 0
                    for target_embedding in modules.globals.simple_map[
                        "target_embeddings"
                    ]:
                        closest_centroid_index, _ = find_closest_centroid(
                            detected_faces_centroids, target_embedding
                        )
                        temp_frame = swap_face(
                            modules.globals.simple_map["source_faces"][i],
                            detected_faces[closest_centroid_index],
                            temp_frame,
                        )
                        i += 1
    return temp_frame


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    if not modules.globals.map_faces:
        source_face = get_one_face(cv2.imread(source_path))
        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            try:
                result = process_frame(source_face, temp_frame)
                cv2.imwrite(temp_frame_path, result)
            except Exception as exception:
                print(exception)
                pass
            if progress:
                progress.update(1)
    else:
        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            try:
                result = process_frame_v2(temp_frame, temp_frame_path)
                cv2.imwrite(temp_frame_path, result)
            except Exception as exception:
                print(exception)
                pass
            if progress:
                progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    if not modules.globals.map_faces:
        source_face = get_one_face(cv2.imread(source_path))
        target_frame = cv2.imread(target_path)
        result = process_frame(source_face, target_frame)
        cv2.imwrite(output_path, result)
    else:
        if modules.globals.many_faces:
            update_status(
                "Many faces enabled. Using first source image. Progressing...", NAME
            )
        target_frame = cv2.imread(output_path)
        result = process_frame_v2(target_frame)
        cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if modules.globals.map_faces and modules.globals.many_faces:
        update_status(
            "Many faces enabled. Using first source image. Progressing...", NAME
        )
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames
    )
