from typing import Any, List
import cv2
import insightface
import threading
import numpy as np
import modules.globals
import logging
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, default_source_face
from modules.typing import Face, Frame
from modules.utilities import (
    conditional_download,
    is_image,
    is_video,
)
from modules.cluster_analysis import find_closest_centroid
import os

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-SWAPPER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def pre_check() -> bool:
    download_directory_path = models_dir
    model_url = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx"
    if "CUDAExecutionProvider" in modules.globals.execution_providers:
        model_url = "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"

    conditional_download(
        download_directory_path,
        [model_url],
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
            model_name = "inswapper_128.onnx"
            if "CUDAExecutionProvider" in modules.globals.execution_providers:
                model_name = "inswapper_128_fp16.onnx"
            model_path = os.path.join(models_dir, model_name)
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path, providers=modules.globals.execution_providers
            )
    return FACE_SWAPPER


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    face_swapper = get_face_swapper()

    # Simple face swap - maximum FPS
    swapped_frame = face_swapper.get(
        temp_frame, target_face, source_face, paste_back=True
    )

    if modules.globals.mouth_mask:
        # Create a mask for the target face
        face_mask = create_face_mask(target_face, temp_frame)

        # Create the mouth mask
        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
            create_lower_mouth_mask(target_face, temp_frame)
        )

        # Apply the mouth area
        swapped_frame = apply_mouth_area(
            swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
        )

        if modules.globals.show_mouth_mask_box:
            mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
            swapped_frame = draw_mouth_mask_visualization(
                swapped_frame, target_face, mouth_mask_data
            )

    return swapped_frame


# Simple face position smoothing for stability
_last_face_position = None
_position_smoothing = 0.7  # Higher = more stable, lower = more responsive

def swap_face_stable(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Ultra-fast face swap - maximum FPS priority"""
    # Skip all complex processing for maximum FPS
    face_swapper = get_face_swapper()
    swapped_frame = face_swapper.get(temp_frame, target_face, source_face, paste_back=True)
    
    # Skip all post-processing to maximize FPS
    return swapped_frame


def swap_face_ultra_fast(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Fast face swap with mouth mask support and forehead protection"""
    face_swapper = get_face_swapper()
    swapped_frame = face_swapper.get(temp_frame, target_face, source_face, paste_back=True)
    
    # Fix forehead hair issue - blend forehead area back to original
    swapped_frame = fix_forehead_hair_issue(swapped_frame, target_face, temp_frame)
    
    # Add mouth mask functionality back (only if enabled)
    if modules.globals.mouth_mask:
        # Create a mask for the target face
        face_mask = create_face_mask(target_face, temp_frame)

        # Create the mouth mask
        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
            create_lower_mouth_mask(target_face, temp_frame)
        )

        # Apply the mouth area
        swapped_frame = apply_mouth_area(
            swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
        )

        if modules.globals.show_mouth_mask_box:
            mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
            swapped_frame = draw_mouth_mask_visualization(
                swapped_frame, target_face, mouth_mask_data
            )
    
    return swapped_frame


def fix_forehead_hair_issue(swapped_frame: Frame, target_face: Face, original_frame: Frame) -> Frame:
    """Fix hair falling on forehead by blending forehead area back to original"""
    try:
        # Get face bounding box
        bbox = target_face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within frame bounds
        h, w = swapped_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return swapped_frame
        
        # Focus on forehead area (upper 35% of face)
        forehead_height = int((y2 - y1) * 0.35)
        forehead_y2 = y1 + forehead_height
        
        if forehead_y2 > y1:
            # Extract forehead regions
            swapped_forehead = swapped_frame[y1:forehead_y2, x1:x2]
            original_forehead = original_frame[y1:forehead_y2, x1:x2]
            
            # Create a soft blend mask for forehead area
            mask = np.ones(swapped_forehead.shape[:2], dtype=np.float32)
            
            # Apply strong Gaussian blur for very soft blending
            mask = cv2.GaussianBlur(mask, (31, 31), 10)
            mask = mask[:, :, np.newaxis]
            
            # Blend forehead areas (keep much more of original to preserve hair)
            blended_forehead = (swapped_forehead * 0.3 + original_forehead * 0.7).astype(np.uint8)
            
            # Apply the blended forehead back
            swapped_frame[y1:forehead_y2, x1:x2] = blended_forehead
        
        return swapped_frame
        
    except Exception:
        return swapped_frame


def improve_forehead_matching(swapped_frame: Frame, source_face: Face, target_face: Face, original_frame: Frame) -> Frame:
    """Create precise face mask - only swap core facial features (eyes, nose, cheeks, chin)"""
    try:
        # Get face landmarks for precise masking
        if hasattr(target_face, 'landmark_2d_106') and target_face.landmark_2d_106 is not None:
            landmarks = target_face.landmark_2d_106.astype(np.int32)
            
            # Create precise face mask excluding forehead and hair
            mask = create_precise_face_mask(landmarks, swapped_frame.shape[:2])
            
            if mask is not None:
                # Apply the precise mask
                mask_3d = mask[:, :, np.newaxis] / 255.0
                
                # Blend only the core facial features
                result = (swapped_frame * mask_3d + original_frame * (1 - mask_3d)).astype(np.uint8)
                return result
        
        # Fallback: use bounding box method but exclude forehead
        bbox = target_face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within frame bounds
        h, w = swapped_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return swapped_frame
        
        # Exclude forehead area (upper 25% of face) to avoid hair swapping
        forehead_height = int((y2 - y1) * 0.25)
        face_start_y = y1 + forehead_height
        
        if face_start_y < y2:
            # Only blend the lower face area (eyes, nose, cheeks, chin)
            swapped_face_area = swapped_frame[face_start_y:y2, x1:x2]
            original_face_area = original_frame[face_start_y:y2, x1:x2]
            
            # Create soft mask for the face area only
            mask = np.ones(swapped_face_area.shape[:2], dtype=np.float32)
            mask = cv2.GaussianBlur(mask, (15, 15), 5)
            mask = mask[:, :, np.newaxis]
            
            # Apply the face area back (keep original forehead/hair)
            swapped_frame[face_start_y:y2, x1:x2] = swapped_face_area
        
        return swapped_frame
        
    except Exception:
        return swapped_frame


def create_precise_face_mask(landmarks: np.ndarray, frame_shape: tuple) -> np.ndarray:
    """Create precise mask for core facial features only (exclude forehead and hair)"""
    try:
        mask = np.zeros(frame_shape, dtype=np.uint8)
        
        # For 106-point landmarks, use correct indices
        # Face contour (jawline) - points 0-32
        jaw_line = landmarks[0:33]
        
        # Eyes area - approximate indices for 106-point model
        left_eye_area = landmarks[33:42]   # Left eye region
        right_eye_area = landmarks[87:96]  # Right eye region
        
        # Eyebrows (start from eyebrow level, not forehead)
        left_eyebrow = landmarks[43:51]    # Left eyebrow
        right_eyebrow = landmarks[97:105]  # Right eyebrow
        
        # Create face contour that excludes forehead
        # Start from eyebrow level and go around the face
        face_contour_points = []
        
        # Add eyebrow points (this will be our "top" instead of forehead)
        face_contour_points.extend(left_eyebrow)
        face_contour_points.extend(right_eyebrow)
        
        # Add jawline points (bottom and sides of face)
        face_contour_points.extend(jaw_line)
        
        # Convert to numpy array
        face_contour_points = np.array(face_contour_points)
        
        # Create convex hull for the core face area (excluding forehead)
        hull = cv2.convexHull(face_contour_points)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Apply Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (21, 21), 7)
        
        return mask
        
    except Exception as e:
        print(f"Error creating precise face mask: {e}")
        return None


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    # Skip color correction for maximum FPS
    # if modules.globals.color_correction:
    #     temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)

    if modules.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                if source_face and target_face:
                    temp_frame = swap_face_ultra_fast(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face and source_face:
            temp_frame = swap_face_ultra_fast(source_face, target_face, temp_frame)
    return temp_frame


def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    if is_image(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map in modules.globals.source_target_map:
                target_face = map["target"]["face"]
                temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            for map in modules.globals.source_target_map:
                if "source" in map:
                    source_face = map["source"]["face"]
                    target_face = map["target"]["face"]
                    temp_frame = swap_face(source_face, target_face, temp_frame)

    elif is_video(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map in modules.globals.source_target_map:
                target_frame = [
                    f
                    for f in map["target_faces_in_frame"]
                    if f["location"] == temp_frame_path
                ]

                for frame in target_frame:
                    for target_face in frame["faces"]:
                        temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            for map in modules.globals.source_target_map:
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


def create_lower_mouth_mask(
    face: Face, frame: Frame
) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        lower_lip_order = [
            65,
            66,
            62,
            70,
            69,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            0,
            8,
            7,
            6,
            5,
            4,
            3,
            2,
            65,
        ]
        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.float32)

        center = np.mean(lower_lip_landmarks, axis=0)
        expansion_factor = 1 + modules.globals.mask_down_size
        expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center

        toplip_indices = [20, 0, 1, 2, 3, 4, 5]
        toplip_extension = modules.globals.mask_size * 0.5
        for idx in toplip_indices:
            direction = expanded_landmarks[idx] - center
            direction = direction / np.linalg.norm(direction)
            expanded_landmarks[idx] += direction * toplip_extension

        chin_indices = [11, 12, 13, 14, 15, 16]
        chin_extension = 2 * 0.2
        for idx in chin_indices:
            expanded_landmarks[idx][1] += (
                expanded_landmarks[idx][1] - center[1]
            ) * chin_extension

        expanded_landmarks = expanded_landmarks.astype(np.int32)

        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)

        padding = int((max_x - min_x) * 0.1)
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding)
        max_y = min(frame.shape[0], max_y + padding)

        if max_x <= min_x or max_y <= min_y:
            if (max_x - min_x) <= 1:
                max_x = min_x + 1
            if (max_y - min_y) <= 1:
                max_y = min_y + 1

        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        cv2.fillPoly(mask_roi, [expanded_landmarks - [min_x, min_y]], 255)
        # Improved smoothing for mouth mask
        mask_roi = cv2.GaussianBlur(mask_roi, (25, 25), 8)
        mask[min_y:max_y, min_x:max_x] = mask_roi
        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
        lower_lip_polygon = expanded_landmarks

    return mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon


def draw_mouth_mask_visualization(frame: Frame, face: Face, mouth_mask_data: tuple) -> Frame:
    landmarks = face.landmark_2d_106
    if landmarks is not None and mouth_mask_data is not None:
        mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon = mouth_mask_data
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(width, max_x), min(height, max_y)
        cv2.polylines(vis_frame, [lower_lip_polygon], True, (0, 255, 0), 2)
        cv2.putText(vis_frame, "Lower Mouth Mask", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return vis_frame
    return frame


def apply_mouth_area(frame: np.ndarray, mouth_cutout: np.ndarray, mouth_box: tuple, face_mask: np.ndarray, mouth_polygon: np.ndarray) -> np.ndarray:
    min_x, min_y, max_x, max_y = mouth_box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if mouth_cutout is None or box_width is None or box_height is None or face_mask is None or mouth_polygon is None:
        return frame

    try:
        resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
        roi = frame[min_y:max_y, min_x:max_x]

        if roi.shape != resized_mouth_cutout.shape:
            resized_mouth_cutout = cv2.resize(resized_mouth_cutout, (roi.shape[1], roi.shape[0]))

        color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi)
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        # Improved feathering for smoother mouth mask
        feather_amount = min(35, box_width // modules.globals.mask_feather_ratio, box_height // modules.globals.mask_feather_ratio)
        feathered_mask = cv2.GaussianBlur(polygon_mask.astype(float), (0, 0), feather_amount * 1.2)
        feathered_mask = feathered_mask / feathered_mask.max()
        
        # Additional smoothing pass for extra softness
        feathered_mask = cv2.GaussianBlur(feathered_mask, (7, 7), 2)
        
        # Fix black line artifacts by ensuring smooth mask transitions
        feathered_mask = np.clip(feathered_mask, 0.1, 0.9)  # Avoid pure 0 and 1 values

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask = feathered_mask * (face_mask_roi / 255.0)
        combined_mask = combined_mask[:, :, np.newaxis]
        blended = (color_corrected_mouth * combined_mask + roi * (1 - combined_mask)).astype(np.uint8)

        face_mask_3channel = np.repeat(face_mask_roi[:, :, np.newaxis], 3, axis=2) / 255.0
        final_blend = blended * face_mask_3channel + roi * (1 - face_mask_3channel)
        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
    except Exception:
        pass

    return frame


def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        landmarks = landmarks.astype(np.int32)
        right_side_face = landmarks[0:16]
        left_side_face = landmarks[17:32]
        right_eye = landmarks[33:42]
        right_eye_brow = landmarks[43:51]
        left_eye = landmarks[87:96]
        left_eye_brow = landmarks[97:105]

        right_eyebrow_top = np.min(right_eye_brow[:, 1])
        left_eyebrow_top = np.min(left_eye_brow[:, 1])
        eyebrow_top = min(right_eyebrow_top, left_eyebrow_top)

        face_top = np.min([right_side_face[0, 1], left_side_face[-1, 1]])
        forehead_height = face_top - eyebrow_top
        extended_forehead_height = int(forehead_height * 5.0)

        forehead_left = right_side_face[0].copy()
        forehead_right = left_side_face[-1].copy()
        forehead_left[1] -= extended_forehead_height
        forehead_right[1] -= extended_forehead_height

        face_outline = np.vstack([[forehead_left], right_side_face, left_side_face[::-1], [forehead_right]])
        padding = int(np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05)

        hull = cv2.convexHull(face_outline)
        hull_padded = []
        for point in hull:
            x, y = point[0]
            center = np.mean(face_outline, axis=0)
            direction = np.array([x, y]) - center
            direction = direction / np.linalg.norm(direction)
            padded_point = np.array([x, y]) + direction * padding
            hull_padded.append(padded_point)

        hull_padded = np.array(hull_padded, dtype=np.int32)
        cv2.fillConvexPoly(mask, hull_padded, 255)
        mask = cv2.GaussianBlur(mask, (5, 5), 3)

    return mask


def apply_color_transfer(source, target):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)

    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    source = (source - source_mean) * (target_std / source_std) + target_mean
    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)