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
    download_directory_path = abs_dir
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
            model_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path, providers=modules.globals.execution_providers
            )
    return FACE_SWAPPER


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    face_swapper = get_face_swapper()

    # Apply the face swap
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

# This should be the core function that applies mappings from simple_map to a frame
def _apply_mapping_to_frame(temp_frame: Frame) -> Frame:
    if not modules.globals.simple_map or \
       not modules.globals.simple_map.get('target_embeddings') or \
       not modules.globals.simple_map.get('source_faces'):
        # print("FaceSwapper: simple_map not populated for mapped processing. Returning original frame.")
        return temp_frame

    detected_faces = get_many_faces(temp_frame)
    if not detected_faces:
        return temp_frame

    for detected_face in detected_faces:
        if not hasattr(detected_face, 'normed_embedding') or detected_face.normed_embedding is None:
            continue # Skip if face has no embedding

        closest_centroid_index, _ = find_closest_centroid(
            modules.globals.simple_map['target_embeddings'],
            detected_face.normed_embedding
        )

        if closest_centroid_index < len(modules.globals.simple_map['source_faces']):
            source_face_to_use = modules.globals.simple_map['source_faces'][closest_centroid_index]
            if source_face_to_use: # Ensure a source face is actually there
                 temp_frame = swap_face(source_face_to_use, detected_face, temp_frame)
        # else: print(f"Warning: Centroid index {closest_centroid_index} out of bounds for source_faces.")

    return temp_frame


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    # This is for single source_face to potentially many target_faces (if many_faces is on)
    # Or single source to single target (if many_faces is off)
    # This function should NOT be used if Globals.map_faces is True.
    if modules.globals.color_correction: # This global might need namespacing if other modules use it
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)

    if modules.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                if source_face and target_face:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
                else:
                    print("Face detection failed for target/source.")
    else:
        target_face = get_one_face(temp_frame)
        if target_face and source_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
        else:
            logging.error("Face detection failed for target or source.")
    return temp_frame



# This is the new V2 for mapped processing of a single frame (used by live feed and process_video_v2)
# It should not rely on Globals.target_path for context, only on Globals.simple_map
def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame: # temp_frame_path is mostly for debug here
    if modules.globals.color_correction: # This global might need namespacing
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)

    if not modules.globals.map_faces:
        # This case should ideally not be reached if called from process_video_v2 or live_feed when map_faces is true.
        # However, if it is, it implies a logic error or fallback.
        # For now, if map_faces is false, it means use the single Globals.source_path.
        # This makes process_frame_v2 behave like process_frame if map_faces is off.
        # This might be confusing. A clearer separation would be better.
        # print("Warning: process_frame_v2 called when map_faces is False. Using standard process_frame logic.")
        source_face = None
        if modules.globals.source_path and os.path.exists(modules.globals.source_path):
            source_cv2_img = cv2.imread(modules.globals.source_path)
            if source_cv2_img is not None:
                source_face = get_one_face(source_cv2_img)

        if source_face:
            return process_frame(source_face, temp_frame) # Fallback to old logic for this scenario
        else: # No source face, return original frame
            return temp_frame

    # If map_faces is True, proceed with mapped logic using _apply_mapping_to_frame
    return _apply_mapping_to_frame(temp_frame)


# Old process_frames, used by old process_video. Kept for now if any CLI path uses process_video directly.
# Should be deprecated in favor of core.py's video loop calling process_frame or process_frame_v2.
def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    # This function's logic is now largely superseded by core.py's process_media loop.
    # If map_faces is True, core.py will call process_video_v2 which then calls process_frame_v2.
    # If map_faces is False, core.py will call process_video which calls this,
    # and this will use the single source_face.

    source_face = None
    if not modules.globals.map_faces: # Only get single source if not mapping
        if source_path and os.path.exists(source_path): # Ensure source_path is valid
            source_img_content = cv2.imread(source_path)
            if source_img_content is not None:
                source_face = get_one_face(source_img_content)
        if not source_face:
            update_status("Warning: No source face found for standard video processing. Frames will not be swapped.", NAME)
            if progress: progress.update(len(temp_frame_paths)) # Mark all as "processed"
            return

    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            if progress: progress.update(1)
            continue
        try:
            if modules.globals.map_faces: # Should be handled by process_video_v2 now
                result = process_frame_v2(temp_frame, temp_frame_path)
            elif source_face: # Standard single source processing
                result = process_frame(source_face, temp_frame)
            else: # No source, no map
                result = temp_frame
            cv2.imwrite(temp_frame_path, result)
        except Exception as e:
            print(f"Error processing frame {temp_frame_path}: {e}")
            pass # Keep original frame if error
        if progress:
            progress.update(1)


# process_image is called by core.py when not map_faces
def process_image(source_path: str, target_path: str, output_path: str) -> None:
    # This is for single source_path to target_path.
    # map_faces=True scenario is handled by process_image_v2.
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    if source_face and target_frame is not None:
        result = process_frame(source_face, target_frame) # process_frame handles many_faces internally
        cv2.imwrite(output_path, result)
    elif target_frame is not None : # No source face, but target exists
        update_status("No source face for process_image, saving original target.", NAME)
        cv2.imwrite(output_path, target_frame)
    else:
        update_status("Failed to read target image in process_image.", NAME)


# process_image_v2 is called by core.py when map_faces is True
def process_image_v2(target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        update_status(f"Failed to read target image at {target_path}", NAME)
        return

    if modules.globals.color_correction:
         target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

    result_frame = _apply_mapping_to_frame(target_frame)
    cv2.imwrite(output_path, result_frame)


# process_video is called by core.py when not map_faces
def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    # This function should setup for process_frames which handles single source processing.
    # core.py's process_media calls this.
    # process_frames will get the single source face from source_path.
    modules.processors.frame.core.process_video( # This is a generic utility from core
        source_path, temp_frame_paths, process_frames # Pass our process_frames
    )

# process_video_v2 is called by core.py when map_faces is True
def process_video_v2(temp_frame_paths: List[str]) -> None:
    # This function iterates frames and calls the mapped version of process_frame_v2
    for frame_path in temp_frame_paths:
        current_frame = cv2.imread(frame_path)
        if current_frame is None:
            print(f"Warning: Could not read frame {frame_path} in process_video_v2. Skipping.")
            continue

        processed_frame = process_frame_v2(current_frame, frame_path) # process_frame_v2 now uses _apply_mapping_to_frame
        cv2.imwrite(frame_path, processed_frame)


def create_lower_mouth_mask(
    face: Face, frame: Frame
) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        #                  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
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
        lower_lip_landmarks = landmarks[lower_lip_order].astype(
            np.float32
        )  # Use float for precise calculations

        # Calculate the center of the landmarks
        center = np.mean(lower_lip_landmarks, axis=0)

        # Expand the landmarks outward
        expansion_factor = (
            1 + modules.globals.mask_down_size
        )  # Adjust this for more or less expansion
        expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center

        # Extend the top lip part
        toplip_indices = [
            20,
            0,
            1,
            2,
            3,
            4,
            5,
        ]  # Indices for landmarks 2, 65, 66, 62, 70, 69, 18
        toplip_extension = (
            modules.globals.mask_size * 0.5
        )  # Adjust this factor to control the extension
        for idx in toplip_indices:
            direction = expanded_landmarks[idx] - center
            direction = direction / np.linalg.norm(direction)
            expanded_landmarks[idx] += direction * toplip_extension

        # Extend the bottom part (chin area)
        chin_indices = [
            11,
            12,
            13,
            14,
            15,
            16,
        ]  # Indices for landmarks 21, 22, 23, 24, 0, 8
        chin_extension = 2 * 0.2  # Adjust this factor to control the extension
        for idx in chin_indices:
            expanded_landmarks[idx][1] += (
                expanded_landmarks[idx][1] - center[1]
            ) * chin_extension

        # Convert back to integer coordinates
        expanded_landmarks = expanded_landmarks.astype(np.int32)

        # Calculate bounding box for the expanded lower mouth
        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)

        # Add some padding to the bounding box
        padding = int((max_x - min_x) * 0.1)  # 10% padding
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(frame.shape[1], max_x + padding)
        max_y = min(frame.shape[0], max_y + padding)

        # Ensure the bounding box dimensions are valid
        if max_x <= min_x or max_y <= min_y:
            if (max_x - min_x) <= 1:
                max_x = min_x + 1
            if (max_y - min_y) <= 1:
                max_y = min_y + 1

        # Create the mask
        mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        cv2.fillPoly(mask_roi, [expanded_landmarks - [min_x, min_y]], 255)

        # Apply Gaussian blur to soften the mask edges
        mask_roi = cv2.GaussianBlur(mask_roi, (15, 15), 5)

        # Place the mask ROI in the full-sized mask
        mask[min_y:max_y, min_x:max_x] = mask_roi

        # Extract the masked area from the frame
        mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()

        # Return the expanded lower lip polygon in original frame coordinates
        lower_lip_polygon = expanded_landmarks

    return mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon


def draw_mouth_mask_visualization(
    frame: Frame, face: Face, mouth_mask_data: tuple
) -> Frame:
    landmarks = face.landmark_2d_106
    if landmarks is not None and mouth_mask_data is not None:
        mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon = (
            mouth_mask_data
        )

        vis_frame = frame.copy()

        # Ensure coordinates are within frame bounds
        height, width = vis_frame.shape[:2]
        min_x, min_y = max(0, min_x), max(0, min_y)
        max_x, max_y = min(width, max_x), min(height, max_y)

        # Adjust mask to match the region size
        mask_region = mask[0 : max_y - min_y, 0 : max_x - min_x]

        # Remove the color mask overlay
        # color_mask = cv2.applyColorMap((mask_region * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Ensure shapes match before blending
        vis_region = vis_frame[min_y:max_y, min_x:max_x]
        # Remove blending with color_mask
        # if vis_region.shape[:2] == color_mask.shape[:2]:
        #     blended = cv2.addWeighted(vis_region, 0.7, color_mask, 0.3, 0)
        #     vis_frame[min_y:max_y, min_x:max_x] = blended

        # Draw the lower lip polygon
        cv2.polylines(vis_frame, [lower_lip_polygon], True, (0, 255, 0), 2)

        # Remove the red box
        # cv2.rectangle(vis_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

        # Visualize the feathered mask
        feather_amount = max(
            1,
            min(
                30,
                (max_x - min_x) // modules.globals.mask_feather_ratio,
                (max_y - min_y) // modules.globals.mask_feather_ratio,
            ),
        )
        # Ensure kernel size is odd
        kernel_size = 2 * feather_amount + 1
        feathered_mask = cv2.GaussianBlur(
            mask_region.astype(float), (kernel_size, kernel_size), 0
        )
        feathered_mask = (feathered_mask / feathered_mask.max() * 255).astype(np.uint8)
        # Remove the feathered mask color overlay
        # color_feathered_mask = cv2.applyColorMap(feathered_mask, cv2.COLORMAP_VIRIDIS)

        # Ensure shapes match before blending feathered mask
        # if vis_region.shape == color_feathered_mask.shape:
        #     blended_feathered = cv2.addWeighted(vis_region, 0.7, color_feathered_mask, 0.3, 0)
        #     vis_frame[min_y:max_y, min_x:max_x] = blended_feathered

        # Add labels
        cv2.putText(
            vis_frame,
            "Lower Mouth Mask",
            (min_x, min_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            vis_frame,
            "Feathered Mask",
            (min_x, max_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return vis_frame
    return frame


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: tuple,
    face_mask: np.ndarray,
    mouth_polygon: np.ndarray,
) -> np.ndarray:
    min_x, min_y, max_x, max_y = mouth_box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (
        mouth_cutout is None
        or box_width is None
        or box_height is None
        or face_mask is None
        or mouth_polygon is None
    ):
        return frame

    try:
        resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
        roi = frame[min_y:max_y, min_x:max_x]

        if roi.shape != resized_mouth_cutout.shape:
            resized_mouth_cutout = cv2.resize(
                resized_mouth_cutout, (roi.shape[1], roi.shape[0])
            )

        color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi)

        # Use the provided mouth polygon to create the mask
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        # Apply feathering to the polygon mask
        feather_amount = min(
            30,
            box_width // modules.globals.mask_feather_ratio,
            box_height // modules.globals.mask_feather_ratio,
        )
        feathered_mask = cv2.GaussianBlur(
            polygon_mask.astype(float), (0, 0), feather_amount
        )
        feathered_mask = feathered_mask / feathered_mask.max()

        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask = feathered_mask * (face_mask_roi / 255.0)

        combined_mask = combined_mask[:, :, np.newaxis]
        blended = (
            color_corrected_mouth * combined_mask + roi * (1 - combined_mask)
        ).astype(np.uint8)

        # Apply face mask to blended result
        face_mask_3channel = (
            np.repeat(face_mask_roi[:, :, np.newaxis], 3, axis=2) / 255.0
        )
        final_blend = blended * face_mask_3channel + roi * (1 - face_mask_3channel)

        frame[min_y:max_y, min_x:max_x] = final_blend.astype(np.uint8)
    except Exception as e:
        pass

    return frame


def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is not None:
        # Convert landmarks to int32
        landmarks = landmarks.astype(np.int32)

        # Extract facial features
        right_side_face = landmarks[0:16]
        left_side_face = landmarks[17:32]
        right_eye = landmarks[33:42]
        right_eye_brow = landmarks[43:51]
        left_eye = landmarks[87:96]
        left_eye_brow = landmarks[97:105]

        # Calculate forehead extension
        right_eyebrow_top = np.min(right_eye_brow[:, 1])
        left_eyebrow_top = np.min(left_eye_brow[:, 1])
        eyebrow_top = min(right_eyebrow_top, left_eyebrow_top)

        face_top = np.min([right_side_face[0, 1], left_side_face[-1, 1]])
        forehead_height = face_top - eyebrow_top
        extended_forehead_height = int(forehead_height * 5.0)  # Extend by 50%

        # Create forehead points
        forehead_left = right_side_face[0].copy()
        forehead_right = left_side_face[-1].copy()
        forehead_left[1] -= extended_forehead_height
        forehead_right[1] -= extended_forehead_height

        # Combine all points to create the face outline
        face_outline = np.vstack(
            [
                [forehead_left],
                right_side_face,
                left_side_face[
                    ::-1
                ],  # Reverse left side to create a continuous outline
                [forehead_right],
            ]
        )

        # Calculate padding
        padding = int(
            np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05
        )  # 5% of face width

        # Create a slightly larger convex hull for padding
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

        # Fill the padded convex hull
        cv2.fillConvexPoly(mask, hull_padded, 255)

        # Smooth the mask edges
        mask = cv2.GaussianBlur(mask, (5, 5), 3)

    return mask


def apply_color_transfer(source, target):
    """
    Apply color transfer from target to source image
    """
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    source_mean, source_std = cv2.meanStdDev(source)
    target_mean, target_std = cv2.meanStdDev(target)

    # Reshape mean and std to be broadcastable
    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    # Perform the color transfer
    source = (source - source_mean) * (target_std / source_std) + target_mean

    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)
