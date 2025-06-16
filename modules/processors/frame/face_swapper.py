from typing import Any, List, Optional, Tuple
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
from modules.hair_segmenter import segment_hair
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


def _prepare_warped_source_material_and_mask(
    source_face_obj: Face,
    source_frame_full: Frame,
    matrix: np.ndarray,
    dsize: tuple # Built-in tuple is fine here for parameter type
) -> Tuple[Optional[Frame], Optional[Frame]]:
    """
    Prepares warped source material (full image) and a combined (face+hair) mask for blending.
    Returns (None, None) if essential masks cannot be generated.
    """
    try:
        # Generate Hair Mask
        hair_only_mask_source_raw = segment_hair(source_frame_full)
        if hair_only_mask_source_raw is None:
            logging.error("segment_hair returned None, which is unexpected.")
            return None, None
        if hair_only_mask_source_raw.ndim == 3 and hair_only_mask_source_raw.shape[2] == 3:
            hair_only_mask_source_raw = cv2.cvtColor(hair_only_mask_source_raw, cv2.COLOR_BGR2GRAY)
        _, hair_only_mask_source_binary = cv2.threshold(hair_only_mask_source_raw, 127, 255, cv2.THRESH_BINARY)
    except Exception as e:
        logging.error(f"Hair segmentation failed: {e}", exc_info=True)
        return None, None

    try:
        # Generate Face Mask
        face_only_mask_source_raw = create_face_mask(source_face_obj, source_frame_full)
        if face_only_mask_source_raw is None:
            logging.error("create_face_mask returned None, which is unexpected.")
            return None, None
        _, face_only_mask_source_binary = cv2.threshold(face_only_mask_source_raw, 127, 255, cv2.THRESH_BINARY)
    except Exception as e:
        logging.error(f"Face mask creation failed for source: {e}", exc_info=True)
        return None, None

    # Combine Face and Hair Masks and Warp
    try:
        if face_only_mask_source_binary.shape != hair_only_mask_source_binary.shape:
            logging.warning("Resizing hair mask to match face mask for source during preparation.")
            hair_only_mask_source_binary = cv2.resize(
                hair_only_mask_source_binary,
                (face_only_mask_source_binary.shape[1], face_only_mask_source_binary.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        actual_combined_source_mask = cv2.bitwise_or(face_only_mask_source_binary, hair_only_mask_source_binary)
        actual_combined_source_mask_blurred = cv2.GaussianBlur(actual_combined_source_mask, (5, 5), 3)

        warped_full_source_material = cv2.warpAffine(source_frame_full, matrix, dsize)
        warped_combined_mask_temp = cv2.warpAffine(actual_combined_source_mask_blurred, matrix, dsize)
        _, warped_combined_mask_binary_for_clone = cv2.threshold(warped_combined_mask_temp, 127, 255, cv2.THRESH_BINARY)
    except Exception as e:
        logging.error(f"Mask combination or warping failed: {e}", exc_info=True)
        return None, None

    return warped_full_source_material, warped_combined_mask_binary_for_clone

def _blend_material_onto_frame(
    base_frame: Frame,
    material_to_blend: Frame,
    mask_for_blending: Frame
) -> Frame:
    """
    Blends material onto a base frame using a mask.
    Uses seamlessClone if possible, otherwise falls back to simple masking.
    """
    x, y, w, h = cv2.boundingRect(mask_for_blending)
    output_frame = base_frame # Start with base, will be modified by blending

    if w > 0 and h > 0:
        center = (x + w // 2, y + h // 2)

        if material_to_blend.shape == base_frame.shape and \
           material_to_blend.dtype == base_frame.dtype and \
           mask_for_blending.dtype == np.uint8:
            try:
                output_frame = cv2.seamlessClone(material_to_blend, base_frame, mask_for_blending, center, cv2.NORMAL_CLONE)
            except cv2.error as e:
                logging.warning(f"cv2.seamlessClone failed: {e}. Falling back to simple blending.")
                boolean_mask = mask_for_blending > 127
                output_frame[boolean_mask] = material_to_blend[boolean_mask]
        else:
            logging.warning("Mismatch in shape/type for seamlessClone. Falling back to simple blending.")
            boolean_mask = mask_for_blending > 127
            output_frame[boolean_mask] = material_to_blend[boolean_mask]
    else:
        logging.info("Warped mask for blending is empty. Skipping blending.")

    return output_frame


def swap_face(source_face_obj: Face, target_face: Face, source_frame_full: Frame, temp_frame: Frame) -> Frame:
    face_swapper = get_face_swapper()

    # Apply the base face swap
    swapped_frame = face_swapper.get(temp_frame, target_face, source_face_obj, paste_back=True)
    final_swapped_frame = swapped_frame # Initialize with the base swap. Copy is made only if needed.

    if getattr(modules.globals, 'enable_hair_swapping', True): # Default to True if attribute is missing
        if not (source_face_obj.kps is not None and \
                target_face.kps is not None and \
                source_face_obj.kps.shape[0] >= 3 and \
                target_face.kps.shape[0] >= 3):
            logging.warning(
                f"Skipping hair blending due to insufficient keypoints. "
                f"Source kps: {source_face_obj.kps.shape if source_face_obj.kps is not None else 'None'}, "
                f"Target kps: {target_face.kps.shape if target_face.kps is not None else 'None'}."
            )
        else:
            source_kps_float = source_face_obj.kps.astype(np.float32)
            target_kps_float = target_face.kps.astype(np.float32)
            matrix, _ = cv2.estimateAffinePartial2D(source_kps_float, target_kps_float, method=cv2.LMEDS)

            if matrix is None:
                logging.warning("Failed to estimate affine transformation matrix for hair. Skipping hair blending.")
            else:
                dsize = (temp_frame.shape[1], temp_frame.shape[0]) # width, height

                warped_material, warped_mask = _prepare_warped_source_material_and_mask(
                    source_face_obj, source_frame_full, matrix, dsize
                )

                if warped_material is not None and warped_mask is not None:
                    # Make a copy only now that we are sure we will modify it for hair.
                    final_swapped_frame = swapped_frame.copy()

                    try:
                        color_corrected_material = apply_color_transfer(warped_material, final_swapped_frame)
                    except Exception as e:
                        logging.warning(f"Color transfer failed: {e}. Proceeding with uncorrected material for hair blending.", exc_info=True)
                        color_corrected_material = warped_material # Use uncorrected material as fallback

                    final_swapped_frame = _blend_material_onto_frame(
                        final_swapped_frame,
                        color_corrected_material,
                        warped_mask
                    )

    # Mouth Mask Logic (operates on final_swapped_frame)
    if modules.globals.mouth_mask:
        # If final_swapped_frame wasn't copied for hair, it needs to be copied now before mouth mask modification.
        if final_swapped_frame is swapped_frame: # Check if it's still the same object
            final_swapped_frame = swapped_frame.copy()

        # Create a mask for the target face
        face_mask = create_face_mask(target_face, temp_frame)

        # Create the mouth mask
        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
            create_lower_mouth_mask(target_face, temp_frame)
        )

        # Apply the mouth area
        # Apply to final_swapped_frame if hair blending happened, otherwise to swapped_frame
        final_swapped_frame = apply_mouth_area(
            final_swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
        )

        if modules.globals.show_mouth_mask_box:
            mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
            final_swapped_frame = draw_mouth_mask_visualization(
                final_swapped_frame, target_face, mouth_mask_data
            )

    return final_swapped_frame


def process_frame(source_face_obj: Face, source_frame_full: Frame, temp_frame: Frame) -> Frame:
    if modules.globals.color_correction:
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)

    if modules.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                if source_face_obj and target_face:
                    temp_frame = swap_face(source_face_obj, target_face, source_frame_full, temp_frame)
                else:
                    print("Face detection failed for target/source.")
    else:
        target_face = get_one_face(temp_frame)
        if target_face and source_face_obj:
            temp_frame = swap_face(source_face_obj, target_face, source_frame_full, temp_frame)
        else:
            logging.error("Face detection failed for target or source.")
    return temp_frame


def _process_image_target_v2(source_frame_full: Frame, temp_frame: Frame) -> Frame:
    if modules.globals.many_faces:
        source_face_obj = default_source_face()
        if source_face_obj:
            for map_item in modules.globals.source_target_map:
                target_face = map_item["target"]["face"]
                temp_frame = swap_face(source_face_obj, target_face, source_frame_full, temp_frame)
    else: # not many_faces
        for map_item in modules.globals.source_target_map:
            if "source" in map_item:
                source_face_obj = map_item["source"]["face"]
                target_face = map_item["target"]["face"]
                temp_frame = swap_face(source_face_obj, target_face, source_frame_full, temp_frame)
    return temp_frame

def _process_video_target_v2(source_frame_full: Frame, temp_frame: Frame, temp_frame_path: str) -> Frame:
    if modules.globals.many_faces:
        source_face_obj = default_source_face()
        if source_face_obj:
            for map_item in modules.globals.source_target_map:
                target_frames_data = [f for f in map_item.get("target_faces_in_frame", []) if f.get("location") == temp_frame_path]
                for frame_data in target_frames_data:
                    for target_face in frame_data.get("faces", []):
                        temp_frame = swap_face(source_face_obj, target_face, source_frame_full, temp_frame)
    else: # not many_faces
        for map_item in modules.globals.source_target_map:
            if "source" in map_item:
                source_face_obj = map_item["source"]["face"]
                target_frames_data = [f for f in map_item.get("target_faces_in_frame", []) if f.get("location") == temp_frame_path]
                for frame_data in target_frames_data:
                    for target_face in frame_data.get("faces", []):
                        temp_frame = swap_face(source_face_obj, target_face, source_frame_full, temp_frame)
    return temp_frame

def _process_live_target_v2(source_frame_full: Frame, temp_frame: Frame) -> Frame:
    detected_faces = get_many_faces(temp_frame)
    if not detected_faces:
        return temp_frame

    if modules.globals.many_faces:
        source_face_obj = default_source_face()
        if source_face_obj:
            for target_face in detected_faces:
                temp_frame = swap_face(source_face_obj, target_face, source_frame_full, temp_frame)
    else: # not many_faces (apply simple_map logic)
        if not modules.globals.simple_map or \
           not modules.globals.simple_map.get("target_embeddings") or \
           not modules.globals.simple_map.get("source_faces"):
            logging.warning("Simple map is not configured correctly. Skipping face swap.")
            return temp_frame

        target_embeddings = modules.globals.simple_map["target_embeddings"]
        source_faces_from_map = modules.globals.simple_map["source_faces"]

        if len(detected_faces) <= len(target_embeddings):
            for detected_face in detected_faces:
                closest_centroid_index, _ = find_closest_centroid(target_embeddings, detected_face.normed_embedding)
                if closest_centroid_index < len(source_faces_from_map):
                    source_face_obj_from_map = source_faces_from_map[closest_centroid_index]
                    temp_frame = swap_face(source_face_obj_from_map, detected_face, source_frame_full, temp_frame)
                else:
                    logging.warning(f"Centroid index {closest_centroid_index} out of bounds for source_faces_from_map.")
        else: # More detected faces than target embeddings in simple_map
            detected_faces_embeddings = [face.normed_embedding for face in detected_faces]
            for i, target_embedding in enumerate(target_embeddings):
                if i < len(source_faces_from_map):
                    closest_detected_face_index, _ = find_closest_centroid(detected_faces_embeddings, target_embedding)
                    source_face_obj_from_map = source_faces_from_map[i]
                    target_face_to_swap = detected_faces[closest_detected_face_index]
                    temp_frame = swap_face(source_face_obj_from_map, target_face_to_swap, source_frame_full, temp_frame)
                    # Optionally, remove the swapped detected face to prevent re-swapping if one source maps to multiple targets.
                    # This depends on desired behavior. For now, simple independent mapping.
                else:
                    logging.warning(f"Index {i} out of bounds for source_faces_from_map in simple_map else case.")
    return temp_frame


def process_frame_v2(source_frame_full: Frame, temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    if is_image(modules.globals.target_path):
        return _process_image_target_v2(source_frame_full, temp_frame)
    elif is_video(modules.globals.target_path):
        return _process_video_target_v2(source_frame_full, temp_frame, temp_frame_path)
    else: # This is the live cam / generic case
        return _process_live_target_v2(source_frame_full, temp_frame)


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    source_img = cv2.imread(source_path)
    if source_img is None:
        logging.error(f"Failed to read source image from {source_path}")
        return

    if not modules.globals.map_faces:
        source_face_obj = get_one_face(source_img) # Use source_img here
        if not source_face_obj:
            logging.error(f"No face detected in source image {source_path}")
            return
        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is None:
                logging.warning(f"Failed to read temp_frame from {temp_frame_path}, skipping.")
                continue
            try:
                result = process_frame(source_face_obj, source_img, temp_frame)
                cv2.imwrite(temp_frame_path, result)
            except Exception as exception:
                logging.error(f"Error processing frame {temp_frame_path}: {exception}", exc_info=True)
                pass
            if progress:
                progress.update(1)
    else: # This is for map_faces == True
        # In map_faces=True, source_face is determined per mapping.
        # process_frame_v2 will need source_frame_full for hair,
        # which should be the original source_path image.
        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is None:
                logging.warning(f"Failed to read temp_frame from {temp_frame_path}, skipping.")
                continue
            try:
                # Pass source_img (as source_frame_full) to process_frame_v2
                result = process_frame_v2(source_img, temp_frame, temp_frame_path)
                cv2.imwrite(temp_frame_path, result)
            except Exception as exception:
                logging.error(f"Error processing frame {temp_frame_path} with map_faces: {exception}", exc_info=True)
                pass
            if progress:
                progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_img = cv2.imread(source_path)
    if source_img is None:
        logging.error(f"Failed to read source image from {source_path}")
        return

    target_frame = cv2.imread(target_path)
    if target_frame is None:
        logging.error(f"Failed to read target image from {target_path}")
        return

    # Read the original target frame once at the beginning
    original_target_frame = cv2.imread(target_path)
    if original_target_frame is None:
        logging.error(f"Failed to read original target image from {target_path}")
        return

    result = None  # Initialize result

    if not modules.globals.map_faces:
        source_face_obj = get_one_face(source_img) # Use source_img here
        if not source_face_obj:
            logging.error(f"No face detected in source image {source_path}")
            return
        result = process_frame(source_face_obj, source_img, original_target_frame)
    else: # map_faces is True
        if modules.globals.many_faces:
            update_status(
                "Many faces enabled. Using first source image. Progressing...", NAME
            )
        # process_frame_v2 takes the original target frame for processing.
        # target_path is passed as temp_frame_path for consistency with process_frame_v2's signature,
        # used for map lookups in video context but less critical for single images.
        result = process_frame_v2(source_img, original_target_frame, target_path)

    if result is not None:
        cv2.imwrite(output_path, result)
    else:
        logging.error(f"Processing image {target_path} failed, result was None.")


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
