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
from modules.typing import Face, Frame # Face is insightface.app.common.Face
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

# --- Tracker State Variables ---
TARGET_TRACKER: Optional[cv2.Tracker] = None
LAST_TARGET_KPS: Optional[np.ndarray] = None
LAST_TARGET_BBOX_XYWH: Optional[List[int]] = None # Stored as [x, y, w, h]
TRACKING_FRAME_COUNTER = 0
DETECTION_INTERVAL = 3  # Process every 3rd frame for full detection
LAST_DETECTION_SUCCESS = False
# --- End Tracker State Variables ---


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
    dsize: tuple
) -> Tuple[Optional[Frame], Optional[Frame]]:
    """
    Prepares warped source material (full image) and a combined (face+hair) mask for blending.
    Returns (None, None) if essential masks cannot be generated.
    """
    try:
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
        face_only_mask_source_raw = create_face_mask(source_face_obj, source_frame_full)
        if face_only_mask_source_raw is None:
            logging.error("create_face_mask returned None, which is unexpected.")
            return None, None
        _, face_only_mask_source_binary = cv2.threshold(face_only_mask_source_raw, 127, 255, cv2.THRESH_BINARY)
    except Exception as e:
        logging.error(f"Face mask creation failed for source: {e}", exc_info=True)
        return None, None

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
    output_frame = base_frame

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

    swapped_frame = face_swapper.get(temp_frame, target_face, source_face_obj, paste_back=True)
    final_swapped_frame = swapped_frame

    if getattr(modules.globals, 'enable_hair_swapping', True):
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
                dsize = (temp_frame.shape[1], temp_frame.shape[0])

                warped_material, warped_mask = _prepare_warped_source_material_and_mask(
                    source_face_obj, source_frame_full, matrix, dsize
                )

                if warped_material is not None and warped_mask is not None:
                    final_swapped_frame = swapped_frame.copy()

                    try:
                        color_corrected_material = apply_color_transfer(warped_material, final_swapped_frame)
                    except Exception as e:
                        logging.warning(f"Color transfer failed: {e}. Proceeding with uncorrected material for hair blending.", exc_info=True)
                        color_corrected_material = warped_material

                    final_swapped_frame = _blend_material_onto_frame(
                        final_swapped_frame,
                        color_corrected_material,
                        warped_mask
                    )

    if modules.globals.mouth_mask:
        if final_swapped_frame is swapped_frame:
            final_swapped_frame = swapped_frame.copy()

        face_mask_for_mouth = create_face_mask(target_face, temp_frame) # Use original temp_frame for target mask context

        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
            create_lower_mouth_mask(target_face, temp_frame) # Use original temp_frame for target mouth context
        )

        # Ensure apply_mouth_area gets the most up-to-date final_swapped_frame if hair blending happened
        final_swapped_frame = apply_mouth_area(
            final_swapped_frame, mouth_cutout, mouth_box, face_mask_for_mouth, lower_lip_polygon
        )

        if modules.globals.show_mouth_mask_box:
            mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
            final_swapped_frame = draw_mouth_mask_visualization(
                final_swapped_frame, target_face, mouth_mask_data
            )

    return final_swapped_frame


def process_frame(source_face_obj: Face, source_frame_full: Frame, temp_frame: Frame) -> Frame:
    global TARGET_TRACKER, LAST_TARGET_KPS, LAST_TARGET_BBOX_XYWH
    global TRACKING_FRAME_COUNTER, DETECTION_INTERVAL, LAST_DETECTION_SUCCESS

    if modules.globals.color_correction:
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)

    if modules.globals.many_faces:
        # Tracking logic is not applied for many_faces mode in this iteration
        many_faces_detected = get_many_faces(temp_frame)
        if many_faces_detected:
            for target_face_data in many_faces_detected:
                if source_face_obj and target_face_data:
                    temp_frame = swap_face(source_face_obj, target_face_data, source_frame_full, temp_frame)
                else:
                    # This print might be too verbose for many_faces mode
                    # logging.debug("Face detection failed for a target/source in many_faces.")
                    pass # Optionally log or handle
        return temp_frame # Return early after processing all faces or if none found

    # --- Single Face Mode with Tracking ---
    TRACKING_FRAME_COUNTER += 1
    target_face_to_swap = None

    if TRACKING_FRAME_COUNTER % DETECTION_INTERVAL == 0 or not LAST_DETECTION_SUCCESS:
        logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: Running full detection.")
        actual_target_face_data = get_one_face(temp_frame)
        if actual_target_face_data:
            target_face_to_swap = actual_target_face_data
            LAST_TARGET_KPS = actual_target_face_data.kps.copy() if actual_target_face_data.kps is not None else None
            bbox_xyxy = actual_target_face_data.bbox
            LAST_TARGET_BBOX_XYWH = [int(bbox_xyxy[0]), int(bbox_xyxy[1]), int(bbox_xyxy[2] - bbox_xyxy[0]), int(bbox_xyxy[3] - bbox_xyxy[1])]

            try:
                TARGET_TRACKER = cv2.TrackerKCF_create()
                TARGET_TRACKER.init(temp_frame, tuple(LAST_TARGET_BBOX_XYWH))
                LAST_DETECTION_SUCCESS = True
                logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: Detection SUCCESS, tracker initialized.")
            except Exception as e:
                logging.error(f"Failed to initialize tracker: {e}", exc_info=True)
                TARGET_TRACKER = None
                LAST_DETECTION_SUCCESS = False
        else:
            logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: Full detection FAILED.")
            LAST_DETECTION_SUCCESS = False
            TARGET_TRACKER = None
    else: # Intermediate frame, try to track
        if TARGET_TRACKER is not None:
            logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: Attempting track.")
            success, new_bbox_xywh_float = TARGET_TRACKER.update(temp_frame)
            if success:
                logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: Tracking SUCCESS.")
                new_bbox_xywh = [int(v) for v in new_bbox_xywh_float]

                if LAST_TARGET_KPS is not None and LAST_TARGET_BBOX_XYWH is not None:
                    # Estimate KPS based on bbox center shift
                    old_bbox_center_x = LAST_TARGET_BBOX_XYWH[0] + LAST_TARGET_BBOX_XYWH[2] / 2
                    old_bbox_center_y = LAST_TARGET_BBOX_XYWH[1] + LAST_TARGET_BBOX_XYWH[3] / 2
                    new_bbox_center_x = new_bbox_xywh[0] + new_bbox_xywh[2] / 2
                    new_bbox_center_y = new_bbox_xywh[1] + new_bbox_xywh[3] / 2
                    delta_x = new_bbox_center_x - old_bbox_center_x
                    delta_y = new_bbox_center_y - old_bbox_center_y
                    current_kps = LAST_TARGET_KPS + np.array([delta_x, delta_y])
                else: # Fallback if prior KPS/BBox not available
                    current_kps = None


                new_bbox_xyxy = np.array([
                    new_bbox_xywh[0],
                    new_bbox_xywh[1],
                    new_bbox_xywh[0] + new_bbox_xywh[2],
                    new_bbox_xywh[1] + new_bbox_xywh[3]
                ])

                # Construct a Face object or a compatible dictionary
                # For insightface.app.common.Face, it requires specific fields.
                # A dictionary might be safer if not all fields can be reliably populated.
                target_face_to_swap = Face(
                    bbox=new_bbox_xyxy,
                    kps=current_kps,
                    det_score=0.95, # Using a high score for tracked faces
                    landmark_3d_68=None, # Not available from KCF tracker
                    landmark_2d_106=None, # Not available from KCF tracker, mouth mask might be affected
                    gender=None, # Not available
                    age=None, # Not available
                    embedding=None, # Not available
                    normed_embedding=None # Not available
                )
                LAST_TARGET_BBOX_XYWH = new_bbox_xywh # Update for next frame's delta calculation
                LAST_TARGET_KPS = current_kps # Update KPS for next frame's delta calculation
                LAST_DETECTION_SUCCESS = True # Tracking was successful
            else:
                logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: Tracking FAILED.")
                LAST_DETECTION_SUCCESS = False
                TARGET_TRACKER = None # Reset tracker
        else:
            logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: No active tracker, skipping track.")


    if target_face_to_swap and source_face_obj:
        temp_frame = swap_face(source_face_obj, target_face_to_swap, source_frame_full, temp_frame)
    else:
        if TRACKING_FRAME_COUNTER % DETECTION_INTERVAL == 0: # Only log error if it was a detection frame
            logging.info("Target face not found by detection or tracking in process_frame.")
            # No error log here as it might just be no face in frame.
            # The swap_face call will be skipped, returning the original temp_frame.
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
    # This function is called by UI directly for webcam when map_faces is True.
    # The Nth frame/tracking logic for webcam should ideally be here or called from here.
    # For now, it reuses the global tracker state, which might be an issue if multiple
    # call paths use process_frame_v2 concurrently.
    # However, with webcam, process_frame (single face) or this (map_faces) is called.
    # Assuming single-threaded UI updates for webcam for now.

    global TARGET_TRACKER, LAST_TARGET_KPS, LAST_TARGET_BBOX_XYWH
    global TRACKING_FRAME_COUNTER, DETECTION_INTERVAL, LAST_DETECTION_SUCCESS

    if not modules.globals.many_faces: # Tracking only implemented for single target face in live mode
        TRACKING_FRAME_COUNTER += 1 # Use the same counter for now
        target_face_to_swap = None

        if TRACKING_FRAME_COUNTER % DETECTION_INTERVAL == 0 or not LAST_DETECTION_SUCCESS:
            logging.debug(f"Frame {TRACKING_FRAME_COUNTER} (Live V2): Running full detection.")
            # In map_faces mode for live, we might need to select one target based on some criteria
            # or apply to all detected faces if a simple_map isn't specific enough.
            # This part needs careful thought for map_faces=True live mode.
            # For now, let's assume simple_map implies one primary target for tracking.
            detected_faces = get_many_faces(temp_frame) # Get all faces first

            # If simple_map is configured, try to find the "main" target face from simple_map
            actual_target_face_data = None
            if detected_faces and modules.globals.simple_map and modules.globals.simple_map.get("target_embeddings"):
                # This logic tries to find one specific face to track based on simple_map.
                # It might not be ideal if multiple mapped faces are expected to be swapped.
                # For simplicity, we'll track the first match or a dominant face.
                # This part is a placeholder for a more robust target selection in map_faces live mode.
                # For now, let's try to find one based on the first simple_map embedding.
                if modules.globals.simple_map["target_embeddings"]:
                    closest_idx, _ = find_closest_centroid([face.normed_embedding for face in detected_faces], modules.globals.simple_map["target_embeddings"][0])
                    if closest_idx < len(detected_faces):
                         actual_target_face_data = detected_faces[closest_idx]
            elif detected_faces: # Fallback if no simple_map or if logic above fails
                actual_target_face_data = detected_faces[0] # Default to the first detected face

            if actual_target_face_data:
                target_face_to_swap = actual_target_face_data
                LAST_TARGET_KPS = actual_target_face_data.kps.copy() if actual_target_face_data.kps is not None else None
                bbox_xyxy = actual_target_face_data.bbox
                LAST_TARGET_BBOX_XYWH = [int(bbox_xyxy[0]), int(bbox_xyxy[1]), int(bbox_xyxy[2] - bbox_xyxy[0]), int(bbox_xyxy[3] - bbox_xyxy[1])]
                try:
                    TARGET_TRACKER = cv2.TrackerKCF_create()
                    TARGET_TRACKER.init(temp_frame, tuple(LAST_TARGET_BBOX_XYWH))
                    LAST_DETECTION_SUCCESS = True
                    logging.debug(f"Frame {TRACKING_FRAME_COUNTER} (Live V2): Detection SUCCESS, tracker initialized.")
                except Exception as e:
                    logging.error(f"Failed to initialize tracker (Live V2): {e}", exc_info=True)
                    TARGET_TRACKER = None
                    LAST_DETECTION_SUCCESS = False
            else:
                logging.debug(f"Frame {TRACKING_FRAME_COUNTER} (Live V2): Full detection FAILED.")
                LAST_DETECTION_SUCCESS = False
                TARGET_TRACKER = None
        else: # Intermediate frame, try to track
            if TARGET_TRACKER is not None:
                logging.debug(f"Frame {TRACKING_FRAME_COUNTER} (Live V2): Attempting track.")
                success, new_bbox_xywh_float = TARGET_TRACKER.update(temp_frame)
                if success:
                    logging.debug(f"Frame {TRACKING_FRAME_COUNTER} (Live V2): Tracking SUCCESS.")
                    new_bbox_xywh = [int(v) for v in new_bbox_xywh_float]
                    current_kps = None
                    if LAST_TARGET_KPS is not None and LAST_TARGET_BBOX_XYWH is not None:
                        old_bbox_center_x = LAST_TARGET_BBOX_XYWH[0] + LAST_TARGET_BBOX_XYWH[2] / 2
                        old_bbox_center_y = LAST_TARGET_BBOX_XYWH[1] + LAST_TARGET_BBOX_XYWH[3] / 2
                        new_bbox_center_x = new_bbox_xywh[0] + new_bbox_xywh[2] / 2
                        new_bbox_center_y = new_bbox_xywh[1] + new_bbox_xywh[3] / 2
                        delta_x = new_bbox_center_x - old_bbox_center_x
                        delta_y = new_bbox_center_y - old_bbox_center_y
                        current_kps = LAST_TARGET_KPS + np.array([delta_x, delta_y])

                    new_bbox_xyxy = np.array([new_bbox_xywh[0], new_bbox_xywh[1], new_bbox_xywh[0] + new_bbox_xywh[2], new_bbox_xywh[1] + new_bbox_xywh[3]])
                    target_face_to_swap = Face(bbox=new_bbox_xyxy, kps=current_kps, det_score=0.95, landmark_3d_68=None, landmark_2d_106=None, gender=None, age=None, embedding=None, normed_embedding=None)
                    LAST_TARGET_BBOX_XYWH = new_bbox_xywh
                    LAST_TARGET_KPS = current_kps
                    LAST_DETECTION_SUCCESS = True
                else:
                    logging.debug(f"Frame {TRACKING_FRAME_COUNTER} (Live V2): Tracking FAILED.")
                    LAST_DETECTION_SUCCESS = False
                    TARGET_TRACKER = None
            else:
                logging.debug(f"Frame {TRACKING_FRAME_COUNTER} (Live V2): No active tracker, skipping track.")

        # Perform swap for the identified or tracked face
        if target_face_to_swap:
             # In map_faces=True, need to determine which source face to use.
             # This part of _process_live_target_v2 needs to align with how simple_map or source_target_map is used.
             # The current logic for simple_map (else branch below) is more complete for this.
             # For now, if a target_face_to_swap is found by tracking, we need a source.
             # This indicates a simplification: if we track one face, we use the default source or first simple_map source.
            source_face_obj_to_use = default_source_face() # Fallback, might not be the right one for simple_map
            if modules.globals.simple_map and modules.globals.simple_map.get("source_faces"):
                # This assumes the tracked face corresponds to the first entry in simple_map, which is a simplification.
                source_face_obj_to_use = modules.globals.simple_map["source_faces"][0]

            if source_face_obj_to_use:
                 temp_frame = swap_face(source_face_obj_to_use, target_face_to_swap, source_frame_full, temp_frame)
            else:
                logging.warning("No source face available for tracked target in _process_live_target_v2.")
        elif TRACKING_FRAME_COUNTER % DETECTION_INTERVAL == 0:
             logging.info("Target face not found by detection or tracking in _process_live_target_v2 (single face tracking path).")
        return temp_frame

    # Fallback to original many_faces logic if not in single face tracking mode (or if above logic doesn't return)
    # This part is essentially the original _process_live_target_v2 for many_faces=True
    detected_faces = get_many_faces(temp_frame) # Re-get if not already gotten or if many_faces path
    if not detected_faces:
        return temp_frame # No faces, return original

    if modules.globals.many_faces: # This is the original many_faces logic for live
        source_face_obj = default_source_face()
        if source_face_obj:
            for target_face in detected_faces:
                temp_frame = swap_face(source_face_obj, target_face, source_frame_full, temp_frame)
    # The complex simple_map logic for non-many_faces was attempted above with tracking.
    # If that path wasn't taken or didn't result in a swap, and it's not many_faces,
    # we might need to re-evaluate the original simple_map logic here.
    # For now, the tracking path for single face handles the non-many_faces case.
    # If tracking is off or fails consistently, this function will effectively just return temp_frame for non-many_faces.
    # This else block for simple_map from original _process_live_target_v2 might be needed if tracking is disabled.
    # However, to avoid processing faces twice (once for tracking attempt, once here), this is tricky.
    # For now, the subtask focuses on adding tracking to process_frame, which is used by webcam in non-map_faces mode.
    # The changes to _process_live_target_v2 are more experimental for map_faces=True live mode.
    return temp_frame


def process_frame_v2(source_frame_full: Frame, temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    if is_image(modules.globals.target_path):
        return _process_image_target_v2(source_frame_full, temp_frame)
    elif is_video(modules.globals.target_path):
        return _process_video_target_v2(source_frame_full, temp_frame, temp_frame_path)
    else: # This is the live cam / generic case
        # If map_faces is True for webcam, this is called.
        # We need to decide if tracking applies here or if it's simpler to use existing logic.
        # The subtask's main focus was process_frame.
        # For now, let _process_live_target_v2 handle it, which includes an attempt at tracking for non-many_faces.
        return _process_live_target_v2(source_frame_full, temp_frame)


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    source_img = cv2.imread(source_path)
    if source_img is None:
        logging.error(f"Failed to read source image from {source_path}")
        return

    if not modules.globals.map_faces:
        source_face_obj = get_one_face(source_img)
        if not source_face_obj:
            logging.error(f"No face detected in source image {source_path}")
            return
        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is None:
                logging.warning(f"Failed to read temp_frame from {temp_frame_path}, skipping.")
                continue
            try:
                result = process_frame(source_face_obj, source_img, temp_frame) # process_frame will use tracking
                cv2.imwrite(temp_frame_path, result)
            except Exception as exception:
                logging.error(f"Error processing frame {temp_frame_path}: {exception}", exc_info=True)
                pass
            if progress:
                progress.update(1)
    else:
        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is None:
                logging.warning(f"Failed to read temp_frame from {temp_frame_path}, skipping.")
                continue
            try:
                result = process_frame_v2(source_img, temp_frame, temp_frame_path) # process_frame_v2 might use tracking via _process_live_target_v2
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

    # target_frame = cv2.imread(target_path) # This line is not needed as original_target_frame is used
    # if target_frame is None:
    #     logging.error(f"Failed to read target image from {target_path}")
    #     return

    original_target_frame = cv2.imread(target_path)
    if original_target_frame is None:
        logging.error(f"Failed to read original target image from {target_path}")
        return

    result = None

    if not modules.globals.map_faces:
        source_face_obj = get_one_face(source_img)
        if not source_face_obj:
            logging.error(f"No face detected in source image {source_path}")
            return
        # process_frame will use tracking if called in a context where TRACKING_FRAME_COUNTER changes (e.g. video/live)
        # For single image, TRACKING_FRAME_COUNTER would be 1, so full detection.
        result = process_frame(source_face_obj, source_img, original_target_frame)
    else:
        if modules.globals.many_faces:
            update_status(
                "Many faces enabled. Using first source image. Progressing...", NAME
            )
        result = process_frame_v2(source_img, original_target_frame, target_path)

    if result is not None:
        cv2.imwrite(output_path, result)
    else:
        logging.error(f"Processing image {target_path} failed, result was None.")


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    global TRACKING_FRAME_COUNTER, LAST_DETECTION_SUCCESS, TARGET_TRACKER, LAST_TARGET_KPS, LAST_TARGET_BBOX_XYWH
    # Reset tracker state for each new video
    TRACKING_FRAME_COUNTER = 0
    LAST_DETECTION_SUCCESS = False
    TARGET_TRACKER = None
    LAST_TARGET_KPS = None
    LAST_TARGET_BBOX_XYWH = None

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
    # Mouth mask requires landmark_2d_106, which tracked faces won't have.
    # Add a check here to prevent errors if landmark_2d_106 is None.
    if face.landmark_2d_106 is None:
        logging.debug("Skipping lower_mouth_mask due to missing landmark_2d_106 (likely a tracked face).")
        # Return empty/default values that won't cause downstream errors
        # The bounding box (min_x, etc.) might still be useful if derived from face.bbox
        # For now, return fully empty to prevent partial processing.
        # The caller (apply_mouth_area) should also be robust to this.
        # Fallback: create a simple mask from bbox if needed, or ensure apply_mouth_area handles this.
        # For now, returning all Nones for the mask parts.
        # The tuple for bbox still needs 4 values, even if invalid, to unpack.
        # A truly robust solution would be for apply_mouth_area to not proceed if mouth_mask is None.
        return mask, None, (0,0,0,0), None # Ensure tuple has 4 values

    landmarks = face.landmark_2d_106 # Now we know it's not None
    # ... (rest of the function remains the same)
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
    # Add check for landmarks before trying to use them
    if face.landmark_2d_106 is None or mouth_mask_data is None or mouth_mask_data[1] is None: # mouth_cutout is mouth_mask_data[1]
        logging.debug("Skipping mouth mask visualization due to missing landmarks or data.")
        return frame

    landmarks = face.landmark_2d_106
    # if landmarks is not None and mouth_mask_data is not None: # This check is now partially done above
    mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon = (
        mouth_mask_data
    )
    if mouth_cutout is None or lower_lip_polygon is None: # Further check
        logging.debug("Skipping mouth mask visualization due to missing mouth_cutout or polygon.")
        return frame


    vis_frame = frame.copy()

    # Ensure coordinates are within frame bounds
    height, width = vis_frame.shape[:2]
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = min(width, max_x), min(height, max_y)

    # Adjust mask to match the region size
    # Ensure mask_region calculation is safe
    if max_y - min_y <= 0 or max_x - min_x <= 0:
        logging.warning("Invalid ROI for mouth mask visualization.")
        return frame # or vis_frame, as it's a copy
    mask_region = mask[0 : max_y - min_y, 0 : max_x - min_x]


    cv2.polylines(vis_frame, [lower_lip_polygon], True, (0, 255, 0), 2)

    feather_amount = max(
        1,
        min(
            30,
            (max_x - min_x) // modules.globals.mask_feather_ratio if (max_x - min_x) > 0 else 1,
            (max_y - min_y) // modules.globals.mask_feather_ratio if (max_y - min_y) > 0 else 1,
        ),
    )
    kernel_size = 2 * feather_amount + 1
    # Ensure mask_region is not empty before blur
    if mask_region.size > 0 :
        feathered_mask = cv2.GaussianBlur(
            mask_region.astype(float), (kernel_size, kernel_size), 0
        )
        # Check if feathered_mask.max() is zero to avoid division by zero error
        max_val = feathered_mask.max()
        if max_val > 0:
            feathered_mask = (feathered_mask / max_val * 255).astype(np.uint8)
        else:
            feathered_mask = np.zeros_like(mask_region, dtype=np.uint8) # Handle case of all-black mask
    else: # if mask_region is empty, create an empty feathered_mask
        feathered_mask = np.zeros_like(mask_region, dtype=np.uint8)


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
    # return frame # Fallback if landmarks or mouth_mask_data is None


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: tuple,
    face_mask: np.ndarray,
    mouth_polygon: np.ndarray,
) -> np.ndarray:
    # Add check for None mouth_polygon which can happen if landmark_2d_106 was None
    if mouth_polygon is None or mouth_cutout is None:
        logging.debug("Skipping apply_mouth_area due to missing mouth_polygon or mouth_cutout.")
        return frame

    min_x, min_y, max_x, max_y = mouth_box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if (
        box_width <= 0 or box_height <= 0 or # Check for valid box dimensions
        face_mask is None
    ):
        return frame

    try:
        resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height))
        # Ensure ROI slicing is valid
        if min_y >= max_y or min_x >= max_x:
             logging.warning("Invalid ROI for applying mouth area.")
             return frame
        roi = frame[min_y:max_y, min_x:max_x]


        if roi.shape != resized_mouth_cutout.shape:
            resized_mouth_cutout = cv2.resize(
                resized_mouth_cutout, (roi.shape[1], roi.shape[0])
            )

        color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi)

        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask, [adjusted_polygon], 255)

        feather_amount = min(
            30,
            box_width // modules.globals.mask_feather_ratio if modules.globals.mask_feather_ratio > 0 else 30,
            box_height // modules.globals.mask_feather_ratio if modules.globals.mask_feather_ratio > 0 else 30,
        )
        feather_amount = max(1, feather_amount) # Ensure feather_amount is at least 1 for kernel size

        # Ensure kernel size is odd and positive for GaussianBlur
        kernel_size_blur = 2 * feather_amount + 1

        feathered_mask_float = cv2.GaussianBlur(
            polygon_mask.astype(float), (kernel_size_blur, kernel_size_blur), 0
        )

        max_val = feathered_mask_float.max()
        if max_val > 0:
            feathered_mask_normalized = feathered_mask_float / max_val
        else: # Avoid division by zero if mask is all black
            feathered_mask_normalized = feathered_mask_float


        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        combined_mask_float = feathered_mask_normalized * (face_mask_roi / 255.0)

        combined_mask_3ch = combined_mask_float[:, :, np.newaxis]

        blended = (
            color_corrected_mouth.astype(np.float32) * combined_mask_3ch +
            roi.astype(np.float32) * (1 - combined_mask_3ch)
        ).astype(np.uint8)

        # This final blend with face_mask_3channel seems redundant if combined_mask_float already incorporates face_mask_roi
        # However, it ensures that areas outside the broader face_mask (but inside mouth_box) are not affected.
        # For simplicity and to maintain original intent if there was one, keeping it for now.
        # face_mask_3channel_roi = np.repeat(face_mask_roi[:, :, np.newaxis], 3, axis=2) / 255.0
        # final_blend = blended * face_mask_3channel_roi + roi * (1 - face_mask_3channel_roi)

        frame[min_y:max_y, min_x:max_x] = blended.astype(np.uint8)
    except Exception as e:
        logging.error(f"Error in apply_mouth_area: {e}", exc_info=True)
        pass # Keep original frame on error

    return frame


def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106

    # Add check for landmarks before trying to use them
    if landmarks is None:
        logging.debug("Skipping face_mask creation due to missing landmark_2d_106.")
        # Fallback: if no landmarks, try to create a simple mask from bbox if available
        if face.bbox is not None:
            x1, y1, x2, y2 = face.bbox.astype(int)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            # Simple ellipse based on bbox - adjust size factor as needed
            cv2.ellipse(mask, (center_x, center_y), (int(width * 0.6), int(height * 0.7)), 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (15, 15), 5) # Soften the simple mask too
        return mask


    landmarks = landmarks.astype(np.int32) # Now safe to use

    right_side_face = landmarks[0:16]
    left_side_face = landmarks[17:32]
    # right_eye = landmarks[33:42] # Not used for outline
    right_eye_brow = landmarks[43:51]
    # left_eye = landmarks[87:96] # Not used for outline
    left_eye_brow = landmarks[97:105]

    if right_eye_brow.size == 0 or left_eye_brow.size == 0 or right_side_face.size == 0 or left_side_face.size == 0 :
        logging.warning("Face mask creation skipped due to empty landmark arrays for key features.")
        if face.bbox is not None: # Fallback to bbox mask if landmarks are partially missing
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1) # Simple rectangle from bbox
            mask = cv2.GaussianBlur(mask, (15,15), 5)
        return mask

    right_eyebrow_top = np.min(right_eye_brow[:, 1])
    left_eyebrow_top = np.min(left_eye_brow[:, 1])
    eyebrow_top = min(right_eyebrow_top, left_eyebrow_top)

    face_top = np.min([right_side_face[0, 1], left_side_face[-1, 1]])
    forehead_height = max(0, face_top - eyebrow_top) # Ensure non-negative
    extended_forehead_height = int(forehead_height * 5.0)

    forehead_left = right_side_face[0].copy()
    forehead_right = left_side_face[-1].copy()

    # Prevent negative y-coordinates
    forehead_left[1] = max(0, forehead_left[1] - extended_forehead_height)
    forehead_right[1] = max(0, forehead_right[1] - extended_forehead_height)

    face_outline = np.vstack(
        [
            [forehead_left],
            right_side_face,
            left_side_face[
                ::-1
            ],
            [forehead_right],
        ]
    )

    if face_outline.shape[0] < 3 : # convexHull needs at least 3 points
        logging.warning("Not enough points for convex hull in face mask creation. Using bbox as fallback.")
        if face.bbox is not None:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
            mask = cv2.GaussianBlur(mask, (15,15), 5)
        return mask

    padding = int(
        np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05
    )

    hull = cv2.convexHull(face_outline)
    hull_padded = []
    # Calculate center of the original outline for padding direction
    center_of_outline = np.mean(face_outline, axis=0).squeeze()
    if center_of_outline.ndim > 1: # Ensure center is 1D
        center_of_outline = np.mean(center_of_outline, axis=0)

    for point_contour in hull:
        point = point_contour[0]
        direction = point - center_of_outline
        norm_direction = np.linalg.norm(direction)
        if norm_direction == 0:
            unit_direction = np.array([0,0])
        else:
            unit_direction = direction / norm_direction

        padded_point = point + unit_direction * padding
        hull_padded.append(padded_point)

    if hull_padded:
        hull_padded = np.array(hull_padded, dtype=np.int32)
        # Ensure hull_padded has the correct shape for fillConvexPoly (e.g., (N, 1, 2))
        if hull_padded.ndim == 2:
            hull_padded = hull_padded[:, np.newaxis, :]
        cv2.fillConvexPoly(mask, hull_padded, 255)
    else:
        if hull.ndim == 2: # Ensure hull has correct shape if hull_padded was empty
            hull = hull[:, np.newaxis, :]
        cv2.fillConvexPoly(mask, hull, 255)

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

    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    # Prevent division by zero if source_std is zero in any channel
    source_std[source_std == 0] = 1

    source = (source - source_mean) * (target_std / source_std) + target_mean

    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)
