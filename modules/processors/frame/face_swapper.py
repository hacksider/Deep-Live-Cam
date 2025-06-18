from typing import Any, List, Optional, Tuple, Callable # Added Callable
import cv2
import insightface
import threading
import numpy as np
import modules.globals
import logging
import modules.processors.frame.core
# from modules.core import update_status # Removed import
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
import platform # Added for potential platform-specific tracker choices later, though KCF is cross-platform

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
LAST_TARGET_BBOX_XYWH: Optional[List[int]] = None
TRACKING_FRAME_COUNTER = 0
DETECTION_INTERVAL = 5  # Process every 5th frame for full detection
LAST_DETECTION_SUCCESS = False
PREV_GRAY_FRAME: Optional[np.ndarray] = None # For optical flow
# --- End Tracker State Variables ---

def reset_tracker_state():
    """Resets all global tracker state variables."""
    global TARGET_TRACKER, LAST_TARGET_KPS, LAST_TARGET_BBOX_XYWH
    global TRACKING_FRAME_COUNTER, LAST_DETECTION_SUCCESS, PREV_GRAY_FRAME

    TARGET_TRACKER = None
    LAST_TARGET_KPS = None
    LAST_TARGET_BBOX_XYWH = None
    TRACKING_FRAME_COUNTER = 0
    LAST_DETECTION_SUCCESS = False # Important to ensure first frame after reset does detection
    PREV_GRAY_FRAME = None
    logging.debug("Global tracker state has been reset.")


def pre_check() -> bool:
    # download_directory_path = abs_dir # Old line
    download_directory_path = models_dir # New line
    conditional_download(
        download_directory_path,
        [
            "https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx"
        ],
    )
    return True


def pre_start(status_fn_callback: Callable[[str, str], None]) -> bool:
    if not modules.globals.map_faces and not is_image(modules.globals.source_path):
        status_fn_callback("Select an image for source path.", NAME)
        return False
    elif not modules.globals.map_faces and not get_one_face(
        cv2.imread(modules.globals.source_path)
    ):
        status_fn_callback("No face in source path detected.", NAME)
        return False
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        status_fn_callback("Select an image or video for target path.", NAME)
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

        face_mask_for_mouth = create_face_mask(target_face, temp_frame)

        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
            create_lower_mouth_mask(target_face, temp_frame)
        )

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
    global TRACKING_FRAME_COUNTER, DETECTION_INTERVAL, LAST_DETECTION_SUCCESS, PREV_GRAY_FRAME

    if modules.globals.color_correction: # This should apply to temp_frame before gray conversion
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)

    current_gray_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)
    target_face_to_swap = None

    if modules.globals.many_faces:
        # Tracking logic is not applied for many_faces mode in this iteration
        # Revert to Nth frame detection for all faces in many_faces mode for now for performance
        TRACKING_FRAME_COUNTER += 1
        if TRACKING_FRAME_COUNTER % DETECTION_INTERVAL == 0:
            logging.debug(f"Frame {TRACKING_FRAME_COUNTER} (ManyFaces): Running full detection.")
            many_faces_detected = get_many_faces(temp_frame)
            if many_faces_detected:
                for target_face_data in many_faces_detected:
                    if source_face_obj and target_face_data:
                        temp_frame = swap_face(source_face_obj, target_face_data, source_frame_full, temp_frame)
            LAST_DETECTION_SUCCESS = bool(many_faces_detected) # Update based on if any face was found
        else:
            # For many_faces on non-detection frames, we currently don't have individual trackers.
            # The frame will pass through without additional swapping if we don't store and reuse old face data.
            # This means non-detection frames in many_faces mode might show unsynced swaps or no swaps if not handled.
            # For now, it means only Nth frame gets swaps in many_faces.
            logging.debug(f"Frame {TRACKING_FRAME_COUNTER} (ManyFaces): Skipping swap on intermediate frame.")
            pass
    else:
        # --- Single Face Mode with Tracking ---
        TRACKING_FRAME_COUNTER += 1

        if TRACKING_FRAME_COUNTER % DETECTION_INTERVAL == 0 or not LAST_DETECTION_SUCCESS:
            logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: Running full detection.")
            actual_target_face_data = get_one_face(temp_frame) # get_one_face returns a Face object or None
            if actual_target_face_data:
                target_face_to_swap = actual_target_face_data
                if actual_target_face_data.kps is not None:
                    LAST_TARGET_KPS = actual_target_face_data.kps.copy()
                else: # Should not happen with buffalo_l but good for robustness
                    LAST_TARGET_KPS = None

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
            if TARGET_TRACKER is not None and PREV_GRAY_FRAME is not None and LAST_TARGET_KPS is not None:
                logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: Attempting track.")
                success_tracker, new_bbox_xywh_float = TARGET_TRACKER.update(temp_frame)
                if success_tracker:
                    logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: KCF Tracking SUCCESS.")
                    new_bbox_xywh = [int(v) for v in new_bbox_xywh_float]

                    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    tracked_kps_float32 = LAST_TARGET_KPS.astype(np.float32) # Optical flow needs float32

                    new_kps_tracked, opt_flow_status, opt_flow_err = cv2.calcOpticalFlowPyrLK(
                        PREV_GRAY_FRAME, current_gray_frame, tracked_kps_float32, None, **lk_params
                    )

                    if new_kps_tracked is not None and opt_flow_status is not None:
                        good_new_kps = new_kps_tracked[opt_flow_status.ravel() == 1]
                        # good_old_kps_for_ref = tracked_kps_float32[opt_flow_status.ravel() == 1]

                        if len(good_new_kps) >= 3: # Need at least 3 points for stability
                            current_kps = good_new_kps
                            new_bbox_xyxy_np = np.array([
                                new_bbox_xywh[0],
                                new_bbox_xywh[1],
                                new_bbox_xywh[0] + new_bbox_xywh[2],
                                new_bbox_xywh[1] + new_bbox_xywh[3]
                            ], dtype=np.float32) # insightface Face expects float bbox

                            # Construct Face object (ensure all required fields are present, others None)
                            target_face_to_swap = Face(
                                bbox=new_bbox_xyxy_np,
                                kps=current_kps.astype(np.float32), # kps are float
                                det_score=0.90, # Indicate high confidence for tracked face
                                landmark_3d_68=None,
                                landmark_2d_106=None,
                                gender=None,
                                age=None,
                                embedding=None, # Not available from tracking
                                normed_embedding=None # Not available from tracking
                            )
                            LAST_TARGET_KPS = current_kps.copy()
                            LAST_TARGET_BBOX_XYWH = new_bbox_xywh
                            LAST_DETECTION_SUCCESS = True
                            logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: Optical Flow SUCCESS, {len(good_new_kps)} points tracked.")
                        else:
                            logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: Optical flow lost too many KPS ({len(good_new_kps)} found). Triggering re-detection.")
                            LAST_DETECTION_SUCCESS = False
                            TARGET_TRACKER = None
                    else:
                        logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: Optical flow calculation failed. Triggering re-detection.")
                        LAST_DETECTION_SUCCESS = False
                        TARGET_TRACKER = None
                else:
                    logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: KCF Tracking FAILED. Triggering re-detection.")
                    LAST_DETECTION_SUCCESS = False
                    TARGET_TRACKER = None
            else:
                logging.debug(f"Frame {TRACKING_FRAME_COUNTER}: No active tracker or prerequisite data. Skipping track.")
                # target_face_to_swap remains None

        if target_face_to_swap and source_face_obj:
            temp_frame = swap_face(source_face_obj, target_face_to_swap, source_frame_full, temp_frame)
        else:
            if TRACKING_FRAME_COUNTER % DETECTION_INTERVAL == 0 and not LAST_DETECTION_SUCCESS: # Only log if it was a detection attempt that failed
                logging.info("Target face not found by detection in process_frame.")

    PREV_GRAY_FRAME = current_gray_frame.copy() # Update for the next frame
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
    # It now uses the same Nth frame + tracking logic as process_frame for its single-face path.
    global TARGET_TRACKER, LAST_TARGET_KPS, LAST_TARGET_BBOX_XYWH
    global TRACKING_FRAME_COUNTER, DETECTION_INTERVAL, LAST_DETECTION_SUCCESS, PREV_GRAY_FRAME

    current_gray_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY) # Needed for optical flow

    if modules.globals.many_faces:
        # For many_faces in map_faces=True live mode, use existing logic (detect all, swap all with default source)
        # This part does not use the new tracking logic.
        TRACKING_FRAME_COUNTER += 1 # Still increment for consistency, though not strictly for Nth frame here
        if TRACKING_FRAME_COUNTER % DETECTION_INTERVAL == 0: # Optional: Nth frame for many_faces too
            detected_faces = get_many_faces(temp_frame)
            if detected_faces:
                source_face_obj = default_source_face()
                if source_face_obj:
                    for target_face in detected_faces:
                        temp_frame = swap_face(source_face_obj, target_face, source_frame_full, temp_frame)
        # On non-detection frames for many_faces, no swap occurs unless we cache all detected faces, which is complex.
    else: # Not many_faces (single face logic with tracking or simple_map)
        TRACKING_FRAME_COUNTER += 1
        target_face_to_swap = None

        if TRACKING_FRAME_COUNTER % DETECTION_INTERVAL == 0 or not LAST_DETECTION_SUCCESS:
            logging.debug(f"Frame {TRACKING_FRAME_COUNTER} (Live V2): Running full detection.")
            detected_faces = get_many_faces(temp_frame) # Get all faces
            actual_target_face_data = None

            if detected_faces:
                if modules.globals.simple_map and modules.globals.simple_map.get("target_embeddings") and modules.globals.simple_map["target_embeddings"][0] is not None:
                    # Try to find the "main" target face from simple_map's first entry
                    # This assumes the first simple_map entry is the one to track.
                    try:
                        closest_idx, _ = find_closest_centroid([face.normed_embedding for face in detected_faces], modules.globals.simple_map["target_embeddings"][0])
                        if closest_idx < len(detected_faces):
                            actual_target_face_data = detected_faces[closest_idx]
                    except Exception as e_centroid: # Broad exception for safety with list indexing
                        logging.warning(f"Error finding closest centroid for simple_map in live_v2: {e_centroid}")
                        actual_target_face_data = detected_faces[0] # Fallback
                else: # Fallback if no simple_map or if logic above fails
                    actual_target_face_data = detected_faces[0]

            if actual_target_face_data:
                target_face_to_swap = actual_target_face_data
                if actual_target_face_data.kps is not None:
                    LAST_TARGET_KPS = actual_target_face_data.kps.copy()
                else:
                    LAST_TARGET_KPS = None
                bbox_xyxy = actual_target_face_data.bbox
                LAST_TARGET_BBOX_XYWH = [int(bbox_xyxy[0]), int(bbox_xyxy[1]), int(bbox_xyxy[2] - bbox_xyxy[0]), int(bbox_xyxy[3] - bbox_xyxy[1])]
                try:
                    TARGET_TRACKER = cv2.TrackerKCF_create()
                    TARGET_TRACKER.init(temp_frame, tuple(LAST_TARGET_BBOX_XYWH))
                    LAST_DETECTION_SUCCESS = True
                except Exception as e:
                    logging.error(f"Failed to initialize tracker (Live V2): {e}", exc_info=True)
                    TARGET_TRACKER = None; LAST_DETECTION_SUCCESS = False
            else:
                LAST_DETECTION_SUCCESS = False; TARGET_TRACKER = None
        else: # Intermediate frame tracking
            if TARGET_TRACKER is not None and PREV_GRAY_FRAME is not None and LAST_TARGET_KPS is not None:
                success_tracker, new_bbox_xywh_float = TARGET_TRACKER.update(temp_frame)
                if success_tracker:
                    new_bbox_xywh = [int(v) for v in new_bbox_xywh_float]
                    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    tracked_kps_float32 = LAST_TARGET_KPS.astype(np.float32)
                    new_kps_tracked, opt_flow_status, _ = cv2.calcOpticalFlowPyrLK(PREV_GRAY_FRAME, current_gray_frame, tracked_kps_float32, None, **lk_params)

                    if new_kps_tracked is not None and opt_flow_status is not None:
                        good_new_kps = new_kps_tracked[opt_flow_status.ravel() == 1]
                        if len(good_new_kps) >= 3:
                            current_kps = good_new_kps
                            new_bbox_xyxy_np = np.array([new_bbox_xywh[0], new_bbox_xywh[1], new_bbox_xywh[0] + new_bbox_xywh[2], new_bbox_xywh[1] + new_bbox_xywh[3]], dtype=np.float32)
                            target_face_to_swap = Face(bbox=new_bbox_xyxy_np, kps=current_kps.astype(np.float32), det_score=0.90, landmark_3d_68=None, landmark_2d_106=None, gender=None, age=None, embedding=None, normed_embedding=None)
                            LAST_TARGET_KPS = current_kps.copy()
                            LAST_TARGET_BBOX_XYWH = new_bbox_xywh
                            LAST_DETECTION_SUCCESS = True
                        else: # Optical flow lost points
                            LAST_DETECTION_SUCCESS = False; TARGET_TRACKER = None
                    else: # Optical flow failed
                        LAST_DETECTION_SUCCESS = False; TARGET_TRACKER = None
                else: # KCF Tracker failed
                    LAST_DETECTION_SUCCESS = False; TARGET_TRACKER = None

        # Perform swap using the determined target_face_to_swap
        if target_face_to_swap:
            # Determine source face based on simple_map (if available and target_face_to_swap has embedding for matching)
            # This part requires target_face_to_swap to have 'normed_embedding' if we want to use simple_map matching.
            # Tracked faces currently don't have embedding. So, this will likely use default_source_face.
            source_face_obj_to_use = None
            if modules.globals.simple_map and modules.globals.simple_map.get("target_embeddings") and hasattr(target_face_to_swap, 'normed_embedding') and target_face_to_swap.normed_embedding is not None:
                 closest_centroid_index, _ = find_closest_centroid(modules.globals.simple_map["target_embeddings"], target_face_to_swap.normed_embedding)
                 if closest_centroid_index < len(modules.globals.simple_map["source_faces"]):
                     source_face_obj_to_use = modules.globals.simple_map["source_faces"][closest_centroid_index]

            if source_face_obj_to_use is None: # Fallback if no match or no embedding
                source_face_obj_to_use = default_source_face()

            if source_face_obj_to_use:
                temp_frame = swap_face(source_face_obj_to_use, target_face_to_swap, source_frame_full, temp_frame)
            else:
                logging.warning("No source face available for tracked/detected target in _process_live_target_v2 (single).")
        elif TRACKING_FRAME_COUNTER % DETECTION_INTERVAL == 0 and not LAST_DETECTION_SUCCESS:
             logging.info("Target face not found in _process_live_target_v2 (single face path).")

    PREV_GRAY_FRAME = current_gray_frame.copy()
    return temp_frame


def process_frame_v2(source_frame_full: Frame, temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    if is_image(modules.globals.target_path):
        return _process_image_target_v2(source_frame_full, temp_frame)
    elif is_video(modules.globals.target_path):
        # For video files with map_faces=True, use the original _process_video_target_v2
        # as tracking state management across distinct mapped faces is complex and not yet implemented.
        # The Nth frame + tracking is primarily for single face mode or live mode.
        return _process_video_target_v2(source_frame_full, temp_frame, temp_frame_path) # Original logic without tracking
    else: # This is the live cam / generic case (map_faces=True)
        return _process_live_target_v2(source_frame_full, temp_frame)


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    source_img = cv2.imread(source_path)
    if source_img is None:
        logging.error(f"Failed to read source image from {source_path}")
        return

    if not is_video(modules.globals.target_path): # Reset only if not a video (video handles it in process_video)
        reset_tracker_state()

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
                result = process_frame(source_face_obj, source_img, temp_frame)
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
                result = process_frame_v2(source_img, temp_frame, temp_frame_path)
                cv2.imwrite(temp_frame_path, result)
            except Exception as exception:
                logging.error(f"Error processing frame {temp_frame_path} with map_faces: {exception}", exc_info=True)
                pass
            if progress:
                progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str, status_fn_callback: Callable[[str, str], None]) -> None:
    source_img = cv2.imread(source_path)
    if source_img is None:
        logging.error(f"Failed to read source image from {source_path}")
        return

    original_target_frame = cv2.imread(target_path)
    if original_target_frame is None:
        logging.error(f"Failed to read original target image from {target_path}")
        return

    result = None

    reset_tracker_state() # Ensure fresh state for single image processing


    if not modules.globals.map_faces:
        source_face_obj = get_one_face(source_img)
        if not source_face_obj:
            logging.error(f"No face detected in source image {source_path}")
            return
        result = process_frame(source_face_obj, source_img, original_target_frame)
    else:
        if modules.globals.many_faces:
            status_fn_callback(
                "Many faces enabled. Using first source image. Progressing...", NAME
            )
        result = process_frame_v2(source_img, original_target_frame, target_path)

    if result is not None:
        cv2.imwrite(output_path, result)
    else:
        logging.error(f"Processing image {target_path} failed, result was None.")


def process_video(source_path: str, temp_frame_paths: List[str], status_fn_callback: Callable[[str, str], None]) -> None:
    reset_tracker_state() # Ensure fresh state for each video processing

    if modules.globals.map_faces and modules.globals.many_faces:
        status_fn_callback(
            "Many faces enabled. Using first source image. Progressing...", NAME
        )
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames
    )


def create_lower_mouth_mask(
    face: Face, frame: Frame
) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int, int], Optional[np.ndarray]]:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    lower_lip_polygon_details = None # Initialize to ensure it's always defined

    if face.landmark_2d_106 is None:
        logging.debug("Skipping lower_mouth_mask due to missing landmark_2d_106 (likely a tracked face).")
        return mask, None, (0,0,0,0), None

    landmarks = face.landmark_2d_106

    lower_lip_order = [
        65, 66, 62, 70, 69, 18, 19, 20, 21, 22,
        23, 24, 0,  8,  7,  6,  5,  4,  3,  2, 65,
    ]
    try:
        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.float32)
    except IndexError:
        logging.warning("Failed to get lower_lip_landmarks due to landmark indexing issue.")
        return mask, None, (0,0,0,0), None

    center = np.mean(lower_lip_landmarks, axis=0)
    expansion_factor = (1 + modules.globals.mask_down_size)
    expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center

    toplip_indices = [20, 0, 1, 2, 3, 4, 5]
    toplip_extension = (modules.globals.mask_size * 0.5)
    for idx in toplip_indices:
        direction = expanded_landmarks[idx] - center
        norm_direction = np.linalg.norm(direction)
        if norm_direction == 0: continue
        expanded_landmarks[idx] += (direction / norm_direction) * toplip_extension

    chin_indices = [11, 12, 13, 14, 15, 16]
    chin_extension = 2 * 0.2
    for idx in chin_indices:
        expanded_landmarks[idx][1] += (expanded_landmarks[idx][1] - center[1]) * chin_extension

    expanded_landmarks = expanded_landmarks.astype(np.int32)

    min_x, min_y = np.min(expanded_landmarks, axis=0)
    max_x, max_y = np.max(expanded_landmarks, axis=0)

    padding = int((max_x - min_x) * 0.1)
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(frame.shape[1] - 1, max_x + padding) # Ensure max_x is within bounds
    max_y = min(frame.shape[0] - 1, max_y + padding) # Ensure max_y is within bounds

    # Ensure min is less than max after adjustments
    if max_x <= min_x: max_x = min_x + 1
    if max_y <= min_y: max_y = min_y + 1

    # Ensure ROI dimensions are positive
    if max_y - min_y <= 0 or max_x - min_x <= 0:
        logging.warning(f"Invalid ROI for mouth mask creation: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
        return mask, None, (min_x, min_y, max_x, max_y), None # Return current min/max for bbox

    mask_roi = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    # Adjust landmarks to be relative to the ROI
    adjusted_landmarks = expanded_landmarks - [min_x, min_y]
    cv2.fillPoly(mask_roi, [adjusted_landmarks], 255)

    # Apply Gaussian blur to soften the mask edges
    # Ensure kernel size is odd and positive
    blur_kernel_size = (15, 15) # Make sure this is appropriate
    if blur_kernel_size[0] % 2 == 0: blur_kernel_size = (blur_kernel_size[0]+1, blur_kernel_size[1])
    if blur_kernel_size[1] % 2 == 0: blur_kernel_size = (blur_kernel_size[0], blur_kernel_size[1]+1)
    if blur_kernel_size[0] <=0 : blur_kernel_size = (1, blur_kernel_size[1])
    if blur_kernel_size[1] <=0 : blur_kernel_size = (blur_kernel_size[0], 1)

    mask_roi = cv2.GaussianBlur(mask_roi, blur_kernel_size, 5) # Sigma might also need tuning

    mask[min_y:max_y, min_x:max_x] = mask_roi
    mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()
    lower_lip_polygon_details = expanded_landmarks

    return mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon_details


def draw_mouth_mask_visualization(
    frame: Frame, face: Face, mouth_mask_data: tuple
) -> Frame:
    if face.landmark_2d_106 is None or mouth_mask_data is None or mouth_mask_data[1] is None:
        logging.debug("Skipping mouth mask visualization due to missing landmarks or data.")
        return frame

    mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon = mouth_mask_data
    if mouth_cutout is None or lower_lip_polygon is None:
        logging.debug("Skipping mouth mask visualization due to missing mouth_cutout or polygon.")
        return frame

    vis_frame = frame.copy()
    height, width = vis_frame.shape[:2]
    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = min(width, max_x), min(height, max_y)

    if max_y - min_y <= 0 or max_x - min_x <= 0:
        logging.warning("Invalid ROI for mouth mask visualization.")
        return vis_frame
    mask_region = mask[0 : max_y - min_y, 0 : max_x - min_x] # This line might be problematic if mask is full frame

    cv2.polylines(vis_frame, [lower_lip_polygon], True, (0, 255, 0), 2) # This uses original lower_lip_polygon coordinates

    # For displaying the mask itself, it's better to show the ROI where it was applied
    # or create a version of the mask that is full frame for visualization.
    # The current `mask_region` is a crop of the full `mask`.
    # Let's ensure we are visualizing the correct part or the full mask.
    # If `mask` is the full-frame mask, and `mask_region` was just for feathering calculation,
    # then we should use `mask` for display or a ROI from `mask`.

    # To make vis_frame part where mask is applied red (for example):
    # vis_frame_roi = vis_frame[min_y:max_y, min_x:max_x]
    # boolean_mask_roi = mask[min_y:max_y, min_x:max_x] > 127 # Assuming mask is full frame
    # if vis_frame_roi.shape[:2] == boolean_mask_roi.shape:
    #    vis_frame_roi[boolean_mask_roi] = [0,0,255] # Red where mask is active

    # The existing feathering logic for visualization:
    feather_amount = max(1, min(30,
        (max_x - min_x) // modules.globals.mask_feather_ratio if (max_x - min_x) > 0 and modules.globals.mask_feather_ratio > 0 else 1,
        (max_y - min_y) // modules.globals.mask_feather_ratio if (max_y - min_y) > 0 and modules.globals.mask_feather_ratio > 0 else 1
    ))
    kernel_size = 2 * feather_amount + 1

    # Assuming mask_region was correctly extracted for visualization purposes (e.g., a crop of the mask)
    # If mask_region is intended to be the mask that was applied, its size should match the ROI.
    if mask_region.size > 0 and mask_region.shape[0] == (max_y-min_y) and mask_region.shape[1] == (max_x-min_x):
        feathered_mask_vis = cv2.GaussianBlur(mask_region.astype(float), (kernel_size, kernel_size), 0)
        max_val = feathered_mask_vis.max()
        if max_val > 0: feathered_mask_vis = (feathered_mask_vis / max_val * 255).astype(np.uint8)
        else: feathered_mask_vis = np.zeros_like(mask_region, dtype=np.uint8)

        # Create a 3-channel version of the feathered mask for overlay if desired
        # feathered_mask_vis_3ch = cv2.cvtColor(feathered_mask_vis, cv2.COLOR_GRAY2BGR)
        # vis_frame_roi = vis_frame[min_y:max_y, min_x:max_x]
        # blended_roi = cv2.addWeighted(vis_frame_roi, 0.7, feathered_mask_vis_3ch, 0.3, 0)
        # vis_frame[min_y:max_y, min_x:max_x] = blended_roi
    else:
        # If mask_region is not what we expect, log or handle.
        # For now, we'll skip drawing the feathered_mask part if dimensions mismatch.
        logging.debug("Skipping feathered mask visualization part due to mask_region issues.")


    cv2.putText(vis_frame, "Lower Mouth Mask (Polygon)", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # cv2.putText(vis_frame, "Feathered Mask (Visualization)", (min_x, max_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # Optional text

    return vis_frame


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: tuple,
    face_mask: np.ndarray,
    mouth_polygon: np.ndarray,
) -> np.ndarray:
    if mouth_polygon is None or mouth_cutout is None:
        logging.debug("Skipping apply_mouth_area due to missing mouth_polygon or mouth_cutout.")
        return frame

    min_x, min_y, max_x, max_y = mouth_box
    box_width = max_x - min_x
    box_height = max_y - min_y

    if box_width <= 0 or box_height <= 0 or face_mask is None:
        logging.debug(f"Skipping apply_mouth_area due to invalid box dimensions or missing face_mask. W:{box_width} H:{box_height}")
        return frame

    try:
        # Ensure ROI is valid before attempting to access frame data
        if min_y >= max_y or min_x >= max_x:
             logging.warning(f"Invalid ROI for applying mouth area: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")
             return frame

        roi = frame[min_y:max_y, min_x:max_x]

        # Resize mouth_cutout to match the ROI dimensions if they differ
        if roi.shape[:2] != mouth_cutout.shape[:2]:
            resized_mouth_cutout = cv2.resize(mouth_cutout, (roi.shape[1], roi.shape[0]))
        else:
            resized_mouth_cutout = mouth_cutout

        color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi)

        # Create polygon_mask for the ROI
        polygon_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        cv2.fillPoly(polygon_mask, [adjusted_polygon.astype(np.int32)], 255) # Ensure polygon points are int32

        # Calculate feathering based on ROI dimensions
        feather_amount = max(1, min(30,
            roi.shape[1] // modules.globals.mask_feather_ratio if modules.globals.mask_feather_ratio > 0 else 30,
            roi.shape[0] // modules.globals.mask_feather_ratio if modules.globals.mask_feather_ratio > 0 else 30
        ))
        kernel_size_blur = 2 * feather_amount + 1 # Ensure it's odd
        if kernel_size_blur <= 0: kernel_size_blur = 1 # Ensure positive

        feathered_mask_float = cv2.GaussianBlur(polygon_mask.astype(float), (kernel_size_blur, kernel_size_blur), 0)

        max_val = feathered_mask_float.max()
        feathered_mask_normalized = feathered_mask_float / max_val if max_val > 0 else feathered_mask_float

        # Ensure face_mask_roi matches dimensions of feathered_mask_normalized
        face_mask_roi = face_mask[min_y:max_y, min_x:max_x]
        if face_mask_roi.shape != feathered_mask_normalized.shape:
            face_mask_roi = cv2.resize(face_mask_roi, (feathered_mask_normalized.shape[1], feathered_mask_normalized.shape[0]))
            logging.warning("Resized face_mask_roi to match feathered_mask_normalized in apply_mouth_area.")


        combined_mask_float = feathered_mask_normalized * (face_mask_roi / 255.0)
        combined_mask_3ch = combined_mask_float[:, :, np.newaxis] # Ensure broadcasting for 3 channels

        # Ensure all inputs to blending are float32 for precision, then convert back to uint8
        blended_float = (
            color_corrected_mouth.astype(np.float32) * combined_mask_3ch +
            roi.astype(np.float32) * (1.0 - combined_mask_3ch) # Ensure 1.0 for float subtraction
        )
        blended = np.clip(blended_float, 0, 255).astype(np.uint8)

        frame[min_y:max_y, min_x:max_x] = blended
    except Exception as e:
        logging.error(f"Error in apply_mouth_area: {e}", exc_info=True)

    return frame


def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106

    if landmarks is None:
        logging.debug("Face landmarks (landmark_2d_106) not available for face mask creation (likely tracked face). Using bbox as fallback.")
        if face.bbox is not None:
            x1, y1, x2, y2 = face.bbox.astype(int)
            # Ensure coordinates are within frame boundaries
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
            if x1 < x2 and y1 < y2:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1
                cv2.ellipse(mask, (center_x, center_y), (int(width * 0.6), int(height * 0.7)), 0, 0, 360, 255, -1)
                # Ensure kernel size is odd and positive for GaussianBlur
                blur_kernel_size_face = (15,15) # Example, can be tuned
                if blur_kernel_size_face[0] % 2 == 0: blur_kernel_size_face = (blur_kernel_size_face[0]+1, blur_kernel_size_face[1])
                if blur_kernel_size_face[1] % 2 == 0: blur_kernel_size_face = (blur_kernel_size_face[0], blur_kernel_size_face[1]+1)
                if blur_kernel_size_face[0] <=0 : blur_kernel_size_face = (1, blur_kernel_size_face[1])
                if blur_kernel_size_face[1] <=0 : blur_kernel_size_face = (blur_kernel_size_face[0], 1)
                mask = cv2.GaussianBlur(mask, blur_kernel_size_face, 5)
        return mask

    landmarks = landmarks.astype(np.int32)
    right_side_face = landmarks[0:16]
    left_side_face = landmarks[17:32]
    right_eye_brow = landmarks[43:51]
    left_eye_brow = landmarks[97:105]

    if right_eye_brow.size == 0 or left_eye_brow.size == 0 or right_side_face.size == 0 or left_side_face.size == 0 :
        logging.warning("Face mask creation skipped due to empty landmark arrays for key features.")
        if face.bbox is not None:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
            # Ensure kernel size is odd and positive for GaussianBlur
            blur_kernel_size_face_fallback = (15,15)
            if blur_kernel_size_face_fallback[0] % 2 == 0: blur_kernel_size_face_fallback = (blur_kernel_size_face_fallback[0]+1, blur_kernel_size_face_fallback[1])
            if blur_kernel_size_face_fallback[1] % 2 == 0: blur_kernel_size_face_fallback = (blur_kernel_size_face_fallback[0], blur_kernel_size_face_fallback[1]+1)
            if blur_kernel_size_face_fallback[0] <=0 : blur_kernel_size_face_fallback = (1, blur_kernel_size_face_fallback[1])
            if blur_kernel_size_face_fallback[1] <=0 : blur_kernel_size_face_fallback = (blur_kernel_size_face_fallback[0], 1)
            mask = cv2.GaussianBlur(mask, blur_kernel_size_face_fallback, 5)
        return mask

    right_eyebrow_top = np.min(right_eye_brow[:, 1])
    left_eyebrow_top = np.min(left_eye_brow[:, 1])
    eyebrow_top = min(right_eyebrow_top, left_eyebrow_top)

    face_top = np.min([right_side_face[0, 1], left_side_face[-1, 1]])
    forehead_height = max(0, face_top - eyebrow_top)
    extended_forehead_height = int(forehead_height * 5.0)

    forehead_left = right_side_face[0].copy()
    forehead_right = left_side_face[-1].copy()

    forehead_left[1] = max(0, forehead_left[1] - extended_forehead_height)
    forehead_right[1] = max(0, forehead_right[1] - extended_forehead_height)

    face_outline = np.vstack(
        [
            [forehead_left], right_side_face, left_side_face[::-1], [forehead_right],
        ]
    )

    if face_outline.shape[0] < 3 :
        logging.warning("Not enough points for convex hull in face mask creation. Using bbox as fallback.")
        if face.bbox is not None:
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1)
            # Ensure kernel size is odd and positive for GaussianBlur
            blur_kernel_size_face_hull_fallback = (15,15)
            if blur_kernel_size_face_hull_fallback[0] % 2 == 0: blur_kernel_size_face_hull_fallback = (blur_kernel_size_face_hull_fallback[0]+1, blur_kernel_size_face_hull_fallback[1])
            if blur_kernel_size_face_hull_fallback[1] % 2 == 0: blur_kernel_size_face_hull_fallback = (blur_kernel_size_face_hull_fallback[0], blur_kernel_size_face_hull_fallback[1]+1)
            if blur_kernel_size_face_hull_fallback[0] <=0 : blur_kernel_size_face_hull_fallback = (1, blur_kernel_size_face_hull_fallback[1])
            if blur_kernel_size_face_hull_fallback[1] <=0 : blur_kernel_size_face_hull_fallback = (blur_kernel_size_face_hull_fallback[0], 1)
            mask = cv2.GaussianBlur(mask, blur_kernel_size_face_hull_fallback, 5)
        return mask

    padding = int(np.linalg.norm(right_side_face[0] - left_side_face[-1]) * 0.05)
    hull = cv2.convexHull(face_outline)
    hull_padded = []

    center_of_outline = np.mean(face_outline, axis=0).squeeze()
    if center_of_outline.ndim > 1:
        center_of_outline = np.mean(center_of_outline, axis=0) # Ensure center_of_outline is 1D

    for point_contour in hull:
        point = point_contour[0]
        direction = point - center_of_outline
        norm_direction = np.linalg.norm(direction)
        if norm_direction == 0: unit_direction = np.array([0,0], dtype=float) # Ensure float for multiplication
        else: unit_direction = direction / norm_direction

        padded_point = point + unit_direction * padding
        hull_padded.append(padded_point)

    if hull_padded:
        hull_padded_np = np.array(hull_padded, dtype=np.int32)
        # cv2.fillConvexPoly expects a 2D array for points, or 3D with shape (N,1,2)
        if hull_padded_np.ndim == 3 and hull_padded_np.shape[1] == 1: # Already (N,1,2)
             cv2.fillConvexPoly(mask, hull_padded_np, 255)
        elif hull_padded_np.ndim == 2: # Shape (N,2)
             cv2.fillConvexPoly(mask, hull_padded_np[:, np.newaxis, :], 255) # Reshape to (N,1,2)
        else: # Fallback if shape is unexpected
            logging.warning("Unexpected shape for hull_padded in create_face_mask. Using raw hull.")
            if hull.ndim == 2: hull = hull[:,np.newaxis,:] # Ensure hull is (N,1,2)
            cv2.fillConvexPoly(mask, hull, 255)
    else:
        # Fallback to raw hull if hull_padded is empty for some reason
        if hull.ndim == 2: hull = hull[:,np.newaxis,:] # Ensure hull is (N,1,2)
        cv2.fillConvexPoly(mask, hull, 255)

    # Ensure kernel size is odd and positive for GaussianBlur
    blur_kernel_size_face_final = (5,5)
    if blur_kernel_size_face_final[0] % 2 == 0: blur_kernel_size_face_final = (blur_kernel_size_face_final[0]+1, blur_kernel_size_face_final[1])
    if blur_kernel_size_face_final[1] % 2 == 0: blur_kernel_size_face_final = (blur_kernel_size_face_final[0], blur_kernel_size_face_final[1]+1)
    if blur_kernel_size_face_final[0] <=0 : blur_kernel_size_face_final = (1, blur_kernel_size_face_final[1])
    if blur_kernel_size_face_final[1] <=0 : blur_kernel_size_face_final = (blur_kernel_size_face_final[0], 1)
    mask = cv2.GaussianBlur(mask, blur_kernel_size_face_final, 3)
    return mask


def apply_color_transfer(source, target):
    # Ensure inputs are not empty
    if source is None or source.size == 0 or target is None or target.size == 0:
        logging.warning("Color transfer skipped due to empty source or target image.")
        return source # Or target, depending on desired behavior for empty inputs

    try:
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)

        source_mean = source_mean.reshape((1, 1, 3))
        source_std = source_std.reshape((1, 1, 3))
        target_mean = target_mean.reshape((1, 1, 3))
        target_std = target_std.reshape((1, 1, 3))

        # Avoid division by zero if source_std is zero
        source_std[source_std == 0] = 1e-6 # A small epsilon instead of 1 to avoid large scaling if target_std is also small

        adjusted_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean
        adjusted_lab = np.clip(adjusted_lab, 0, 255) # Clip values to be within valid range for LAB

        result_bgr = cv2.cvtColor(adjusted_lab.astype("uint8"), cv2.COLOR_LAB2BGR)
    except cv2.error as e:
        logging.error(f"OpenCV error in apply_color_transfer: {e}", exc_info=True)
        return source # Return original source on error
    except Exception as e:
        logging.error(f"Unexpected error in apply_color_transfer: {e}", exc_info=True)
        return source # Return original source on error

    return result_bgr
