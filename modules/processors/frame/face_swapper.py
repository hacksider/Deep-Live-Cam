from typing import Any, List
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
    is_image,
    is_video,
)
from modules.cluster_analysis import find_closest_centroid
# Removed modules.globals.face_swapper_enabled - assuming controlled elsewhere or implicitly true if used
# Removed modules.globals.opacity - accessed via getattr
import os

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-SWAPPER"

# --- START: Added for Interpolation ---
PREVIOUS_FRAME_RESULT = None # Stores the final processed frame from the previous step
# --- END: Added for Interpolation ---

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

def pre_check() -> bool:
    """Ensure required model is downloaded."""
    download_directory_path = models_dir
    conditional_download(
        download_directory_path,
        [
            "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx"
        ],
    )
    return True


def pre_start() -> bool:
    """Check if source and target paths are valid before starting."""
    if not modules.globals.map_faces and not is_image(modules.globals.source_path):
        update_status("Select an image for source path.", NAME)
        return False
    elif not modules.globals.map_faces and not get_one_face(
        cv2.imread(modules.globals.source_path)
    ):
        update_status("No face in source path detected.", NAME)
        return False

    # Try to get the face swapper to ensure it loads correctly
    if get_face_swapper() is None:
        # Error message already printed within get_face_swapper
        return False

    # Add other essential checks if needed, e.g., target/source path validity
    return True


def get_face_swapper() -> Any:
    """Thread-safe singleton loader for the face swapper model."""
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_name = "inswapper_128.onnx"
            if "CUDAExecutionProvider" in modules.globals.execution_providers:
                model_name = "inswapper_128_fp16.onnx"
            model_path = os.path.join(models_dir, model_name)
            update_status(f"Loading face swapper model from: {model_path}", NAME)
            try:
                # Ensure the providers list is correctly passed
                # Apply CoreML optimization for Mac systems
                FACE_SWAPPER = insightface.model_zoo.get_model(
                    model_path,
                    providers=[
                        (
                            (
                                "CoreMLExecutionProvider",
                                {
                                    "ModelFormat": "MLProgram",
                                    "MLComputeUnits": "CPUAndGPU",
                                    "SpecializationStrategy": "FastPrediction",
                                    "AllowLowPrecisionAccumulationOnGPU": 1,
                                },
                            )
                            if p == "CoreMLExecutionProvider"
                            else p
                        )
                        for p in modules.globals.execution_providers
                    ],
                )
                update_status("Face swapper model loaded successfully.", NAME)
            except Exception as e:
                update_status(f"Error loading face swapper model: {e}", NAME)
                # print traceback maybe?
                # import traceback
                # traceback.print_exc()
                FACE_SWAPPER = None # Ensure it remains None on failure
                return None
    return FACE_SWAPPER


def swap_face(source_face: Any, target_face: Any, temp_frame: Any) -> Any:
    """Swap source_face onto target_face in temp_frame, with improved Poisson blending and optional mouth region blending."""
    face_swapper = get_face_swapper()
    try:
        face_swapper = get_face_swapper()
        swapped_frame = face_swapper.get(
            temp_frame, target_face, source_face, paste_back=True
        )
        if modules.globals.color_correction:
            mask = create_face_mask(target_face, temp_frame)
            # Find the center of the mask for seamlessClone
            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) > 0 and len(y_indices) > 0:
                center_x = int(np.mean(x_indices))
                center_y = int(np.mean(y_indices))
                center = (center_x, center_y)
                # Use seamlessClone for Poisson blending
                swapped_frame = cv2.seamlessClone(
                    swapped_frame, temp_frame, mask, center, cv2.NORMAL_CLONE
                )
        # --- Mouth region blending (optional, after Poisson blending) ---
        if hasattr(modules.globals, "mouth_mask") and modules.globals.mouth_mask:
            # Extract mouth region from the original frame
            mouth_mask_data = create_lower_mouth_mask(target_face, temp_frame)
            if mouth_mask_data is not None:
                mask, mouth_cutout, mouth_box, mouth_polygon = mouth_mask_data
                face_mask = create_face_mask(target_face, temp_frame)
                swapped_frame = apply_mouth_area(
                    swapped_frame, mouth_cutout, mouth_box, face_mask, mouth_polygon
                )
        return swapped_frame
    except Exception as e:
        logging.error(f"Face swap failed: {e}")
        return temp_frame


def process_frame(source_face: Any, temp_frame: Any) -> Any:
    """Process a single frame for face swapping."""
    if modules.globals.many_faces:
        many_faces = get_many_faces(processed_frame)
        if many_faces:
            current_swap_target = processed_frame.copy() # Apply swaps sequentially on a copy
            for target_face in many_faces:
                if source_face and target_face:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
                else:
                    logging.warning("Face detection failed for target/source.")
    else:
        target_face = get_one_face(processed_frame)
        if target_face:
            processed_frame = swap_face(source_face, target_face, processed_frame)
            if target_face is not None and hasattr(target_face, "bbox") and target_face.bbox is not None:
                    swapped_face_bboxes.append(target_face.bbox.astype(int))

    # Apply sharpening and interpolation
    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)

    return final_frame

def process_frame_v2(temp_frame: Any, temp_frame_path: str = "") -> Any:
    """Process a frame using mapped faces (for mapped face mode)."""
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
        # Live stream or webcam processing (analyze faces on the fly)
        detected_faces = get_many_faces(processed_frame)
        if detected_faces:
            if modules.globals.many_faces:
                 source_face = default_source_face() # Use default source for all detected targets
                 if source_face:
                     for target_face in detected_faces:
                        source_target_pairs.append((source_face, target_face))
            elif simple_map:
                # Use simple_map (source_faces <-> target_embeddings)
                source_faces = simple_map.get("source_faces", [])
                target_embeddings = simple_map.get("target_embeddings", [])

                if source_faces and target_embeddings and len(source_faces) == len(target_embeddings):
                     # Match detected faces to the closest target embedding
                     if len(detected_faces) <= len(target_embeddings):
                          # More targets defined than detected - match each detected face
                          for detected_face in detected_faces:
                              if detected_face.normed_embedding is None: continue
                              closest_idx, _ = find_closest_centroid(target_embeddings, detected_face.normed_embedding)
                              if 0 <= closest_idx < len(source_faces):
                                  source_target_pairs.append((source_faces[closest_idx], detected_face))
                     else:
                          # More faces detected than targets defined - match each target embedding to closest detected face
                          detected_embeddings = [f.normed_embedding for f in detected_faces if f.normed_embedding is not None]
                          detected_faces_with_embedding = [f for f in detected_faces if f.normed_embedding is not None]
                          if not detected_embeddings: return processed_frame # No embeddings to match

                          for i, target_embedding in enumerate(target_embeddings):
                              if 0 <= i < len(source_faces): # Ensure source face exists for this embedding
                                 closest_idx, _ = find_closest_centroid(detected_embeddings, target_embedding)
                                 if 0 <= closest_idx < len(detected_faces_with_embedding):
                                     source_target_pairs.append((source_faces[i], detected_faces_with_embedding[closest_idx]))
            else: # Fallback: if no map, use default source for the single detected face (if any)
                source_face = default_source_face()
                target_face = get_one_face(processed_frame, detected_faces) # Use faces already detected
                if source_face and target_face:
                    source_target_pairs.append((source_face, target_face))


    # Perform swaps based on the collected pairs
    current_swap_target = processed_frame.copy() # Apply swaps sequentially
    for source_face, target_face in source_target_pairs:
        if source_face and target_face:
            current_swap_target = swap_face(source_face, target_face, current_swap_target)
            if target_face is not None and hasattr(target_face, "bbox") and target_face.bbox is not None:
                swapped_face_bboxes.append(target_face.bbox.astype(int))
    processed_frame = current_swap_target # Assign final result


    # Apply sharpening and interpolation
    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)

    return final_frame


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """Process a list of frames for face swapping, updating progress and handling errors."""
    if not modules.globals.map_faces:
        source_face = get_one_face(cv2.imread(source_path))
        if source_face is None:
            logging.warning("No face detected in source image. Skipping all frames.")
            if progress:
                for _ in temp_frame_paths:
                    progress.update(1)
            return
        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            try:
                result = process_frame(source_face, temp_frame)
                if np.array_equal(result, temp_frame):
                    logging.warning(f"No face detected in target frame: {temp_frame_path}. Skipping write.")
                else:
                    cv2.imwrite(temp_frame_path, result)
            except Exception as exception:
                logging.error(f"Frame processing failed: {exception}")
            finally:
                if progress:
                    progress.update(1)
    else:
        for temp_frame_path in temp_frame_paths:
            temp_frame = cv2.imread(temp_frame_path)
            try:
                result = process_frame_v2(temp_frame, temp_frame_path)
                if np.array_equal(result, temp_frame):
                    logging.warning(f"No face detected in mapped target frame: {temp_frame_path}. Skipping write.")
                else:
                    cv2.imwrite(temp_frame_path, result)
            except Exception as exception:
                logging.error(f"Frame processing failed: {exception}")
            finally:
                if progress:
                    progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> bool:
    """Process a single image and return True if successful, False if no face detected."""
    if not modules.globals.map_faces:
        source_face = get_one_face(cv2.imread(source_path))
        if source_face is None:
            logging.warning("No face detected in source image. Skipping output.")
            return False
        target_frame = cv2.imread(target_path)
        result = process_frame(source_face, target_frame)
        if np.array_equal(result, target_frame):
            logging.warning("No face detected in target image. Skipping output.")
            return False
        cv2.imwrite(output_path, result)
        return True
    else:
        if modules.globals.many_faces:
            update_status(
                "Many faces enabled. Using first source image. Progressing...", NAME
            )
        target_frame = cv2.imread(target_path)
        result = process_frame_v2(target_frame)
        if np.array_equal(result, target_frame):
            logging.warning("No face detected in mapped target image. Skipping output.")
            return False
        cv2.imwrite(output_path, result)
        return True


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Sets up and calls the frame processing for video."""
    # --- Reset interpolation state before starting video processing ---
    global PREVIOUS_FRAME_RESULT
    PREVIOUS_FRAME_RESULT = None
    # ---

    mode_desc = "'map_faces'" if getattr(modules.globals, "map_faces", False) else "'simple'"
    if getattr(modules.globals, "map_faces", False) and getattr(modules.globals, "many_faces", False):
        mode_desc += " and 'many_faces'. Using pre-analysis map."
    update_status(f"Processing video with {mode_desc} mode.", NAME)

    # Pass the correct source_path (needed for simple mode in process_frames)
    # The core processing logic handles calling the right frame function (process_frames)
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames # Pass the newly modified process_frames
    )

# ==========================
# MASKING FUNCTIONS (Mostly unchanged, added safety checks and minor improvements)
# ==========================

def color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    s_mean, s_std = cv2.meanStdDev(source_lab)
    t_mean, t_std = cv2.meanStdDev(target_lab)
    s_mean = s_mean.reshape(1, 1, 3)
    s_std = s_std.reshape(1, 1, 3)
    t_mean = t_mean.reshape(1, 1, 3)
    t_std = t_std.reshape(1, 1, 3)
    result = (source_lab - s_mean) * (t_std / (s_std + 1e-6)) + t_mean
    result = np.clip(result, 0, 255).astype("uint8")
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def create_lower_mouth_mask(face, frame: np.ndarray):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mouth_cutout = None
    lower_lip_polygon = None # Initialize
    mouth_box = (0,0,0,0) # Initialize

    # Validate face and landmarks
    if face is None or not hasattr(face, 'landmark_2d_106'):
        # print("Warning: Invalid face object passed to create_lower_mouth_mask.")
        return mask, mouth_cutout, mouth_box, lower_lip_polygon

    landmarks = face.landmark_2d_106

    # Check landmark validity
    if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape[0] < 106:
        # print("Warning: Invalid or insufficient landmarks for mouth mask.")
        return mask, mouth_cutout, mouth_box, lower_lip_polygon

    try: # Wrap main logic in try-except
        #                  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
        lower_lip_order = [65, 66, 62, 70, 69, 18, 19, 20, 21, 22, 23, 24, 0, 8, 7, 6, 5, 4, 3, 2, 65] # 21 points

        # Check if all indices are valid for the loaded landmarks (already partially done by < 106 check)
        if max(lower_lip_order) >= landmarks.shape[0]:
            # print(f"Warning: Landmark index {max(lower_lip_order)} out of bounds for shape {landmarks.shape[0]}.")
            return mask, mouth_cutout, mouth_box, lower_lip_polygon

        lower_lip_landmarks = landmarks[lower_lip_order].astype(np.float32)

        # Filter out potential NaN or Inf values in landmarks
        if not np.all(np.isfinite(lower_lip_landmarks)):
            # print("Warning: Non-finite values detected in lower lip landmarks.")
            return mask, mouth_cutout, mouth_box, lower_lip_polygon

        center = np.mean(lower_lip_landmarks, axis=0)
        if not np.all(np.isfinite(center)): # Check center calculation
            # print("Warning: Could not calculate valid center for mouth mask.")
            return mask, mouth_cutout, mouth_box, lower_lip_polygon


        mask_down_size = getattr(modules.globals, "mask_down_size", 0.1) # Default 0.1
        expansion_factor = 1 + mask_down_size
        expanded_landmarks = (lower_lip_landmarks - center) * expansion_factor + center

        mask_size = getattr(modules.globals, "mask_size", 1.0) # Default 1.0
        toplip_extension = mask_size * 0.5

        # Define toplip indices relative to lower_lip_order (safer)
        toplip_local_indices = [0, 1, 2, 3, 4, 5, 19] # Indices in lower_lip_order for [65, 66, 62, 70, 69, 18, 2]

        for idx in toplip_local_indices:
            if idx < len(expanded_landmarks): # Boundary check
                direction = expanded_landmarks[idx] - center
                norm = np.linalg.norm(direction)
                if norm > 1e-6: # Avoid division by zero
                   direction_normalized = direction / norm
                   expanded_landmarks[idx] += direction_normalized * toplip_extension

        # Define chin indices relative to lower_lip_order
        chin_local_indices = [9, 10, 11, 12, 13, 14] # Indices for [22, 23, 24, 0, 8, 7]
        chin_extension = 2 * 0.2

        for idx in chin_local_indices:
            if idx < len(expanded_landmarks): # Boundary check
               # Extend vertically based on distance from center y
               y_diff = expanded_landmarks[idx][1] - center[1]
               expanded_landmarks[idx][1] += y_diff * chin_extension


        # Ensure landmarks are finite after adjustments
        if not np.all(np.isfinite(expanded_landmarks)):
            # print("Warning: Non-finite values detected after expanding landmarks.")
            return mask, mouth_cutout, mouth_box, lower_lip_polygon

        expanded_landmarks = expanded_landmarks.astype(np.int32)

        min_x, min_y = np.min(expanded_landmarks, axis=0)
        max_x, max_y = np.max(expanded_landmarks, axis=0)

        # Add padding *after* initial min/max calculation
        padding_ratio = 0.1 # Percentage padding
        padding_x = int((max_x - min_x) * padding_ratio)
        padding_y = int((max_y - min_y) * padding_ratio) # Use y-range for y-padding

        # Apply padding and clamp to frame boundaries
        frame_h, frame_w = frame.shape[:2]
        min_x = max(0, min_x - padding_x)
        min_y = max(0, min_y - padding_y)
        max_x = min(frame_w, max_x + padding_x)
        max_y = min(frame_h, max_y + padding_y)


        if max_x > min_x and max_y > min_y:
            # Create the mask ROI
            mask_roi_h = max_y - min_y
            mask_roi_w = max_x - min_x
            mask_roi = np.zeros((mask_roi_h, mask_roi_w), dtype=np.uint8)

            # Shift polygon coordinates relative to the ROI's top-left corner
            polygon_relative_to_roi = expanded_landmarks - [min_x, min_y]

            # Draw polygon on the ROI mask
            cv2.fillPoly(mask_roi, [polygon_relative_to_roi], 255)

            # Apply Gaussian blur (ensure kernel size is odd and positive)
            blur_k_size = getattr(modules.globals, "mask_blur_kernel", 15) # Default 15
            blur_k_size = max(1, blur_k_size // 2 * 2 + 1) # Ensure odd
            mask_roi = cv2.GaussianBlur(mask_roi, (blur_k_size, blur_k_size), 0) # Sigma=0 calculates from kernel

            # Place the mask ROI in the full-sized mask
            mask[min_y:max_y, min_x:max_x] = mask_roi

            # Extract the masked area from the *original* frame
            mouth_cutout = frame[min_y:max_y, min_x:max_x].copy()

            lower_lip_polygon = expanded_landmarks # Return polygon in original frame coords
            mouth_box = (min_x, min_y, max_x, max_y) # Return the calculated box
        else:
            # print("Warning: Invalid mouth mask bounding box after padding/clamping.") # Optional debug
            pass

    except IndexError as idx_e:
        # print(f"Warning: Landmark index out of bounds during mouth mask creation: {idx_e}") # Optional debug
        pass
    except Exception as e:
        print(f"Error in create_lower_mouth_mask: {e}") # Print unexpected errors
        # import traceback
        # traceback.print_exc()
        pass

    # Return values, ensuring defaults if errors occurred
    return mask, mouth_cutout, mouth_box, lower_lip_polygon


def draw_mouth_mask_visualization(frame: np.ndarray, face, mouth_mask_data: tuple) -> np.ndarray:
    landmarks = face.landmark_2d_106
    if landmarks is not None and mouth_mask_data is not None:
        mask, mouth_cutout, (min_x, min_y, max_x, max_y), lower_lip_polygon = (
            mouth_mask_data
        )

    # Validate inputs
    if frame is None or face is None or mouth_mask_data is None or len(mouth_mask_data) != 4:
        return frame # Return original frame if inputs are invalid

    mask, mouth_cutout, box, lower_lip_polygon = mouth_mask_data
    (min_x, min_y, max_x, max_y) = box

    # Check if polygon is valid for drawing
    if lower_lip_polygon is None or not isinstance(lower_lip_polygon, np.ndarray) or len(lower_lip_polygon) < 3:
        return frame # Cannot draw without a valid polygon

    vis_frame = frame.copy()
    height, width = vis_frame.shape[:2]

    # Ensure box coordinates are valid integers within frame bounds
    try:
        min_x, min_y = max(0, int(min_x)), max(0, int(min_y))
        max_x, max_y = min(width, int(max_x)), min(height, int(max_y))
    except ValueError:
        # print("Warning: Invalid coordinates for mask visualization box.")
        return frame

    if max_x <= min_x or max_y <= min_y:
        return frame # Invalid box

    # Draw the lower lip polygon (green outline)
    try:
         # Ensure polygon points are within frame boundaries before drawing
         safe_polygon = lower_lip_polygon.copy()
         safe_polygon[:, 0] = np.clip(safe_polygon[:, 0], 0, width - 1)
         safe_polygon[:, 1] = np.clip(safe_polygon[:, 1], 0, height - 1)
         cv2.polylines(vis_frame, [safe_polygon.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    except Exception as e:
        print(f"Error drawing polygon for visualization: {e}") # Optional debug
        pass

    # Optional: Draw bounding box (red rectangle)
    # cv2.rectangle(vis_frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 1)

    # Optional: Add labels
    label_pos_y = min_y - 10 if min_y > 20 else max_y + 15 # Adjust position based on box location
    label_pos_x = min_x
    try:
        cv2.putText(vis_frame, "Mouth Mask", (label_pos_x, label_pos_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    except Exception as e:
        # print(f"Error drawing text for visualization: {e}") # Optional debug
        pass


    return vis_frame


def apply_mouth_area(
    frame: np.ndarray,
    mouth_cutout: np.ndarray,
    mouth_box: tuple,
    face_mask: np.ndarray, # Full face mask (for blending edges)
    mouth_polygon: np.ndarray, # Specific polygon for the mouth area itself
) -> np.ndarray:

    # Basic validation
    if (frame is None or mouth_cutout is None or mouth_box is None or
        face_mask is None or mouth_polygon is None):
        # print("Warning: Invalid input (None value) to apply_mouth_area") # Optional debug
        return frame
    if (mouth_cutout.size == 0 or face_mask.size == 0 or len(mouth_polygon) < 3):
        # print("Warning: Invalid input (empty array/polygon) to apply_mouth_area") # Optional debug
        return frame

    try: # Wrap main logic in try-except
        min_x, min_y, max_x, max_y = map(int, mouth_box) # Ensure integer coords
        box_width = max_x - min_x
        box_height = max_y - min_y

        # Check box validity
        if box_width <= 0 or box_height <= 0:
            # print("Warning: Invalid mouth box dimensions in apply_mouth_area.")
            return frame

        # Define the Region of Interest (ROI) on the target frame (swapped frame)
        frame_h, frame_w = frame.shape[:2]
        # Clamp coordinates strictly within frame boundaries
        min_y, max_y = max(0, min_y), min(frame_h, max_y)
        min_x, max_x = max(0, min_x), min(frame_w, max_x)

        # Recalculate box dimensions based on clamped coords
        box_width = max_x - min_x
        box_height = max_y - min_y
        if box_width <= 0 or box_height <= 0:
            # print("Warning: ROI became invalid after clamping in apply_mouth_area.")
            return frame # ROI is invalid

        color_corrected_mouth = color_transfer(resized_mouth_cutout, roi)

        # Ensure ROI extraction was successful
        if roi.size == 0:
            # print("Warning: Extracted ROI is empty in apply_mouth_area.")
            return frame

        # Resize mouth cutout from original frame to fit the ROI size
        resized_mouth_cutout = None
        if roi.shape[:2] != mouth_cutout.shape[:2]:
             # Check if mouth_cutout has valid dimensions before resizing
             if mouth_cutout.shape[0] > 0 and mouth_cutout.shape[1] > 0:
                 resized_mouth_cutout = cv2.resize(mouth_cutout, (box_width, box_height), interpolation=cv2.INTER_LINEAR)
             else:
                 # print("Warning: mouth_cutout has invalid dimensions, cannot resize.")
                 return frame # Cannot proceed without valid cutout
        else:
             resized_mouth_cutout = mouth_cutout

        # If resize failed or original was invalid
        if resized_mouth_cutout is None or resized_mouth_cutout.size == 0:
            # print("Warning: Mouth cutout is invalid after resize attempt.")
            return frame

        # --- Color Correction Step ---
        # Apply color transfer from ROI (swapped face region) to the original mouth cutout
        # This helps match lighting/color before blending
        color_corrected_mouth = resized_mouth_cutout # Default to resized if correction fails
        try:
           # Ensure both images are 3 channels for color transfer
           if len(resized_mouth_cutout.shape) == 3 and resized_mouth_cutout.shape[2] == 3 and \
              len(roi.shape) == 3 and roi.shape[2] == 3:
                 color_corrected_mouth = apply_color_transfer(resized_mouth_cutout, roi)
           else:
               # print("Warning: Cannot apply color transfer, images not BGR.")
               pass
        except cv2.error as ct_e: # Handle potential errors in color transfer
           # print(f"Warning: Color transfer failed: {ct_e}. Using uncorrected mouth cutout.") # Optional debug
           pass
        except Exception as ct_gen_e:
           # print(f"Warning: Unexpected error during color transfer: {ct_gen_e}")
           pass
        # --- End Color Correction ---


        # --- Mask Creation ---
        # Create a mask based *specifically* on the mouth_polygon, relative to the ROI
        polygon_mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
        # Adjust polygon coordinates relative to the ROI's top-left corner
        adjusted_polygon = mouth_polygon - [min_x, min_y]
        # Draw the filled polygon on the ROI mask
        cv2.fillPoly(polygon_mask_roi, [adjusted_polygon.astype(np.int32)], 255)

        # Feather the polygon mask (Gaussian blur)
        mask_feather_ratio = getattr(modules.globals, "mask_feather_ratio", 12) # Default 12
        # Calculate feather amount based on the smaller dimension of the box
        feather_base_dim = min(box_width, box_height)
        feather_amount = max(1, min(30, feather_base_dim // max(1, mask_feather_ratio))) # Avoid div by zero
        # Ensure kernel size is odd and positive
        kernel_size = 2 * feather_amount + 1
        feathered_polygon_mask = cv2.GaussianBlur(polygon_mask_roi.astype(float), (kernel_size, kernel_size), 0)

        # Normalize feathered mask to [0.0, 1.0] range
        max_val = feathered_polygon_mask.max()
        if max_val > 1e-6: # Avoid division by zero
           feathered_polygon_mask = feathered_polygon_mask / max_val
        else:
           feathered_polygon_mask.fill(0.0) # Mask is all black if max is near zero
        # --- End Mask Creation ---


        # --- Refined Blending ---
        # Get the corresponding ROI from the *full face mask* (already blurred)
        # Ensure face_mask is float and normalized [0.0, 1.0]
        if face_mask.dtype != np.float64 and face_mask.dtype != np.float32:
            face_mask_float = face_mask.astype(float) / 255.0
        else: # Assume already float [0,1] if type is float
            face_mask_float = face_mask
        face_mask_roi = face_mask_float[min_y:max_y, min_x:max_x]

        # Combine the feathered mouth polygon mask with the face mask ROI
        # Use minimum to ensure we only affect area inside both masks (mouth area within face)
        # This helps blend the edges smoothly with the surrounding swapped face region
        combined_mask = np.minimum(feathered_polygon_mask, face_mask_roi)

        # Expand mask to 3 channels for blending (ensure it matches image channels)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            combined_mask_3channel = combined_mask[:, :, np.newaxis]

            # Ensure data types are compatible for blending (float or double for mask, uint8 for images)
            color_corrected_mouth_uint8 = color_corrected_mouth.astype(np.uint8)
            roi_uint8 = roi.astype(np.uint8)
            combined_mask_float = combined_mask_3channel.astype(np.float64) # Use float64 for precision in mask

            # Blend: (original_mouth * combined_mask) + (swapped_face_roi * (1 - combined_mask))
            blended_roi = (color_corrected_mouth_uint8 * combined_mask_float +
                           roi_uint8 * (1.0 - combined_mask_float))

            # Place the blended ROI back into the frame
            frame[min_y:max_y, min_x:max_x] = blended_roi.astype(np.uint8)
        else:
            # print("Warning: Cannot apply mouth mask blending, frame is not 3-channel BGR.")
            pass # Don't modify frame if it's not BGR

    except Exception as e:
        print(f"Error applying mouth area: {e}") # Optional debug
        # import traceback
        # traceback.print_exc()
        pass # Don't crash, just return the frame as is

    return frame


def create_face_mask(face, frame: np.ndarray) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    landmarks = face.landmark_2d_106
    if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape[0] < 106:
        # print("Warning: Invalid or insufficient landmarks for face mask.")
        return mask # Return empty mask

    try: # Wrap main logic in try-except
        # Filter out non-finite landmark values
        if not np.all(np.isfinite(landmarks)):
            # print("Warning: Non-finite values detected in landmarks for face mask.")
            return mask

        landmarks_int = landmarks.astype(np.int32)

        # Use standard face outline landmarks (0-32)
        face_outline_points = landmarks_int[0:33] # Points 0 to 32 cover chin and sides


        # Calculate convex hull of these points
        # Use try-except as convexHull can fail on degenerate input
        try:
             hull = cv2.convexHull(full_face_poly.astype(np.float32)) # Use float for accuracy
             if hull is None or len(hull) < 3:
                 # print("Warning: Convex hull calculation failed or returned too few points.")
                 # Fallback: use bounding box of landmarks? Or just return empty mask?
                 return mask

    return mask
