from typing import Any, List, Optional
import cv2
import insightface
import threading
import numpy as np
import platform
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
import os
from collections import deque
import time

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-SWAPPER"

# --- START: Added for Interpolation ---
PREVIOUS_FRAME_RESULT = None # Stores the final processed frame from the previous step
# --- END: Added for Interpolation ---

# --- START: Mac M1-M5 Optimizations ---
IS_APPLE_SILICON = platform.system() == 'Darwin' and platform.machine() == 'arm64'
FRAME_CACHE = deque(maxlen=3)  # Cache for frame reuse
FACE_DETECTION_CACHE = {}  # Cache face detections
LAST_DETECTION_TIME = 0
DETECTION_INTERVAL = 0.033  # ~30 FPS detection rate for live mode
FRAME_SKIP_COUNTER = 0
ADAPTIVE_QUALITY = True
# --- END: Mac M1-M5 Optimizations ---

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
    # Simplified pre_start, assuming checks happen before calling process functions
    model_path = os.path.join(models_dir, "inswapper_128_fp16.onnx")
    if not os.path.exists(model_path):
        update_status(f"Model not found: {model_path}. Please download it.", NAME)
        return False

    # Try to get the face swapper to ensure it loads correctly
    if get_face_swapper() is None:
        # Error message already printed within get_face_swapper
        return False

    # Add other essential checks if needed, e.g., target/source path validity
    return True


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_name = "inswapper_128.onnx"
            if "CUDAExecutionProvider" in modules.globals.execution_providers:
                model_name = "inswapper_128_fp16.onnx"
            model_path = os.path.join(models_dir, model_name)
            update_status(f"Loading face swapper model from: {model_path}", NAME)
            try:
                # Optimized provider configuration for Apple Silicon
                providers_config = []
                for p in modules.globals.execution_providers:
                    if p == "CoreMLExecutionProvider" and IS_APPLE_SILICON:
                        # Enhanced CoreML configuration for M1-M5
                        providers_config.append((
                            "CoreMLExecutionProvider",
                            {
                                "ModelFormat": "MLProgram",
                                "MLComputeUnits": "ALL",  # Use Neural Engine + GPU + CPU
                                "SpecializationStrategy": "FastPrediction",
                                "AllowLowPrecisionAccumulationOnGPU": 1,
                                "EnableOnSubgraphs": 1,
                                "RequireStaticShapes": 0,
                                "MaximumCacheSize": 1024 * 1024 * 512,  # 512MB cache
                            }
                        ))
                    else:
                        providers_config.append(p)
                
                FACE_SWAPPER = insightface.model_zoo.get_model(
                    model_path,
                    providers=providers_config,
                )
                update_status("Face swapper model loaded successfully.", NAME)
            except Exception as e:
                update_status(f"Error loading face swapper model: {e}", NAME)
                FACE_SWAPPER = None
                return None
    return FACE_SWAPPER


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    face_swapper = get_face_swapper()
    if face_swapper is None:
        update_status("Face swapper model not loaded or failed to load. Skipping swap.", NAME)
        return temp_frame

    # Store a copy of the original frame before swapping for opacity blending
    original_frame = temp_frame.copy()

    # Pre-swap Input Check with optimization
    if temp_frame.dtype != np.uint8:
        temp_frame = np.clip(temp_frame, 0, 255).astype(np.uint8)

    # Apply the face swap with optimized memory handling
    try:
        # For Apple Silicon, use optimized inference
        if IS_APPLE_SILICON:
            # Ensure contiguous memory layout for better performance
            temp_frame = np.ascontiguousarray(temp_frame)
        
        swapped_frame_raw = face_swapper.get(
            temp_frame, target_face, source_face, paste_back=True
        )

        # --- START: CRITICAL FIX FOR ORT 1.17 ---
        # Check the output type and range from the model
        if swapped_frame_raw is None:
             # print("Warning: face_swapper.get returned None.") # Debug
             return original_frame # Return original if swap somehow failed internally

        # Ensure the output is a numpy array
        if not isinstance(swapped_frame_raw, np.ndarray):
            # print(f"Warning: face_swapper.get returned type {type(swapped_frame_raw)}, expected numpy array.") # Debug
            return original_frame

        # Ensure the output has the correct shape (like the input frame)
        if swapped_frame_raw.shape != temp_frame.shape:
             # print(f"Warning: Swapped frame shape {swapped_frame_raw.shape} differs from input {temp_frame.shape}.") # Debug
             # Attempt resize (might distort if aspect ratio changed, but better than crashing)
             try:
                 swapped_frame_raw = cv2.resize(swapped_frame_raw, (temp_frame.shape[1], temp_frame.shape[0]))
             except Exception as resize_e:
                 # print(f"Error resizing swapped frame: {resize_e}") # Debug
                 return original_frame

        # Explicitly clip values to 0-255 and convert to uint8
        # This handles cases where the model might output floats or values outside the valid range
        swapped_frame = np.clip(swapped_frame_raw, 0, 255).astype(np.uint8)
        # --- END: CRITICAL FIX FOR ORT 1.17 ---

    except Exception as e:
        print(f"Error during face swap using face_swapper.get: {e}") # More specific error
        # import traceback
        # traceback.print_exc() # Print full traceback for debugging
        return original_frame # Return original if swap fails

    # --- Post-swap Processing (Masking, Opacity, etc.) ---
    # Now, work with the guaranteed uint8 'swapped_frame'

    if getattr(modules.globals, "mouth_mask", False): # Check if mouth_mask is enabled
        # Create a mask for the target face
        face_mask = create_face_mask(target_face, temp_frame) # Use temp_frame (original shape) for mask creation geometry

        # Create the mouth mask using original geometry
        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
            create_lower_mouth_mask(target_face, temp_frame) # Use temp_frame (original) for cutout
        )

        # Apply the mouth area only if mouth_cutout exists
        if mouth_cutout is not None and mouth_box != (0,0,0,0): # Add check for valid box
             # Apply mouth area (from original) onto the 'swapped_frame'
            swapped_frame = apply_mouth_area(
                swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
            )

            if getattr(modules.globals, "show_mouth_mask_box", False):
                        mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
                        # Draw visualization on the swapped_frame *before* opacity blending
                        swapped_frame = draw_mouth_mask_visualization(
                            swapped_frame, target_face, mouth_mask_data
                        )
        
            # --- Poisson Blending ---
            if getattr(modules.globals, "poisson_blend", False):
                face_mask = create_face_mask(target_face, temp_frame)
                if face_mask is not None:
                    # Find bounding box of the mask
                    y_indices, x_indices = np.where(face_mask > 0)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
                        # Calculate center
                        center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
        
                        # Crop src and mask
                        src_crop = swapped_frame[y_min : y_max + 1, x_min : x_max + 1]
                        mask_crop = face_mask[y_min : y_max + 1, x_min : x_max + 1]
        
                        try:
                            # Use original_frame as destination to blend the swapped face onto it
                            swapped_frame = cv2.seamlessClone(
                                src_crop,
                                original_frame,
                                mask_crop,
                                center,
                                cv2.NORMAL_CLONE,
                            )
                        except Exception as e:
                            print(f"Poisson blending failed: {e}")
        
            # Apply opacity blend between the original frame and the swapped frame
    opacity = getattr(modules.globals, "opacity", 1.0)
    # Ensure opacity is within valid range [0.0, 1.0]
    opacity = max(0.0, min(1.0, opacity))

    # Blend the original_frame with the (potentially mouth-masked) swapped_frame
    # Ensure both frames are uint8 before blending
    final_swapped_frame = cv2.addWeighted(original_frame.astype(np.uint8), 1 - opacity, swapped_frame.astype(np.uint8), opacity, 0)

    # Ensure final frame is uint8 after blending (addWeighted should preserve it, but belt-and-suspenders)
    final_swapped_frame = final_swapped_frame.astype(np.uint8)

    return final_swapped_frame


# --- START: Mac M1-M5 Optimized Face Detection ---
def get_faces_optimized(frame: Frame, use_cache: bool = True) -> Optional[List[Face]]:
    """Optimized face detection for live mode on Apple Silicon"""
    global LAST_DETECTION_TIME, FACE_DETECTION_CACHE
    
    if not use_cache or not IS_APPLE_SILICON:
        # Standard detection
        if modules.globals.many_faces:
            return get_many_faces(frame)
        else:
            face = get_one_face(frame)
            return [face] if face else None
    
    # Adaptive detection rate for live mode
    current_time = time.time()
    time_since_last = current_time - LAST_DETECTION_TIME
    
    # Skip detection if too soon (adaptive frame skipping)
    if time_since_last < DETECTION_INTERVAL and FACE_DETECTION_CACHE:
        return FACE_DETECTION_CACHE.get('faces')
    
    # Perform detection
    LAST_DETECTION_TIME = current_time
    if modules.globals.many_faces:
        faces = get_many_faces(frame)
    else:
        face = get_one_face(frame)
        faces = [face] if face else None
    
    # Cache results
    FACE_DETECTION_CACHE['faces'] = faces
    FACE_DETECTION_CACHE['timestamp'] = current_time
    
    return faces
# --- END: Mac M1-M5 Optimized Face Detection ---

# --- START: Helper function for interpolation and sharpening ---
def apply_post_processing(current_frame: Frame, swapped_face_bboxes: List[np.ndarray]) -> Frame:
    """Applies sharpening and interpolation with Apple Silicon optimizations."""
    global PREVIOUS_FRAME_RESULT

    processed_frame = current_frame.copy()

    # 1. Apply Sharpening (if enabled) with optimized kernel for Apple Silicon
    sharpness_value = getattr(modules.globals, "sharpness", 0.0)
    if sharpness_value > 0.0 and swapped_face_bboxes:
        height, width = processed_frame.shape[:2]
        for bbox in swapped_face_bboxes:
            # Ensure bbox is iterable and has 4 elements
            if not hasattr(bbox, '__iter__') or len(bbox) != 4:
                # print(f"Warning: Invalid bbox format for sharpening: {bbox}") # Debug
                continue
            x1, y1, x2, y2 = bbox
            # Ensure coordinates are integers and within bounds
            try:
                 x1, y1 = max(0, int(x1)), max(0, int(y1))
                 x2, y2 = min(width, int(x2)), min(height, int(y2))
            except ValueError:
                # print(f"Warning: Could not convert bbox coordinates to int: {bbox}") # Debug
                continue


            if x2 <= x1 or y2 <= y1:
                continue

            face_region = processed_frame[y1:y2, x1:x2]
            if face_region.size == 0: continue

            # Apply sharpening with optimized parameters for Apple Silicon
            try:
                # Use smaller sigma for faster processing on Apple Silicon
                sigma = 2 if IS_APPLE_SILICON else 3
                blurred = cv2.GaussianBlur(face_region, (0, 0), sigma)
                sharpened_region = cv2.addWeighted(
                    face_region, 1.0 + sharpness_value,
                    blurred, -sharpness_value,
                    0
                )
                sharpened_region = np.clip(sharpened_region, 0, 255).astype(np.uint8)
                processed_frame[y1:y2, x1:x2] = sharpened_region
            except cv2.error:
                pass


    # 2. Apply Interpolation (if enabled)
    enable_interpolation = getattr(modules.globals, "enable_interpolation", False)
    interpolation_weight = getattr(modules.globals, "interpolation_weight", 0.2)

    final_frame = processed_frame # Start with the current (potentially sharpened) frame

    if enable_interpolation and 0 < interpolation_weight < 1:
        if PREVIOUS_FRAME_RESULT is not None and PREVIOUS_FRAME_RESULT.shape == processed_frame.shape and PREVIOUS_FRAME_RESULT.dtype == processed_frame.dtype:
            # Perform interpolation
            try:
                 final_frame = cv2.addWeighted(
                    PREVIOUS_FRAME_RESULT, 1.0 - interpolation_weight,
                    processed_frame, interpolation_weight,
                    0
                 )
                 # Ensure final frame is uint8
                 final_frame = np.clip(final_frame, 0, 255).astype(np.uint8)
            except cv2.error as interp_e:
                 # print(f"Warning: OpenCV error during interpolation: {interp_e}") # Debug
                 final_frame = processed_frame # Use current frame if interpolation fails
                 PREVIOUS_FRAME_RESULT = None # Reset state if error occurs

            # Update the state for the next frame *with the interpolated result*
            PREVIOUS_FRAME_RESULT = final_frame.copy()
        else:
            # If previous frame invalid or doesn't match, use current frame and update state
            if PREVIOUS_FRAME_RESULT is not None and PREVIOUS_FRAME_RESULT.shape != processed_frame.shape:
                # print("Info: Frame shape changed, resetting interpolation state.") # Debug
                pass
            PREVIOUS_FRAME_RESULT = processed_frame.copy()
    else:
         # If interpolation is off or weight is invalid, just use the current frame
         # Update state with the current (potentially sharpened) frame
         # Reset previous frame state if interpolation was just turned off or weight is invalid
         PREVIOUS_FRAME_RESULT = processed_frame.copy()


    return final_frame
# --- END: Helper function for interpolation and sharpening ---


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    """
    DEPRECATED / SIMPLER VERSION - Processes a single frame using one source face.
    Consider using process_frame_v2 for more complex scenarios.
    """
    if getattr(modules.globals, "opacity", 1.0) == 0:
        # If opacity is 0, no swap happens, so no post-processing needed.
        # Also reset interpolation state if it was active.
        global PREVIOUS_FRAME_RESULT
        PREVIOUS_FRAME_RESULT = None
        return temp_frame

    # Color correction removed from here (better applied before swap if needed)

    processed_frame = temp_frame # Start with the input frame
    swapped_face_bboxes = [] # Keep track of where swaps happened

    if modules.globals.many_faces:
        many_faces = get_many_faces(processed_frame)
        if many_faces:
            current_swap_target = processed_frame.copy() # Apply swaps sequentially on a copy
            for target_face in many_faces:
                current_swap_target = swap_face(source_face, target_face, current_swap_target)
                if target_face is not None and hasattr(target_face, "bbox") and target_face.bbox is not None:
                    swapped_face_bboxes.append(target_face.bbox.astype(int))
            processed_frame = current_swap_target # Assign the final result after all swaps
    else:
        target_face = get_one_face(processed_frame)
        if target_face:
            processed_frame = swap_face(source_face, target_face, processed_frame)
            if target_face is not None and hasattr(target_face, "bbox") and target_face.bbox is not None:
                    swapped_face_bboxes.append(target_face.bbox.astype(int))

    # Apply sharpening and interpolation
    final_frame = apply_post_processing(processed_frame, swapped_face_bboxes)

    return final_frame


def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    """Handles complex mapping scenarios (map_faces=True) and live streams."""
    if getattr(modules.globals, "opacity", 1.0) == 0:
        # If opacity is 0, no swap happens, so no post-processing needed.
        # Also reset interpolation state if it was active.
        global PREVIOUS_FRAME_RESULT
        PREVIOUS_FRAME_RESULT = None
        return temp_frame

    processed_frame = temp_frame # Start with the input frame
    swapped_face_bboxes = [] # Keep track of where swaps happened

    # Determine source/target pairs based on mode
    source_target_pairs = []

    # Ensure maps exist before accessing them
    source_target_map = getattr(modules.globals, "source_target_map", None)
    simple_map = getattr(modules.globals, "simple_map", None)

    # Check if target is a file path (image or video) or live stream
    is_file_target = modules.globals.target_path and (is_image(modules.globals.target_path) or is_video(modules.globals.target_path))

    if is_file_target:
        # Processing specific image or video file with pre-analyzed maps
        if source_target_map:
            if modules.globals.many_faces:
                source_face = default_source_face() # Use default source for all targets
                if source_face:
                    for map_data in source_target_map:
                        if is_image(modules.globals.target_path):
                            target_info = map_data.get("target", {})
                            if target_info: # Check if target info exists
                                target_face = target_info.get("face")
                                if target_face:
                                    source_target_pairs.append((source_face, target_face))
                        elif is_video(modules.globals.target_path):
                             # Find faces for the current frame_path in video map
                             target_frames_data = map_data.get("target_faces_in_frame", [])
                             if target_frames_data: # Check if frame data exists
                                 target_frames = [f for f in target_frames_data if f and f.get("location") == temp_frame_path]
                                 for frame_data in target_frames:
                                     faces_in_frame = frame_data.get("faces", [])
                                     if faces_in_frame: # Check if faces exist
                                         for target_face in faces_in_frame:
                                             source_target_pairs.append((source_face, target_face))
            else: # Single face or specific mapping
                 for map_data in source_target_map:
                    source_info = map_data.get("source", {})
                    if not source_info: continue # Skip if no source info
                    source_face = source_info.get("face")
                    if not source_face: continue # Skip if no source defined for this map entry

                    if is_image(modules.globals.target_path):
                        target_info = map_data.get("target", {})
                        if target_info:
                           target_face = target_info.get("face")
                           if target_face:
                              source_target_pairs.append((source_face, target_face))
                    elif is_video(modules.globals.target_path):
                        target_frames_data = map_data.get("target_faces_in_frame", [])
                        if target_frames_data:
                           target_frames = [f for f in target_frames_data if f and f.get("location") == temp_frame_path]
                           for frame_data in target_frames:
                               faces_in_frame = frame_data.get("faces", [])
                               if faces_in_frame:
                                  for target_face in faces_in_frame:
                                      source_target_pairs.append((source_face, target_face))

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
    """
    Processes a list of frame paths (typically for video).
    Iterates through frames, applies the appropriate swapping logic based on globals,
    and saves the result back to the frame path. Handles multi-threading via caller.
    """
    # Determine which processing function to use based on map_faces global setting
    use_v2 = getattr(modules.globals, "map_faces", False)
    source_face = None # Initialize source_face

    # --- Pre-load source face only if needed (Simple Mode: map_faces=False) ---
    if not use_v2:
        if not source_path or not os.path.exists(source_path):
            update_status(f"Error: Source path invalid or not provided for simple mode: {source_path}", NAME)
            # Log the error but allow proceeding; subsequent check will stop processing.
        else:
            try:
                source_img = cv2.imread(source_path)
                if source_img is None:
                    # Specific error for file reading failure
                    update_status(f"Error reading source image file {source_path}. Please check the path and file integrity.", NAME)
                else:
                    source_face = get_one_face(source_img)
                    if source_face is None:
                        # Specific message for no face detected after successful read
                        update_status(f"Warning: Successfully read source image {source_path}, but no face was detected. Swaps will be skipped.", NAME)
            except Exception as e:
                # Print the specific exception caught
                import traceback
                print(f"{NAME}: Caught exception during source image processing for {source_path}:")
                traceback.print_exc() # Print the full traceback
                update_status(f"Error during source image reading or analysis {source_path}: {e}", NAME)
                # Log general exception during the process

    total_frames = len(temp_frame_paths)
    # update_status(f"Processing {total_frames} frames. Use V2 (map_faces): {use_v2}", NAME) # Optional Debug

    # --- Stop processing entirely if in Simple Mode and source face is invalid ---
    if not use_v2 and source_face is None:
        update_status(f"Halting video processing: Invalid or no face detected in source image for simple mode.", NAME)
        if progress:
            # Ensure the progress bar completes if it was started
            remaining_updates = total_frames - progress.n if hasattr(progress, 'n') else total_frames
            if remaining_updates > 0:
                progress.update(remaining_updates)
        return # Exit the function entirely

    # --- Process each frame path provided in the list ---
    # Note: In the current core.py multi_process_frame, temp_frame_paths will usually contain only ONE path per call.
    for i, temp_frame_path in enumerate(temp_frame_paths):
        # update_status(f"Processing frame {i+1}/{total_frames}: {os.path.basename(temp_frame_path)}", NAME) # Optional Debug

        # Read the target frame
        try:
            temp_frame = cv2.imread(temp_frame_path)
            if temp_frame is None:
                print(f"{NAME}: Error: Could not read frame: {temp_frame_path}, skipping.")
                if progress: progress.update(1)
                continue # Skip this frame if read fails
        except Exception as read_e:
            print(f"{NAME}: Error reading frame {temp_frame_path}: {read_e}, skipping.")
            if progress: progress.update(1)
            continue

        # Select processing function and execute
        result_frame = None
        try:
            if use_v2:
                # V2 uses global maps and needs the frame path for lookup in video mode
                # update_status(f"Using process_frame_v2 for: {os.path.basename(temp_frame_path)}", NAME) # Optional Debug
                result_frame = process_frame_v2(temp_frame, temp_frame_path)
            else:
                # Simple mode uses the pre-loaded source_face (already checked for validity above)
                # update_status(f"Using process_frame (simple) for: {os.path.basename(temp_frame_path)}", NAME) # Optional Debug
                result_frame = process_frame(source_face, temp_frame) # source_face is guaranteed to be valid here

            # Check if processing actually returned a frame
            if result_frame is None:
                 print(f"{NAME}: Warning: Processing returned None for frame {temp_frame_path}. Using original.")
                 result_frame = temp_frame

        except Exception as proc_e:
            print(f"{NAME}: Error processing frame {temp_frame_path}: {proc_e}")
            # import traceback # Optional for detailed debugging
            # traceback.print_exc()
            result_frame = temp_frame # Use original frame on processing error

        # Write the result back to the same frame path
        try:
            write_success = cv2.imwrite(temp_frame_path, result_frame)
            if not write_success:
                print(f"{NAME}: Error: Failed to write processed frame to {temp_frame_path}")
        except Exception as write_e:
            print(f"{NAME}: Error writing frame {temp_frame_path}: {write_e}")

        # Update progress bar
        if progress:
            progress.update(1)
        # else: # Basic console progress (optional)
        #     if (i + 1) % 10 == 0 or (i + 1) == total_frames: # Update every 10 frames or on last frame
        #        update_status(f"Processed frame {i+1}/{total_frames}", NAME)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Processes a single target image."""
    # --- Reset interpolation state for single image processing ---
    global PREVIOUS_FRAME_RESULT
    PREVIOUS_FRAME_RESULT = None
    # ---

    use_v2 = getattr(modules.globals, "map_faces", False)

    # Read target first
    try:
        target_frame = cv2.imread(target_path)
        if target_frame is None:
            update_status(f"Error: Could not read target image: {target_path}", NAME)
            return
    except Exception as read_e:
        update_status(f"Error reading target image {target_path}: {read_e}", NAME)
        return

    result = None
    try:
        if use_v2:
            if getattr(modules.globals, "many_faces", False):
                 update_status("Processing image with 'map_faces' and 'many_faces'. Using pre-analysis map.", NAME)
            # V2 processes based on global maps, doesn't need source_path here directly
            # Assumes maps are pre-populated. Pass target_path for map lookup.
            result = process_frame_v2(target_frame, target_path)

        else: # Simple mode
            try:
                source_img = cv2.imread(source_path)
                if source_img is None:
                    update_status(f"Error: Could not read source image: {source_path}", NAME)
                    return
                source_face = get_one_face(source_img)
                if not source_face:
                    update_status(f"Error: No face found in source image: {source_path}", NAME)
                    return
            except Exception as src_e:
                 update_status(f"Error reading or analyzing source image {source_path}: {src_e}", NAME)
                 return

            result = process_frame(source_face, target_frame)

        # Write the result if processing was successful
        if result is not None:
            write_success = cv2.imwrite(output_path, result)
            if write_success:
                update_status(f"Output image saved to: {output_path}", NAME)
            else:
                update_status(f"Error: Failed to write output image to {output_path}", NAME)
        else:
            # This case might occur if process_frame/v2 returns None unexpectedly
            update_status("Image processing failed (result was None).", NAME)

    except Exception as proc_e:
         update_status(f"Error during image processing: {proc_e}", NAME)
         # import traceback
         # traceback.print_exc()


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

def create_lower_mouth_mask(
    face: Face, frame: Frame
) -> (np.ndarray, np.ndarray, tuple, np.ndarray):
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


def draw_mouth_mask_visualization(
    frame: Frame, face: Face, mouth_mask_data: tuple
) -> Frame:

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

        roi = frame[min_y:max_y, min_x:max_x]

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


def create_face_mask(face: Face, frame: Frame) -> np.ndarray:
    """Creates a feathered mask covering the whole face area based on landmarks."""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8) # Start with uint8

    # Validate inputs
    if face is None or not hasattr(face, 'landmark_2d_106') or frame is None:
        # print("Warning: Invalid face or frame for create_face_mask.")
        return mask # Return empty mask

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

             # Draw the filled convex hull on the mask
             cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        except Exception as hull_e:
             print(f"Error creating convex hull for face mask: {hull_e}")
             return mask # Return empty mask on error


        # Apply Gaussian blur to feather the mask edges
        # Kernel size should be reasonably large, odd, and positive
        blur_k_size = getattr(modules.globals, "face_mask_blur", 31) # Default 31
        blur_k_size = max(1, blur_k_size // 2 * 2 + 1) # Ensure odd and positive

        # Use sigma=0 to let OpenCV calculate from kernel size
        # Apply blur to the uint8 mask directly
        mask = cv2.GaussianBlur(mask, (blur_k_size, blur_k_size), 0)

        # --- Optional: Return float mask for apply_mouth_area ---
        # mask = mask.astype(float) / 255.0
        # ---

    except IndexError:
        # print("Warning: Landmark index out of bounds for face mask.") # Optional debug
        pass
    except Exception as e:
        print(f"Error creating face mask: {e}") # Print unexpected errors
        # import traceback
        # traceback.print_exc()
        pass

    return mask # Return uint8 mask


def apply_color_transfer(source, target):
    """
    Apply color transfer using LAB color space. Handles potential division by zero and ensures output is uint8.
    """
    # Input validation
    if source is None or target is None or source.size == 0 or target.size == 0:
        # print("Warning: Invalid input to apply_color_transfer.")
        return source # Return original source if invalid input

    # Ensure images are 3-channel BGR uint8
    if len(source.shape) != 3 or source.shape[2] != 3 or source.dtype != np.uint8:
        # print("Warning: Source image for color transfer is not uint8 BGR.")
        # Attempt conversion if possible, otherwise return original
        try:
            if len(source.shape) == 2: # Grayscale
                source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            source = np.clip(source, 0, 255).astype(np.uint8)
            if len(source.shape)!= 3 or source.shape[2]!= 3: raise ValueError("Conversion failed")
        except Exception:
            return source
    if len(target.shape) != 3 or target.shape[2] != 3 or target.dtype != np.uint8:
        # print("Warning: Target image for color transfer is not uint8 BGR.")
        try:
            if len(target.shape) == 2: # Grayscale
                target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
            target = np.clip(target, 0, 255).astype(np.uint8)
            if len(target.shape)!= 3 or target.shape[2]!= 3: raise ValueError("Conversion failed")
        except Exception:
             return source # Return original source if target invalid

    result_bgr = source # Default to original source in case of errors

    try:
        # Convert to float32 [0, 1] range for LAB conversion
        source_float = source.astype(np.float32) / 255.0
        target_float = target.astype(np.float32) / 255.0

        source_lab = cv2.cvtColor(source_float, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target_float, cv2.COLOR_BGR2LAB)

        # Compute statistics
        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)

        # Reshape for broadcasting
        source_mean = source_mean.reshape((1, 1, 3))
        source_std = source_std.reshape((1, 1, 3))
        target_mean = target_mean.reshape((1, 1, 3))
        target_std = target_std.reshape((1, 1, 3))

        # Avoid division by zero or very small std deviations (add epsilon)
        epsilon = 1e-6
        source_std = np.maximum(source_std, epsilon)
        # target_std = np.maximum(target_std, epsilon) # Target std can be small

        # Perform color transfer in LAB space
        result_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean

        # --- No explicit clipping needed in LAB space typically ---
        # Clipping is handled implicitly by the conversion back to BGR and then to uint8

        # Convert back to BGR float [0, 1]
        result_bgr_float = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

        # Clip final BGR values to [0, 1] range before scaling to [0, 255]
        result_bgr_float = np.clip(result_bgr_float, 0.0, 1.0)

        # Convert back to uint8 [0, 255]
        result_bgr = (result_bgr_float * 255.0).astype("uint8")

    except cv2.error as e:
         # print(f"OpenCV error during color transfer: {e}. Returning original source.") # Optional debug
         return source # Return original source if conversion fails
    except Exception as e:
         # print(f"Unexpected color transfer error: {e}. Returning original source.") # Optional debug
         # import traceback
         # traceback.print_exc()
         return source

    return result_bgr