import os # <-- Added for os.path.exists
from typing import Any, List
import cv2
import insightface
import threading

import modules.globals
import modules.processors.frame.core
# Ensure update_status is imported if not already globally accessible
# If it's part of modules.core, it might already be accessible via modules.core.update_status
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces, default_source_face
from modules.typing import Face, Frame
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video
from modules.cluster_analysis import find_closest_centroid

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'DLC.FACE-SWAPPER'


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    # Ensure both models are mentioned or downloaded if necessary
    # Conditional download might need adjustment if you want it to fetch FP32 too
    conditional_download(download_directory_path, ['https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx'])
    # Add a check or download for the FP32 model if you have a URL
    # conditional_download(download_directory_path, ['URL_TO_FP32_MODEL_HERE'])
    return True


def pre_start() -> bool:
    # --- No changes needed in pre_start ---
    if not modules.globals.map_faces and not is_image(modules.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not modules.globals.map_faces and not get_one_face(cv2.imread(modules.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            # --- MODIFICATION START ---
            # Define paths for both FP32 and FP16 models
            model_dir = resolve_relative_path('../models')
            model_path_fp32 = os.path.join(model_dir, 'inswapper_128.onnx')
            model_path_fp16 = os.path.join(model_dir, 'inswapper_128_fp16.onnx')
            chosen_model_path = None

            # Prioritize FP32 model
            if os.path.exists(model_path_fp32):
                chosen_model_path = model_path_fp32
                update_status(f"Loading FP32 model: {os.path.basename(chosen_model_path)}", NAME)
            # Fallback to FP16 model
            elif os.path.exists(model_path_fp16):
                chosen_model_path = model_path_fp16
                update_status(f"FP32 model not found. Loading FP16 model: {os.path.basename(chosen_model_path)}", NAME)
            # Error if neither model is found
            else:
                error_message = f"Face Swapper model not found. Please ensure 'inswapper_128.onnx' (recommended) or 'inswapper_128_fp16.onnx' exists in the '{model_dir}' directory."
                update_status(error_message, NAME)
                raise FileNotFoundError(error_message)

            # Load the chosen model
            try:
                FACE_SWAPPER = insightface.model_zoo.get_model(chosen_model_path, providers=modules.globals.execution_providers)
            except Exception as e:
                update_status(f"Error loading Face Swapper model {os.path.basename(chosen_model_path)}: {e}", NAME)
                # Optionally, re-raise the exception or handle it more gracefully
                raise e
            # --- MODIFICATION END ---
    return FACE_SWAPPER


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    # --- No changes needed in swap_face ---
    swapper = get_face_swapper()
    if swapper is None:
         # Handle case where model failed to load
         update_status("Face swapper model not loaded, skipping swap.", NAME)
         return temp_frame
    return swapper.get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    # --- No changes needed in process_frame ---
    # Ensure the frame is in RGB format if color correction is enabled
    # Note: InsightFace swapper often expects BGR by default. Double-check if color issues appear.
    # If color correction is needed *before* swapping and insightface needs BGR:
    # original_was_bgr = True # Assume input is BGR
    # if modules.globals.color_correction:
    #     temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
    #     original_was_bgr = False # Now it's RGB

    if modules.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)

    # Convert back if necessary (example, might not be needed depending on workflow)
    # if modules.globals.color_correction and not original_was_bgr:
    #      temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_RGB2BGR)

    return temp_frame


def process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame:
    # --- No changes needed in process_frame_v2 ---
    # (Assuming swap_face handles the potential None return from get_face_swapper)
    if is_image(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map_entry in modules.globals.souce_target_map: # Renamed 'map' to 'map_entry'
                target_face = map_entry['target']['face']
                temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            for map_entry in modules.globals.souce_target_map: # Renamed 'map' to 'map_entry'
                if "source" in map_entry:
                    source_face = map_entry['source']['face']
                    target_face = map_entry['target']['face']
                    temp_frame = swap_face(source_face, target_face, temp_frame)

    elif is_video(modules.globals.target_path):
        if modules.globals.many_faces:
            source_face = default_source_face()
            for map_entry in modules.globals.souce_target_map: # Renamed 'map' to 'map_entry'
                target_frame = [f for f in map_entry['target_faces_in_frame'] if f['location'] == temp_frame_path]

                for frame in target_frame:
                    for target_face in frame['faces']:
                        temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            for map_entry in modules.globals.souce_target_map: # Renamed 'map' to 'map_entry'
                if "source" in map_entry:
                    target_frame = [f for f in map_entry['target_faces_in_frame'] if f['location'] == temp_frame_path]
                    source_face = map_entry['source']['face']

                    for frame in target_frame:
                        for target_face in frame['faces']:
                            temp_frame = swap_face(source_face, target_face, temp_frame)
    else: # Fallback for neither image nor video (e.g., live feed?)
        detected_faces = get_many_faces(temp_frame)
        if modules.globals.many_faces:
            if detected_faces:
                source_face = default_source_face()
                for target_face in detected_faces:
                    temp_frame = swap_face(source_face, target_face, temp_frame)

        elif not modules.globals.many_faces:
            if detected_faces and hasattr(modules.globals, 'simple_map') and modules.globals.simple_map: # Check simple_map exists
                if len(detected_faces) <= len(modules.globals.simple_map['target_embeddings']):
                    for detected_face in detected_faces:
                        closest_centroid_index, _ = find_closest_centroid(modules.globals.simple_map['target_embeddings'], detected_face.normed_embedding)
                        temp_frame = swap_face(modules.globals.simple_map['source_faces'][closest_centroid_index], detected_face, temp_frame)
                else:
                    detected_faces_centroids = [face.normed_embedding for face in detected_faces]
                    i = 0
                    for target_embedding in modules.globals.simple_map['target_embeddings']:
                        closest_centroid_index, _ = find_closest_centroid(detected_faces_centroids, target_embedding)
                        # Ensure index is valid before accessing detected_faces
                        if closest_centroid_index < len(detected_faces):
                            temp_frame = swap_face(modules.globals.simple_map['source_faces'][i], detected_faces[closest_centroid_index], temp_frame)
                        i += 1
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    # --- No changes needed in process_frames ---
    # Note: Ensure get_one_face is called only once if possible for efficiency if !map_faces
    source_face = None
    if not modules.globals.map_faces:
        source_img = cv2.imread(source_path)
        if source_img is not None:
            source_face = get_one_face(source_img)
        if source_face is None:
             update_status(f"Could not find face in source image: {source_path}, skipping swap.", NAME)
             # If no source face, maybe skip processing? Or handle differently.
             # For now, it will proceed but swap_face might fail later.

    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            update_status(f"Warning: Could not read frame {temp_frame_path}", NAME)
            if progress: progress.update(1) # Still update progress even if frame fails
            continue # Skip to next frame

        try:
            if not modules.globals.map_faces:
                if source_face: # Only process if source face was found
                    result = process_frame(source_face, temp_frame)
                else:
                    result = temp_frame # No source face, return original frame
            else:
                 result = process_frame_v2(temp_frame, temp_frame_path)

            cv2.imwrite(temp_frame_path, result)
        except Exception as exception:
            update_status(f"Error processing frame {os.path.basename(temp_frame_path)}: {exception}", NAME)
            # Decide whether to 'pass' (continue processing other frames) or raise
            pass # Continue processing other frames
        finally:
            if progress:
                progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    # --- No changes needed in process_image ---
    # Note: Added checks for successful image reads and face detection
    target_frame = cv2.imread(target_path) # Read original target for processing
    if target_frame is None:
        update_status(f"Error: Could not read target image: {target_path}", NAME)
        return

    if not modules.globals.map_faces:
        source_img = cv2.imread(source_path)
        if source_img is None:
             update_status(f"Error: Could not read source image: {source_path}", NAME)
             return
        source_face = get_one_face(source_img)
        if source_face is None:
            update_status(f"Error: No face found in source image: {source_path}", NAME)
            return

        result = process_frame(source_face, target_frame)
    else:
        if modules.globals.many_faces:
            update_status('Many faces enabled. Using first source image (if applicable in v2). Processing...', NAME)
        # For process_frame_v2 on single image, it reads the 'output_path' which should be a copy
        # Let's process the 'target_frame' we read instead.
        result = process_frame_v2(target_frame) # Process the frame directly

    # Write the final result to the output path
    success = cv2.imwrite(output_path, result)
    if not success:
        update_status(f"Error: Failed to write output image to: {output_path}", NAME)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    # --- No changes needed in process_video ---
    if modules.globals.map_faces and modules.globals.many_faces:
        update_status('Many faces enabled. Using first source image (if applicable in v2). Processing...', NAME)
    # The core processing logic is delegated, which is good.
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)