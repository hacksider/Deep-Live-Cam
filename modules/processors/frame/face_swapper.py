from typing import Any, List
import cv2
import insightface
import threading
import numpy as np
import modules.globals
import logging
import time
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

    # Apply the face swap with optimized settings for better performance
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


def swap_face_enhanced(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Enhanced face swapping with better quality and performance optimizations"""
    face_swapper = get_face_swapper()
    
    # Apply the face swap
    swapped_frame = face_swapper.get(
        temp_frame, target_face, source_face, paste_back=True
    )
    
    # Enhanced post-processing for better quality
    swapped_frame = enhance_face_swap_quality(swapped_frame, source_face, target_face, temp_frame)
    
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


def enhance_face_swap_quality(swapped_frame: Frame, source_face: Face, target_face: Face, original_frame: Frame) -> Frame:
    """Apply quality enhancements to the swapped face"""
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
            
        # Extract face regions
        swapped_face = swapped_frame[y1:y2, x1:x2]
        original_face = original_frame[y1:y2, x1:x2]
        
        # Apply color matching
        color_matched = apply_advanced_color_matching(swapped_face, original_face)
        
        # Apply edge smoothing
        smoothed = apply_edge_smoothing(color_matched, original_face)
        
        # Blend back into frame
        swapped_frame[y1:y2, x1:x2] = smoothed
        
        return swapped_frame
        
    except Exception as e:
        # Return original swapped frame if enhancement fails
        return swapped_frame


def apply_advanced_color_matching(swapped_face: np.ndarray, target_face: np.ndarray) -> np.ndarray:
    """Apply advanced color matching between swapped and target faces"""
    try:
        # Convert to LAB color space for better color matching
        swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_face, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Calculate statistics for each channel
        swapped_mean = np.mean(swapped_lab, axis=(0, 1))
        swapped_std = np.std(swapped_lab, axis=(0, 1))
        target_mean = np.mean(target_lab, axis=(0, 1))
        target_std = np.std(target_lab, axis=(0, 1))
        
        # Apply color transfer
        for i in range(3):
            if swapped_std[i] > 0:
                swapped_lab[:, :, i] = (swapped_lab[:, :, i] - swapped_mean[i]) * (target_std[i] / swapped_std[i]) + target_mean[i]
        
        # Convert back to BGR
        result = cv2.cvtColor(np.clip(swapped_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result
        
    except Exception:
        return swapped_face


def apply_edge_smoothing(face: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Apply edge smoothing to reduce artifacts"""
    try:
        # Create a soft mask for blending edges
        mask = np.ones(face.shape[:2], dtype=np.float32)
        
        # Apply Gaussian blur to create soft edges
        kernel_size = max(5, min(face.shape[0], face.shape[1]) // 20)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        mask = mask[:, :, np.newaxis]
        
        # Blend with reference for smoother edges
        blended = face * mask + reference * (1 - mask)
        return blended.astype(np.uint8)
        
    except Exception:
        return face


def swap_face_enhanced_with_occlusion(source_face: Face, target_face: Face, temp_frame: Frame, original_frame: Frame) -> Frame:
    """Enhanced face swapping with occlusion handling and stabilization"""
    face_swapper = get_face_swapper()
    
    try:
        # Get face bounding box
        bbox = target_face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within frame bounds
        h, w = temp_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return temp_frame
        
        # Create face mask to handle occlusion
        face_mask = create_enhanced_face_mask(target_face, temp_frame)
        
        # Apply face swap
        swapped_frame = face_swapper.get(temp_frame, target_face, source_face, paste_back=True)
        
        # Apply occlusion-aware blending
        final_frame = apply_occlusion_aware_blending(
            swapped_frame, temp_frame, face_mask, bbox
        )
        
        # Enhanced post-processing for better quality
        final_frame = enhance_face_swap_quality(final_frame, source_face, target_face, original_frame)
        
        # Apply mouth mask if enabled
        if modules.globals.mouth_mask:
            face_mask_full = create_face_mask(target_face, final_frame)
            mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
                create_lower_mouth_mask(target_face, final_frame)
            )
            final_frame = apply_mouth_area(
                final_frame, mouth_cutout, mouth_box, face_mask_full, lower_lip_polygon
            )
            
            if modules.globals.show_mouth_mask_box:
                mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
                final_frame = draw_mouth_mask_visualization(
                    final_frame, target_face, mouth_mask_data
                )
        
        return final_frame
        
    except Exception as e:
        print(f"Error in occlusion-aware face swap: {e}")
        # Fallback to regular enhanced swap
        return swap_face_enhanced(source_face, target_face, temp_frame)


def create_enhanced_face_mask(face: Face, frame: Frame) -> np.ndarray:
    """Create an enhanced face mask that better handles occlusion"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    try:
        # Use landmarks if available for more precise masking
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            landmarks = face.landmark_2d_106.astype(np.int32)
            
            # Create face contour from landmarks
            face_contour = []
            
            # Face outline (jawline and forehead)
            face_outline_indices = list(range(0, 33))  # Jawline and face boundary
            for idx in face_outline_indices:
                if idx < len(landmarks):
                    face_contour.append(landmarks[idx])
            
            if len(face_contour) > 3:
                face_contour = np.array(face_contour)
                
                # Create convex hull for smoother mask
                hull = cv2.convexHull(face_contour)
                
                # Expand the hull slightly for better coverage
                center = np.mean(hull, axis=0)
                expanded_hull = []
                for point in hull:
                    direction = point[0] - center
                    direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else direction
                    expanded_point = point[0] + direction * 10  # Expand by 10 pixels
                    expanded_hull.append(expanded_point)
                
                expanded_hull = np.array(expanded_hull, dtype=np.int32)
                cv2.fillConvexPoly(mask, expanded_hull, 255)
            else:
                # Fallback to bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        else:
            # Fallback to bounding box if no landmarks
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Apply Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
        
    except Exception as e:
        print(f"Error creating enhanced face mask: {e}")
        # Fallback to simple rectangle mask
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
    
    return mask


def apply_occlusion_aware_blending(swapped_frame: Frame, original_frame: Frame, face_mask: np.ndarray, bbox: np.ndarray) -> Frame:
    """Apply occlusion-aware blending to handle hands/objects covering the face"""
    try:
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within bounds
        h, w = swapped_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return swapped_frame
        
        # Extract face regions
        swapped_face_region = swapped_frame[y1:y2, x1:x2]
        original_face_region = original_frame[y1:y2, x1:x2]
        face_mask_region = face_mask[y1:y2, x1:x2]
        
        # Detect potential occlusion using edge detection and color analysis
        occlusion_mask = detect_occlusion(original_face_region, swapped_face_region)
        
        # Combine face mask with occlusion detection
        combined_mask = face_mask_region.astype(np.float32) / 255.0
        occlusion_factor = (255 - occlusion_mask).astype(np.float32) / 255.0
        
        # Apply occlusion-aware blending
        final_mask = combined_mask * occlusion_factor
        final_mask = final_mask[:, :, np.newaxis]
        
        # Blend the regions
        blended_region = (swapped_face_region * final_mask + 
                         original_face_region * (1 - final_mask)).astype(np.uint8)
        
        # Copy back to full frame
        result_frame = swapped_frame.copy()
        result_frame[y1:y2, x1:x2] = blended_region
        
        return result_frame
        
    except Exception as e:
        print(f"Error in occlusion-aware blending: {e}")
        return swapped_frame


def detect_occlusion(original_region: np.ndarray, swapped_region: np.ndarray) -> np.ndarray:
    """Detect potential occlusion areas (hands, objects) in the face region"""
    try:
        # Convert to different color spaces for analysis
        original_hsv = cv2.cvtColor(original_region, cv2.COLOR_BGR2HSV)
        original_lab = cv2.cvtColor(original_region, cv2.COLOR_BGR2LAB)
        
        # Detect skin-like regions (potential hands)
        # HSV ranges for skin detection
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask1 = cv2.inRange(original_hsv, lower_skin, upper_skin)
        
        lower_skin2 = np.array([160, 20, 70], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        skin_mask2 = cv2.inRange(original_hsv, lower_skin2, upper_skin2)
        
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        # Edge detection to find object boundaries
        gray = cv2.cvtColor(original_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to create thicker boundaries
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Combine skin detection and edge detection
        occlusion_mask = cv2.bitwise_or(skin_mask, edges_dilated)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_CLOSE, kernel)
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur for smooth transitions
        occlusion_mask = cv2.GaussianBlur(occlusion_mask, (11, 11), 3)
        
        return occlusion_mask
        
    except Exception as e:
        print(f"Error in occlusion detection: {e}")
        # Return empty mask if detection fails
        return np.zeros(original_region.shape[:2], dtype=np.uint8)


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    from modules.performance_optimizer import performance_optimizer
    from modules.face_tracker import face_tracker
    
    start_time = time.time()
    original_size = temp_frame.shape[:2][::-1]  # (width, height)
    
    # Apply color correction if enabled
    if modules.globals.color_correction:
        temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
    
    # Preprocess frame for performance
    processed_frame = performance_optimizer.preprocess_frame(temp_frame)
    
    if modules.globals.many_faces:
        # Only detect faces if enough time has passed or cache is empty
        if performance_optimizer.should_detect_faces():
            detected_faces = get_many_faces(processed_frame)
            # Apply tracking to each face
            tracked_faces = []
            for i, face in enumerate(detected_faces or []):
                # Use separate tracker for each face (simplified for now)
                tracked_face = face_tracker.track_face(face, processed_frame)
                if tracked_face:
                    tracked_faces.append(tracked_face)
            performance_optimizer.face_cache['many_faces'] = tracked_faces
        else:
            tracked_faces = performance_optimizer.face_cache.get('many_faces', [])
            
        if tracked_faces:
            for target_face in tracked_faces:
                if source_face and target_face:
                    processed_frame = swap_face_enhanced_with_occlusion(source_face, target_face, processed_frame, temp_frame)
                else:
                    print("Face detection failed for target/source.")
    else:
        # Use cached face detection with tracking for better performance
        if performance_optimizer.should_detect_faces():
            detected_face = get_one_face(processed_frame)
            tracked_face = face_tracker.track_face(detected_face, processed_frame)
            performance_optimizer.face_cache['single_face'] = tracked_face
        else:
            tracked_face = performance_optimizer.face_cache.get('single_face')
            
        if tracked_face and source_face:
            processed_frame = swap_face_enhanced_with_occlusion(source_face, tracked_face, processed_frame, temp_frame)
        else:
            # Try to use tracking even without detection
            tracked_face = face_tracker.track_face(None, processed_frame)
            if tracked_face and source_face:
                processed_frame = swap_face_enhanced_with_occlusion(source_face, tracked_face, processed_frame, temp_frame)
            else:
                logging.error("Face detection and tracking failed.")
    
    # Postprocess frame back to original size
    final_frame = performance_optimizer.postprocess_frame(processed_frame, original_size)
    
    # Update performance stats
    frame_time = time.time() - start_time
    performance_optimizer.update_fps_stats(frame_time)
    
    return final_frame



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
