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
    swapped_frame_result = face_swapper.get( # Renamed to avoid confusion
        temp_frame, target_face, source_face, paste_back=True
    )

    # Ensure swapped_frame_result is not None and is a valid image
    if swapped_frame_result is None or not isinstance(swapped_frame_result, np.ndarray):
        logging.error("Face swap operation failed or returned invalid result.")
        return temp_frame # Return original frame if swap failed

    # Color Correction
    if modules.globals.color_correction:
        # Get the bounding box of the target face to apply color correction
        # more accurately to the swapped region.
        # The target_face object should have bbox attribute (x1, y1, x2, y2)
        if hasattr(target_face, 'bbox'):
            x1, y1, x2, y2 = target_face.bbox.astype(int)
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(swapped_frame_result.shape[1], x2), min(swapped_frame_result.shape[0], y2)

            if x1 < x2 and y1 < y2:
                swapped_face_region = swapped_frame_result[y1:y2, x1:x2]
                target_face_region_original = temp_frame[y1:y2, x1:x2]

                if swapped_face_region.size > 0 and target_face_region_original.size > 0:
                    corrected_swapped_face_region = apply_histogram_matching_color_correction(swapped_face_region, target_face_region_original)
                    swapped_frame_result[y1:y2, x1:x2] = corrected_swapped_face_region
                else:
                    # Fallback to full frame color correction if regions are invalid
                    swapped_frame_result = apply_histogram_matching_color_correction(swapped_frame_result, temp_frame)
            else:
                # Fallback to full frame color correction if bbox is invalid
                swapped_frame_result = apply_histogram_matching_color_correction(swapped_frame_result, temp_frame)
        else:
            # Fallback to full frame color correction if no bbox
            swapped_frame_result = apply_histogram_matching_color_correction(swapped_frame_result, temp_frame)

    if modules.globals.mouth_mask:
        # Create a mask for the target face
        face_mask = create_face_mask(target_face, temp_frame)

        # Create the mouth mask
        mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon = (
            create_lower_mouth_mask(target_face, temp_frame)
        )

        # Apply the mouth area
        swapped_frame_result = apply_mouth_area(
            swapped_frame_result, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
        )

        if modules.globals.show_mouth_mask_box:
            mouth_mask_data = (mouth_mask, mouth_cutout, mouth_box, lower_lip_polygon)
            swapped_frame_result = draw_mouth_mask_visualization(
                swapped_frame_result, target_face, mouth_mask_data
            )

    # Poisson Blending
    if modules.globals.use_poisson_blending and hasattr(target_face, 'bbox'):
        # Create a mask for the swapped face region for Poisson blending
        # This mask should cover the area of the swapped face.
        # We can use the target_face.bbox and perhaps expand it slightly,
        # or use a more precise mask from face parsing if available.
        # For simplicity, using a slightly feathered convex hull of landmarks.

        face_mask_for_blending = np.zeros(temp_frame.shape[:2], dtype=np.uint8)

        # Prioritize using the bounding box for a tighter mask
        if hasattr(target_face, 'bbox'):
            x1, y1, x2, y2 = target_face.bbox.astype(int)
            # Ensure coordinates are within frame bounds
            x1_b, y1_b = max(0, x1), max(0, y1) # Use different var names to avoid conflict with center calculation
            x2_b, y2_b = min(temp_frame.shape[1], x2), min(temp_frame.shape[0], y2)

            # Create a rectangular mask based on the bounding box
            if x1_b < x2_b and y1_b < y2_b:
                face_mask_for_blending[y1_b:y2_b, x1_b:x2_b] = 255
            else:
                logging.warning("Invalid bounding box for Poisson mask. Attempting landmark-based mask.")
                # Fallback to landmark-based convex hull if bbox is invalid
                landmarks = target_face.landmark_2d_106 if hasattr(target_face, 'landmark_2d_106') else None
                if landmarks is not None and len(landmarks) > 0:
                    try:
                        hull_points = cv2.convexHull(landmarks.astype(np.int32))
                        cv2.fillConvexPoly(face_mask_for_blending, hull_points, 255)
                    except Exception as e:
                        logging.error(f"Could not form convex hull for Poisson mask from landmarks: {e}. Blending will be skipped.")
                else:
                    logging.error("No valid bbox or landmarks for Poisson mask. Blending will be skipped.")
        else:
            # Fallback to landmark-based convex hull if no bbox attribute
            landmarks = target_face.landmark_2d_106 if hasattr(target_face, 'landmark_2d_106') else None
            if landmarks is not None and len(landmarks) > 0:
                try:
                    hull_points = cv2.convexHull(landmarks.astype(np.int32))
                    cv2.fillConvexPoly(face_mask_for_blending, hull_points, 255)
                except Exception as e:
                    logging.error(f"Could not form convex hull for Poisson mask from landmarks (no bbox): {e}. Blending will be skipped.")
            else:
                logging.error("No bbox or landmarks available for Poisson mask. Blending will be skipped.")

        # Subtract ear regions if preserve_target_ears is enabled
        if modules.globals.preserve_target_ears and np.any(face_mask_for_blending > 0):
            mfx1, mfy1, mfx2, mfy2 = target_face.bbox.astype(int)
            mfw = mfx2 - mfx1
            mfh = mfy2 - mfy1

            ear_w = int(mfw * modules.globals.ear_width_ratio)
            ear_h = int(mfh * modules.globals.ear_height_ratio)
            ear_v_offset = int(mfh * modules.globals.ear_vertical_offset_ratio)
            ear_overlap = int(mfw * modules.globals.ear_horizontal_overlap_ratio)

            # Person's Right Ear (image left side of face bbox)
            # This region in face_mask_for_blending will be set to 0
            rex1 = max(0, mfx1 - ear_w + ear_overlap)
            rey1 = max(0, mfy1 + ear_v_offset)
            rex2 = min(temp_frame.shape[1], mfx1 + ear_overlap) # Extends slightly into face bbox for smoother transition
            rey2 = min(temp_frame.shape[0], rey1 + ear_h)
            if rex1 < rex2 and rey1 < rey2:
                cv2.rectangle(face_mask_for_blending, (rex1, rey1), (rex2, rey2), 0, -1)

            # Person's Left Ear (image right side of face bbox)
            lex1 = max(0, mfx2 - ear_overlap)
            ley1 = max(0, mfy1 + ear_v_offset)
            lex2 = min(temp_frame.shape[1], mfx2 + ear_w - ear_overlap)
            ley2 = min(temp_frame.shape[0], ley1 + ear_h)
            if lex1 < lex2 and ley1 < ley2:
                cv2.rectangle(face_mask_for_blending, (lex1, ley1), (lex2, ley2), 0, -1)

        # Feather the mask to smooth edges for Poisson blending
        if np.any(face_mask_for_blending > 0): # Only feather if there's a mask
            feather_amount = modules.globals.poisson_blending_feather_amount
            if feather_amount > 0:
                # Ensure kernel size is odd
                kernel_size = 2 * feather_amount + 1
                face_mask_for_blending = cv2.GaussianBlur(face_mask_for_blending, (kernel_size, kernel_size), 0)

        # Calculate the center of the target face bbox for seamlessClone
        if hasattr(target_face, 'bbox'):
            x1, y1, x2, y2 = target_face.bbox.astype(int)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Ensure center is within frame dimensions
            center_x = np.clip(center_x, 0, temp_frame.shape[1] -1)
            center_y = np.clip(center_y, 0, temp_frame.shape[0] -1)
            center = (center_x, center_y)

            # Apply Poisson blending
            # swapped_frame_result is the source, temp_frame is the destination
            if np.any(face_mask_for_blending > 0): # Proceed only if mask is not empty
                try:
                    # Ensure swapped_frame_result and temp_frame are 8-bit 3-channel images
                    if swapped_frame_result.dtype != np.uint8:
                        swapped_frame_result = np.clip(swapped_frame_result, 0, 255).astype(np.uint8)
                    if temp_frame.dtype != np.uint8:
                        temp_frame_uint8 = np.clip(temp_frame, 0, 255).astype(np.uint8)
                    else:
                        temp_frame_uint8 = temp_frame

                    swapped_frame_result = cv2.seamlessClone(swapped_frame_result, temp_frame_uint8, face_mask_for_blending, center, cv2.NORMAL_CLONE)
                except cv2.error as e:
                    logging.error(f"Error during Poisson blending: {e}")
                    # Fallback to non-blended result if seamlessClone fails
                    pass # swapped_frame_result remains as is
            else:
                logging.warning("Poisson blending mask is empty. Skipping Poisson blending.")

    return swapped_frame_result


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    # The color_correction logic was moved into swap_face.
    # The initial temp_frame modification `cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)`
    # was incorrect as it changes the color space of the whole frame before processing,
    # which is not what we want for color correction of the swapped part.
    # Histogram matching is now done BGR to BGR.

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


def apply_histogram_matching_color_correction(source_img: Frame, target_img: Frame) -> Frame:
    """
    Applies color correction to the source image to match the target image's color distribution
    using histogram matching on each color channel.
    """
    corrected_img = np.zeros_like(source_img)
    for i in range(source_img.shape[2]):  # Iterate over color channels (B, G, R)
        source_hist, _ = np.histogram(source_img[:, :, i].flatten(), 256, [0, 256])
        target_hist, _ = np.histogram(target_img[:, :, i].flatten(), 256, [0, 256])

        # Compute cumulative distribution functions (CDFs)
        source_cdf = source_hist.cumsum()
        source_cdf_normalized = source_cdf * source_hist.max() / source_cdf.max() # Normalize

        target_cdf = target_hist.cumsum()
        target_cdf_normalized = target_cdf * target_hist.max() / target_cdf.max() # Normalize

        # Create lookup table
        lookup_table = np.zeros(256, 'uint8')

        gj = 0
        for gi in range(256):
            while gj < 256 and target_cdf_normalized[gj] < source_cdf_normalized[gi]:
                gj += 1
            if gj == 256: # If we reach end of target_cdf, map remaining to max value
                lookup_table[gi] = 255
            else:
                lookup_table[gi] = gj

        corrected_img[:, :, i] = cv2.LUT(source_img[:, :, i], lookup_table)

    return corrected_img
