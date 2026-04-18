# --- START OF FILE hair_style_modifier.py ---
# Hair Style Modifier - Modifies hair style and color in images/videos
# Uses image processing techniques and optional ONNX models for hair manipulation

from typing import Any, List, Optional, Tuple
import cv2
import threading
import numpy as np
import os

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces
from modules.typing import Frame, Face
from modules.utilities import (
    is_image,
    is_video,
)
from modules.gpu_processing import gpu_gaussian_blur, gpu_add_weighted, gpu_resize

HAIR_STYLE_MODIFIER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = "DLC.HAIR-STYLE-MODIFIER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

HAIR_COLORS = {
    "none": None,
    "blonde": (255, 230, 180),
    "light_blonde": (255, 245, 220),
    "brown": (160, 110, 70),
    "light_brown": (190, 150, 110),
    "dark_brown": (100, 60, 30),
    "black": (30, 30, 30),
    "red": (180, 50, 50),
    "light_red": (220, 100, 80),
    "burgundy": (130, 30, 80),
    "blue": (50, 80, 180),
    "light_blue": (100, 150, 220),
    "green": (50, 150, 80),
    "purple": (120, 50, 150),
    "pink": (255, 150, 180),
    "white": (240, 240, 240),
    "gray": (150, 150, 150),
}

HAIR_STYLES = {
    "none": "No style modification",
    "bob": "Bob cut (short style)",
    "long": "Long hair extension",
    "curly": "Curly hair effect",
    "straight": "Straight hair effect",
    "pixie": "Pixie cut (very short)",
    "layered": "Layered hair effect",
    "wavy": "Wavy hair effect",
}


def pre_check() -> bool:
    update_status("Hair Style Modifier ready.", NAME)
    return True


def pre_start() -> bool:
    hair_style = getattr(modules.globals, "hair_style", "none")
    hair_color = getattr(modules.globals, "hair_color", "none")
    
    if hair_style == "none" and hair_color == "none":
        update_status("Hair style and color both set to 'none'. Skipping hair modification.", NAME)
        return True
    
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    
    return True


def get_hair_region_mask(frame: Frame, face: Face) -> Optional[np.ndarray]:
    """
    Detects the hair region based on face landmarks and color analysis.
    
    Args:
        frame: The input frame
        face: Detected face object with landmarks
    
    Returns:
        Binary mask of the hair region, or None if detection fails
    """
    if face is None:
        return None
    
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if not hasattr(face, "landmark_2d_106") or face.landmark_2d_106 is None:
        return None
    
    landmarks = face.landmark_2d_106.astype(np.float32)
    
    if landmarks.shape[0] < 106:
        return None
    
    try:
        forehead_landmarks = landmarks[33:43]
        left_eyebrow = landmarks[33:38]
        right_eyebrow = landmarks[38:43]
        
        left_eye = landmarks[52:58]
        right_eye = landmarks[58:64]
        
        eye_center_left = np.mean(left_eye, axis=0)
        eye_center_right = np.mean(right_eye, axis=0)
        eye_center = (eye_center_left + eye_center_right) / 2
        
        chin = landmarks[16]
        
        face_height = np.linalg.norm(chin - eye_center) * 2
        face_width = np.linalg.norm(eye_center_left - eye_center_right) * 2.5
        
        eyebrow_center = np.mean(forehead_landmarks, axis=0)
        
        hair_top_y = max(0, int(eyebrow_center[1] - face_height * 0.8))
        hair_bottom_y = int(eyebrow_center[1] + face_height * 0.1)
        hair_left_x = max(0, int(eyebrow_center[0] - face_width * 0.7))
        hair_right_x = min(w, int(eyebrow_center[0] + face_width * 0.7))
        
        hair_region_points = []
        
        left_brow_min = np.min(left_eyebrow[:, 0])
        right_brow_max = np.max(right_eyebrow[:, 0])
        brow_avg_y = np.mean(forehead_landmarks[:, 1])
        
        hair_region_points.append([hair_left_x, brow_avg_y])
        
        step = (brow_avg_y - hair_top_y) / 10
        for i in range(10):
            y = brow_avg_y - step * (i + 1)
            offset = (i + 1) * 0.05 * face_width
            hair_region_points.append([hair_left_x - offset, y])
        
        hair_region_points.append([eyebrow_center[0], hair_top_y])
        
        for i in range(10):
            y = brow_avg_y - step * (10 - i)
            offset = (10 - i) * 0.05 * face_width
            hair_region_points.append([hair_right_x + offset, y])
        
        hair_region_points.append([hair_right_x, brow_avg_y])
        
        hair_region_points.append([right_brow_max, brow_avg_y])
        hair_region_points.append([right_brow_max, brow_avg_y + face_height * 0.05])
        hair_region_points.append([eyebrow_center[0], brow_avg_y + face_height * 0.1])
        hair_region_points.append([left_brow_min, brow_avg_y + face_height * 0.05])
        hair_region_points.append([left_brow_min, brow_avg_y])
        
        hair_polygon = np.array(hair_region_points, dtype=np.int32)
        
        cv2.fillPoly(mask, [hair_polygon], 255)
        
        try:
            hair_mask_refined = refine_hair_mask_by_color(frame, mask)
            if hair_mask_refined is not None and np.sum(hair_mask_refined) > 0:
                mask = hair_mask_refined
        except Exception:
            pass
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = gpu_gaussian_blur(mask, (15, 15), 0)
        
        return mask
        
    except Exception as e:
        print(f"{NAME}: Error in hair region detection: {e}")
        return None


def refine_hair_mask_by_color(frame: Frame, initial_mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Refines the hair mask using color analysis in the YCrCb color space.
    Hair typically has distinct color characteristics compared to skin.
    """
    if initial_mask is None or np.sum(initial_mask) == 0:
        return None
    
    h, w = frame.shape[:2]
    
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    y_channel = ycrcb[:, :, 0].astype(np.float32)
    cr_channel = ycrcb[:, :, 1].astype(np.float32)
    cb_channel = ycrcb[:, :, 2].astype(np.float32)
    
    hair_pixels = frame[initial_mask > 128]
    if len(hair_pixels) == 0:
        return initial_mask
    
    hair_pixels_ycrcb = cv2.cvtColor(hair_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    
    mean_y = np.mean(hair_pixels_ycrcb[:, 0])
    mean_cr = np.mean(hair_pixels_ycrcb[:, 1])
    mean_cb = np.mean(hair_pixels_ycrcb[:, 2])
    
    std_y = np.std(hair_pixels_ycrcb[:, 0])
    std_cr = np.std(hair_pixels_ycrcb[:, 1])
    std_cb = np.std(hair_pixels_ycrcb[:, 2])
    
    y_low = max(0, mean_y - 2 * std_y - 20)
    y_high = min(255, mean_y + 2 * std_y + 20)
    cr_low = max(0, mean_cr - 2 * std_cr - 10)
    cr_high = min(255, mean_cr + 2 * std_cr + 10)
    cb_low = max(0, mean_cb - 2 * std_cb - 10)
    cb_high = min(255, mean_cb + 2 * std_cb + 10)
    
    color_mask = np.zeros((h, w), dtype=np.uint8)
    
    y_mask = (y_channel >= y_low) & (y_channel <= y_high)
    cr_mask = (cr_channel >= cr_low) & (cr_channel <= cr_high)
    cb_mask = (cb_channel >= cb_low) & (cb_channel <= cb_high)
    
    combined_mask = y_mask & cr_mask & cb_mask
    
    color_mask[combined_mask] = 255
    
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    
    final_mask = cv2.bitwise_and(initial_mask, color_mask)
    
    return final_mask


def apply_hair_color(frame: Frame, hair_mask: np.ndarray, target_color_bgr: Tuple[int, int, int], intensity: float = 0.5) -> Frame:
    """
    Applies a new color to the hair region.
    
    Args:
        frame: Input frame
        hair_mask: Binary mask of hair region
        target_color_bgr: Target BGR color tuple
        intensity: Color blending intensity (0.0-1.0)
    
    Returns:
        Frame with modified hair color
    """
    if hair_mask is None or np.sum(hair_mask) == 0:
        return frame
    
    h, w = frame.shape[:2]
    
    frame_float = frame.astype(np.float32)
    target_color_float = np.array(target_color_bgr, dtype=np.float32)
    
    hair_mask_normalized = (hair_mask / 255.0).astype(np.float32)
    hair_mask_3d = np.stack([hair_mask_normalized] * 3, axis=-1)
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    
    target_color_3d = np.full((h, w, 3), target_color_float, dtype=np.float32)
    
    l_mean = np.mean(l_channel[hair_mask > 128]) if np.sum(hair_mask > 128) > 0 else 128
    
    target_color_lab = cv2.cvtColor(
        target_color_3d.reshape(1, h * w, 3).astype(np.uint8),
        cv2.COLOR_BGR2LAB
    ).reshape(h, w, 3).astype(np.float32)
    
    target_color_lab[:, :, 0] = np.clip(target_color_lab[:, :, 0] * (l_mean / 128.0), 0, 255)
    
    color_modified_bgr = cv2.cvtColor(
        target_color_lab.astype(np.uint8),
        cv2.COLOR_LAB2BGR
    ).astype(np.float32)
    
    intensity_adjusted = intensity * hair_mask_3d
    blended = (1.0 - intensity_adjusted) * frame_float + intensity_adjusted * color_modified_bgr
    
    result = np.clip(blended, 0, 255).astype(np.uint8)
    
    return result


def apply_hair_style_effect(frame: Frame, hair_mask: np.ndarray, hair_style: str, intensity: float = 0.5) -> Frame:
    """
    Applies visual effects to simulate different hair styles.
    
    Args:
        frame: Input frame
        hair_mask: Binary mask of hair region
        hair_style: Style to apply
        intensity: Effect intensity
    
    Returns:
        Frame with hair style effect applied
    """
    if hair_mask is None or np.sum(hair_mask) == 0:
        return frame
    
    h, w = frame.shape[:2]
    result = frame.copy()
    
    hair_mask_3d = np.stack([hair_mask / 255.0] * 3, axis=-1).astype(np.float32)
    
    if hair_style == "curly" or hair_style == "wavy":
        result = apply_curly_effect(result, hair_mask, intensity, hair_style == "wavy")
    
    elif hair_style == "straight":
        result = apply_straight_effect(result, hair_mask, intensity)
    
    elif hair_style == "bob" or hair_style == "pixie":
        result = apply_short_hair_effect(result, hair_mask, intensity, hair_style == "pixie")
    
    elif hair_style == "long":
        result = apply_long_hair_effect(result, hair_mask, intensity)
    
    elif hair_style == "layered":
        result = apply_layered_effect(result, hair_mask, intensity)
    
    return result


def apply_curly_effect(frame: Frame, hair_mask: np.ndarray, intensity: float, is_wavy: bool = False) -> Frame:
    """Applies a curly/wavy effect to the hair region using wave distortion."""
    h, w = frame.shape[:2]
    result = frame.copy()
    
    freq = 30 if is_wavy else 20
    amplitude = (15 if is_wavy else 25) * intensity
    
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    wave_x = amplitude * np.sin(2 * np.pi * y_coords / freq)
    wave_y = amplitude * 0.3 * np.sin(2 * np.pi * x_coords / (freq * 0.7))
    
    map_x = np.clip(x_coords + wave_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(y_coords + wave_y, 0, h - 1).astype(np.float32)
    
    distorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    hair_mask_normalized = (hair_mask / 255.0).astype(np.float32) * intensity
    hair_mask_3d = np.stack([hair_mask_normalized] * 3, axis=-1)
    
    result = (1.0 - hair_mask_3d) * result.astype(np.float32) + hair_mask_3d * distorted.astype(np.float32)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def apply_straight_effect(frame: Frame, hair_mask: np.ndarray, intensity: float) -> Frame:
    """Applies a straightening effect by reducing texture variations."""
    h, w = frame.shape[:2]
    
    hair_region = frame.astype(np.float32)
    
    blurred = gpu_gaussian_blur(hair_region, (int(5 * intensity + 1) | 1, int(5 * intensity + 1) | 1), 0)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    
    edge_mask = (edge_magnitude > 20).astype(np.float32) * intensity
    edge_mask_3d = np.stack([edge_mask] * 3, axis=-1)
    
    hair_mask_normalized = (hair_mask / 255.0).astype(np.float32)
    hair_mask_3d = np.stack([hair_mask_normalized] * 3, axis=-1)
    
    blend_factor = hair_mask_3d * (1.0 - edge_mask_3d * 0.3)
    result = (1.0 - blend_factor) * frame.astype(np.float32) + blend_factor * blurred.astype(np.float32)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def apply_short_hair_effect(frame: Frame, hair_mask: np.ndarray, intensity: float, is_pixie: bool) -> Frame:
    """Simulates short hair by adjusting the hair mask and applying stylized effects."""
    h, w = frame.shape[:2]
    result = frame.copy()
    
    hair_mask_normalized = (hair_mask / 255.0).astype(np.float32)
    
    contours, _ = cv2.findContours(
        (hair_mask_normalized * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) > 0:
        main_contour = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(main_contour)
        
        crop_factor = 0.3 if is_pixie else 0.2
        crop_height = int(ch * crop_factor * intensity)
        
        if crop_height > 0:
            transition_y = y + ch - crop_height
            transition_gradient = np.linspace(0, 1, crop_height)
            transition_gradient = np.clip(transition_gradient * 2, 0, 1)
            
            for gy in range(crop_height):
                yy = transition_y + gy
                if 0 <= yy < h:
                    hair_mask_normalized[yy, :] *= transition_gradient[gy]
    
    hair_mask_3d = np.stack([hair_mask_normalized] * 3, axis=-1)
    
    sharpened = cv2.GaussianBlur(frame, (0, 0), 3)
    sharpened = cv2.addWeighted(frame, 1.5, sharpened, -0.5, 0)
    
    result = (1.0 - hair_mask_3d * intensity * 0.3) * result.astype(np.float32) + \
             hair_mask_3d * intensity * 0.3 * sharpened.astype(np.float32)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def apply_long_hair_effect(frame: Frame, hair_mask: np.ndarray, intensity: float) -> Frame:
    """Simulates longer hair by extending and smoothing the hair region."""
    h, w = frame.shape[:2]
    result = frame.copy()
    
    hair_mask_normalized = (hair_mask / 255.0).astype(np.float32)
    
    kernel = np.ones((int(20 * intensity + 1), int(10 * intensity + 1)), np.uint8)
    dilated_mask = cv2.dilate(hair_mask_normalized, kernel, iterations=1)
    
    contours, _ = cv2.findContours(
        (hair_mask_normalized * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    extend_mask = np.zeros_like(hair_mask_normalized)
    if len(contours) > 0:
        main_contour = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(main_contour)
        
        extend_amount = int(ch * 0.3 * intensity)
        bottom_y = y + ch
        
        if bottom_y + extend_amount < h:
            for gy in range(extend_amount):
                yy = bottom_y + gy
                if 0 <= yy < h:
                    alpha = 1.0 - (gy / extend_amount) * 0.5
                    extend_mask[yy, x:x+cw] = alpha
    
    combined_mask = np.maximum(dilated_mask, extend_mask)
    combined_mask_3d = np.stack([combined_mask] * 3, axis=-1)
    
    blurred = gpu_gaussian_blur(frame, (7, 7), 0)
    
    result = (1.0 - combined_mask_3d * 0.2) * result.astype(np.float32) + \
             combined_mask_3d * 0.2 * blurred.astype(np.float32)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def apply_layered_effect(frame: Frame, hair_mask: np.ndarray, intensity: float) -> Frame:
    """Simulates layered hair by adding texture variations."""
    h, w = frame.shape[:2]
    result = frame.copy()
    
    hair_mask_normalized = (hair_mask / 255.0).astype(np.float32)
    hair_mask_3d = np.stack([hair_mask_normalized] * 3, axis=-1)
    
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    layers = 5
    layer_pattern = np.zeros((h, w), dtype=np.float32)
    
    for i in range(layers):
        freq = 40 + i * 15
        amplitude = 5 + i * 3
        layer_pattern += amplitude * np.sin(2 * np.pi * (y_coords + x_coords * 0.3) / freq)
    
    layer_pattern = (layer_pattern - layer_pattern.min()) / (layer_pattern.max() - layer_pattern.min() + 1e-8)
    layer_pattern = layer_pattern * intensity * 0.3
    layer_pattern_3d = np.stack([layer_pattern] * 3, axis=-1)
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    
    l_modified = np.clip(l_channel + (layer_pattern - 0.15) * 50, 0, 255)
    lab[:, :, 0] = l_modified
    
    modified_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    
    result = (1.0 - hair_mask_3d) * result.astype(np.float32) + hair_mask_3d * modified_bgr
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def modify_hair(frame: Frame) -> Frame:
    """
    Main function to modify hair style and color for all faces in a frame.
    """
    hair_style = getattr(modules.globals, "hair_style", "none")
    hair_color = getattr(modules.globals, "hair_color", "none")
    hair_opacity = getattr(modules.globals, "hair_opacity", 1.0)
    color_intensity = getattr(modules.globals, "hair_color_intensity", 0.5)
    curl_intensity = getattr(modules.globals, "hair_curl_intensity", 0.5)
    
    if hair_style == "none" and hair_color == "none":
        return frame
    
    faces = get_many_faces(frame)
    if not faces:
        return frame
    
    result_frame = frame.copy()
    
    for face in faces:
        hair_mask = get_hair_region_mask(result_frame, face)
        
        if hair_mask is None or np.sum(hair_mask) == 0:
            continue
        
        temp_frame = result_frame.copy()
        
        if hair_color != "none" and hair_color in HAIR_COLORS:
            target_color = HAIR_COLORS[hair_color]
            if target_color is not None:
                temp_frame = apply_hair_color(
                    temp_frame, 
                    hair_mask, 
                    target_color, 
                    color_intensity
                )
        
        if hair_style != "none" and hair_style in HAIR_STYLES:
            temp_frame = apply_hair_style_effect(
                temp_frame, 
                hair_mask, 
                hair_style, 
                curl_intensity
            )
        
        if hair_opacity < 1.0:
            hair_mask_normalized = (hair_mask / 255.0).astype(np.float32) * hair_opacity
            hair_mask_3d = np.stack([hair_mask_normalized] * 3, axis=-1)
            
            result_frame = (1.0 - hair_mask_3d) * result_frame.astype(np.float32) + \
                          hair_mask_3d * temp_frame.astype(np.float32)
            result_frame = np.clip(result_frame, 0, 255).astype(np.uint8)
        else:
            hair_mask_normalized = (hair_mask / 255.0).astype(np.float32)
            hair_mask_3d = np.stack([hair_mask_normalized] * 3, axis=-1)
            
            result_frame = (1.0 - hair_mask_3d) * result_frame.astype(np.float32) + \
                          hair_mask_3d * temp_frame.astype(np.float32)
            result_frame = np.clip(result_frame, 0, 255).astype(np.uint8)
    
    return result_frame


def process_frame(source_face: Face | None, temp_frame: Frame, target_face: Face = None) -> Frame:
    """Processes a frame: modifies hair style/color if detected."""
    temp_frame = modify_hair(temp_frame)
    return temp_frame


def process_frame_v2(temp_frame: Frame) -> Frame:
    """Processes a frame without source face (used by live webcam preview)."""
    return modify_hair(temp_frame)


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """Processes multiple frames from file paths."""
    for temp_frame_path in temp_frame_paths:
        if not os.path.exists(temp_frame_path):
            print(
                f"{NAME}: Warning: Frame path not found {temp_frame_path}, skipping."
            )
            if progress:
                progress.update(1)
            continue

        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            print(
                f"{NAME}: Warning: Failed to read frame {temp_frame_path}, skipping."
            )
            if progress:
                progress.update(1)
            continue

        result_frame = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result_frame)
        if progress:
            progress.update(1)


def process_image(
    source_path: str | None, target_path: str, output_path: str
) -> None:
    """Processes a single image file."""
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    cv2.imwrite(output_path, result_frame)
    print(f"{NAME}: Hair modified image saved to {output_path}")


def process_video(
    source_path: str | None, temp_frame_paths: List[str]
) -> None:
    """Processes video frames using the frame processor core."""
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames
    )


# --- END OF FILE hair_style_modifier.py ---