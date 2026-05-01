# Uses ONNX Runtime for GFPGAN face enhancement (no torch/gfpgan dependency)

from typing import Any, List
import cv2
import threading
import numpy as np
import os

import onnxruntime

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face, get_many_faces
from modules.typing import Frame, Face
from modules.utilities import (
    is_image,
    is_video,
)

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-ENHANCER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

# Standard FFHQ 5-point face template for 512x512 resolution
# Points: left_eye, right_eye, nose, left_mouth, right_mouth
FFHQ_TEMPLATE_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19366],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)


def pre_check() -> bool:
    model_path = os.path.join(models_dir, "gfpgan-1024.onnx")
    if not os.path.exists(model_path):
        update_status(
            f"GFPGAN ONNX model not found at {model_path}. "
            "Please place gfpgan-1024.onnx in the models folder.",
            NAME,
        )
        return False
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_face_enhancer() -> onnxruntime.InferenceSession:
    """
    Initializes and returns the GFPGAN ONNX Runtime inference session,
    using the execution providers configured in modules.globals.
    """
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = os.path.join(models_dir, "gfpgan-1024.onnx")

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"{NAME}: Model not found at {model_path}"
                )

            try:
                from modules.processors.frame._onnx_enhancer import (
                    create_onnx_session,
                )

                FACE_ENHANCER = create_onnx_session(model_path)

                input_info = FACE_ENHANCER.get_inputs()[0]
                output_info = FACE_ENHANCER.get_outputs()[0]
                active_providers = FACE_ENHANCER.get_providers()
                print(
                    f"{NAME}: GFPGAN ONNX model loaded successfully."
                )
                print(
                    f"{NAME}: Input: {input_info.name}, "
                    f"shape: {input_info.shape}, type: {input_info.type}"
                )
                print(
                    f"{NAME}: Output: {output_info.name}, "
                    f"shape: {output_info.shape}, type: {output_info.type}"
                )
                print(f"{NAME}: Active providers: {active_providers}")

            except Exception as e:
                print(f"{NAME}: Error loading GFPGAN ONNX model: {e}")
                FACE_ENHANCER = None
                raise RuntimeError(
                    f"{NAME}: Failed to load GFPGAN ONNX model: {e}"
                )

    if FACE_ENHANCER is None:
        raise RuntimeError(
            f"{NAME}: Failed to initialize GFPGAN ONNX session. Check logs."
        )

    return FACE_ENHANCER


def _align_face(
    frame: Frame, landmarks_5: np.ndarray, output_size: int
) -> tuple:
    """
    Align and crop a face from the frame using 5-point landmarks and the
    standard FFHQ template.

    Returns:
        (aligned_face, affine_matrix) or (None, None) on failure.
    """
    # Scale the 512-base template to the desired output size
    scale = output_size / 512.0
    template = FFHQ_TEMPLATE_512 * scale

    # Estimate a similarity transform (4 DOF: rotation, scale, tx, ty)
    affine_matrix, _ = cv2.estimateAffinePartial2D(
        landmarks_5, template, method=cv2.LMEDS
    )
    if affine_matrix is None:
        return None, None

    # Warp the face to the aligned position
    aligned_face = cv2.warpAffine(
        frame,
        affine_matrix,
        (output_size, output_size),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(135, 133, 132),
    )

    return aligned_face, affine_matrix


_HAS_TORCH_CUDA = False
try:
    import torch
    if torch.cuda.is_available():
        _HAS_TORCH_CUDA = True
except ImportError:
    pass

# Cache the feathered mask — it's the same for every call at a given size
_enhancer_cache: dict = {'mask': None, 'mask_size': 0}


def _paste_back(
    frame: Frame,
    enhanced_face: np.ndarray,
    affine_matrix: np.ndarray,
    output_size: int,
) -> Frame:
    """
    Paste an enhanced (aligned) face back onto the original frame using the
    inverse affine transform with feathered-edge blending.

    Optimized: operates on a tight crop around the face bbox instead of the
    full frame, and uses GPU for blending when available.
    """
    h, w = frame.shape[:2]
    inv_matrix = cv2.invertAffineTransform(affine_matrix)

    # Build or reuse cached feathered mask (uint8 — blended via cv2 SIMD ops)
    if _enhancer_cache['mask_size'] != output_size:
        face_mask_f = np.ones((output_size, output_size), dtype=np.float32)
        border = max(1, int(output_size * 0.05))
        ramp_up = np.linspace(0.0, 1.0, border, dtype=np.float32)
        ramp_down = np.linspace(1.0, 0.0, border, dtype=np.float32)
        face_mask_f[:border, :] *= ramp_up[:, None]
        face_mask_f[-border:, :] *= ramp_down[:, None]
        face_mask_f[:, :border] *= ramp_up[None, :]
        face_mask_f[:, -border:] *= ramp_down[None, :]
        _enhancer_cache['mask'] = (face_mask_f * 255.0).astype(np.uint8)
        _enhancer_cache['mask_size'] = output_size

    # Compute tight bbox from affine corners (avoids full-frame warpAffine scan)
    corners = np.array([[0, 0], [output_size, 0],
                        [output_size, output_size], [0, output_size]],
                       dtype=np.float32)
    transformed = (inv_matrix[:, :2] @ corners.T).T + inv_matrix[:, 2]
    x1 = max(0, int(np.floor(transformed[:, 0].min())))
    x2 = min(w, int(np.ceil(transformed[:, 0].max())))
    y1 = max(0, int(np.floor(transformed[:, 1].min())))
    y2 = min(h, int(np.ceil(transformed[:, 1].max())))
    if x1 >= x2 or y1 >= y2:
        return frame

    # Pad a few pixels for feathering
    pad = max(1, int(output_size * 0.05)) + 2
    y1p, y2p = max(0, y1 - pad), min(h, y2 + pad)
    x1p, x2p = max(0, x1 - pad), min(w, x2 + pad)
    crop_w, crop_h = x2p - x1p, y2p - y1p

    # Warp enhanced face and mask into crop space only
    inv_crop = inv_matrix.copy()
    inv_crop[0, 2] -= x1p
    inv_crop[1, 2] -= y1p

    inv_restored_crop = cv2.warpAffine(
        enhanced_face, inv_crop, (crop_w, crop_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    inv_mask_crop = cv2.warpAffine(
        _enhancer_cache['mask'], inv_crop, (crop_w, crop_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    target_crop = frame[y1p:y2p, x1p:x2p]

    if _HAS_TORCH_CUDA:
        # Upload uint8 alpha — smaller transfer, scale on device.
        mask_t = torch.from_numpy(inv_mask_crop).cuda().float().mul_(1.0 / 255.0).unsqueeze(2)
        enhanced_t = torch.from_numpy(inv_restored_crop).float().cuda()
        target_t = torch.from_numpy(target_crop).float().cuda()
        blended = (mask_t * enhanced_t + (1.0 - mask_t) * target_t
                   ).to(torch.uint8).cpu().numpy()
        frame[y1p:y2p, x1p:x2p] = blended
    else:
        # Fused uint8 blend via cv2 SIMD — ~7× faster than the float32 round-trip.
        alpha_3c = cv2.merge([inv_mask_crop, inv_mask_crop, inv_mask_crop])
        inv_alpha = 255 - alpha_3c
        a_enh = cv2.multiply(inv_restored_crop, alpha_3c, scale=1.0 / 255.0)
        a_tgt = cv2.multiply(target_crop, inv_alpha, scale=1.0 / 255.0)
        frame[y1p:y2p, x1p:x2p] = cv2.add(a_enh, a_tgt)

    return frame


def _preprocess_face(aligned_face: np.ndarray) -> np.ndarray:
    """
    Convert an aligned BGR uint8 face image to the ONNX model input tensor.
    Format: NCHW float32, normalised to [-1, 1].
    """
    # BGR -> RGB, normalize, and transpose in one pass
    # Fused: (x / 255.0 - 0.5) / 0.5 = x / 127.5 - 1.0
    rgb = aligned_face[:, :, ::-1]  # BGR->RGB zero-copy view
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
    chw *= (1.0 / 127.5)
    chw -= 1.0
    return chw[np.newaxis, ...]  # shape: (1, 3, H, W)


def _postprocess_face(output: np.ndarray) -> np.ndarray:
    """
    Convert the ONNX model output tensor back to a BGR uint8 image.
    Expects input in NCHW format with values in [-1, 1].
    """
    # Fused: ((x + 1.0) / 2.0) * 255 = (x + 1.0) * 127.5
    face = output[0]  # remove batch dim -> (3, H, W)
    face = (face + 1.0) * 127.5
    np.clip(face, 0, 255, out=face)
    face = face.astype(np.uint8).transpose(1, 2, 0)  # CHW -> HWC
    return face[:, :, ::-1].copy()  # RGB -> BGR


# Cache for temporal enhancement skipping in live mode.
# GFPGAN output barely changes between consecutive frames (same face,
# same position), so we run inference every _ENH_INTERVAL frames and
# reuse the cached enhanced face + affine matrix in between.
_enh_live_cache: dict = {
    'enhanced_bgr': None,
    'affine_matrix': None,
    'align_size': 0,
    'frame_count': 0,
}
_ENH_INTERVAL = 2  # run inference every N frames, paste cached result otherwise


def enhance_face(temp_frame: Frame, detected_faces=None) -> Frame:
    """Enhances all faces in a frame using the GFPGAN ONNX model.

    Args:
        detected_faces: Pre-detected face list. When provided, skips
            the internal detection call (saves ~15-20ms per frame).
            Also enables temporal caching — inference runs every
            _ENH_INTERVAL frames, reusing the cached result otherwise.
    """
    session = get_face_enhancer()

    # Determine model input resolution from the session metadata
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape  # e.g. [1, 3, 512, 512]
    try:
        align_size = int(input_shape[2])
        if align_size <= 0:
            align_size = 512
    except (ValueError, TypeError, IndexError):
        align_size = 512

    # Use pre-detected faces if available, otherwise detect
    faces = detected_faces if detected_faces is not None else get_many_faces(temp_frame)
    if not faces:
        return temp_frame

    # Temporal caching: only available when faces are pre-detected (live mode)
    # AND we're in single-face mode — the cache holds exactly one enhancement,
    # so reusing it in many_faces mode would paste the same face onto every
    # detected target.
    many_faces_mode = getattr(modules.globals, "many_faces", False)
    use_cache = detected_faces is not None and not many_faces_mode
    if use_cache:
        _enh_live_cache['frame_count'] += 1
        run_inference_this_frame = (_enh_live_cache['frame_count'] % _ENH_INTERVAL == 0
                                   or _enh_live_cache['enhanced_bgr'] is None)
    else:
        run_inference_this_frame = True

    for face in faces:
        if not hasattr(face, "kps") or face.kps is None:
            continue

        landmarks_5 = face.kps.astype(np.float32)
        if landmarks_5.shape[0] < 5:
            continue

        if run_inference_this_frame:
            aligned_face, affine_matrix = _align_face(
                temp_frame, landmarks_5, output_size=align_size
            )
            if aligned_face is None or affine_matrix is None:
                continue

            try:
                with THREAD_SEMAPHORE:
                    from modules.processors.frame._onnx_enhancer import (
                        run_inference,
                    )
                    input_tensor = _preprocess_face(aligned_face)
                    output_tensor = run_inference(session, input_name, input_tensor)
                    enhanced_bgr = _postprocess_face(output_tensor)

                eh, ew = enhanced_bgr.shape[:2]
                if eh != align_size or ew != align_size:
                    enhanced_bgr = cv2.resize(
                        enhanced_bgr,
                        (align_size, align_size),
                        interpolation=cv2.INTER_LANCZOS4,
                    )

                # Cache for reuse on next frame
                if use_cache:
                    _enh_live_cache['enhanced_bgr'] = enhanced_bgr
                    _enh_live_cache['affine_matrix'] = affine_matrix
                    _enh_live_cache['align_size'] = align_size

                _paste_back(
                    temp_frame, enhanced_bgr, affine_matrix, output_size=align_size
                )
            except Exception as e:
                print(f"{NAME}: Error enhancing a face: {e}")
                continue
        else:
            # Reuse cached enhanced face — just paste back onto current frame
            cached = _enh_live_cache
            if cached['enhanced_bgr'] is not None:
                _paste_back(
                    temp_frame, cached['enhanced_bgr'],
                    cached['affine_matrix'],
                    output_size=cached['align_size'],
                )
        if not many_faces_mode:
            break  # single-face live mode — only process first face

    return temp_frame


def process_frame(source_face: Face | None, temp_frame: Frame,
                   detected_faces=None) -> Frame:
    """Processes a frame: enhances face if detected."""
    return enhance_face(temp_frame, detected_faces=detected_faces)


def process_frame_v2(temp_frame: Frame, detected_faces=None) -> Frame:
    """Processes a frame without source face (used by live webcam preview)."""
    return enhance_face(temp_frame, detected_faces=detected_faces)


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
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(
    source_path: str | None, temp_frame_paths: List[str]
) -> None:
    """Processes video frames using the frame processor core."""
    modules.processors.frame.core.process_video(
        source_path, temp_frame_paths, process_frames
    )
