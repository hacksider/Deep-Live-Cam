from typing import Any
import cv2
import modules.globals  # Import the globals to check the color correction toggle
from modules.gpu_processing import gpu_cvt_color


def get_video_frame_last_index(frame_total: Any) -> int:
    try:
        frame_total_int = int(frame_total)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(frame_total_int - 1, 0)


def clamp_video_frame_number(frame_number: Any, frame_total: Any) -> int:
    try:
        frame_number_int = int(frame_number)
    except (TypeError, ValueError, OverflowError):
        frame_number_int = 0
    return min(max(frame_number_int, 0), get_video_frame_last_index(frame_total))


def get_video_frame(video_path: str, frame_number: int = 0) -> Any:
    capture = cv2.VideoCapture(video_path)

    # Set MJPEG format to ensure correct color space handling
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # Only force RGB conversion if color correction is enabled
    if modules.globals.color_correction:
        capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    
    frame_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.set(cv2.CAP_PROP_POS_FRAMES, clamp_video_frame_number(frame_number, frame_total))
    has_frame, frame = capture.read()

    if has_frame and modules.globals.color_correction:
        # Convert the frame color if necessary
        frame = gpu_cvt_color(frame, cv2.COLOR_BGR2RGB)

    capture.release()
    return frame if has_frame else None


def get_video_frame_total(video_path: str) -> int:
    capture = cv2.VideoCapture(video_path)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return video_frame_total
