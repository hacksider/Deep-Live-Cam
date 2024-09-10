from typing import Any
import cv2
import modules.globals  # Import the globals to check the color correction toggle


def get_video_frame(video_path: str, frame_number: int = 0) -> Any:
    capture = cv2.VideoCapture(video_path)

    # Set MJPEG format to ensure correct color space handling
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # Only force RGB conversion if color correction is enabled
    if modules.globals.color_correction:
        capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    
    frame_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))
    has_frame, frame = capture.read()

    if has_frame and modules.globals.color_correction:
        # Convert the frame color if necessary
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    capture.release()
    return frame if has_frame else None


def get_video_frame_total(video_path: str) -> int:
    capture = cv2.VideoCapture(video_path)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return video_frame_total
