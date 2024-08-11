from typing import Any, Optional
import cv2

def get_video_frame(video_path: str, frame_number: int = 0) -> Optional[Any]:
    """Retrieve a specific frame from a video."""
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None

    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ensure frame_number is within the valid range
    frame_number = max(0, min(frame_number, frame_total - 1))

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    capture.release()

    if not has_frame:
        print(f"Error: Cannot read frame {frame_number} from {video_path}")
        return None
    
    return frame

def get_video_frame_total(video_path: str) -> int:
    """Get the total number of frames in a video."""
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return 0
    
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    return frame_total
