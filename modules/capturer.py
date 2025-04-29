from typing import Any, Optional
import cv2
import modules.globals
import logging

logger = logging.getLogger(__name__)

def get_video_frame(video_path: str, frame_number: int = 0) -> Optional[Any]:
    """
    Extract a specific frame from a video file with proper color handling.
    
    Args:
        video_path: Path to the video file
        frame_number: Frame number to extract (defaults to first frame)
        
    Returns:
        Video frame as numpy array or None if frame extraction fails
    """
    try:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        # Set MJPEG format to ensure correct color space handling
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Configure color conversion based on setting
        if modules.globals.color_correction:
            capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        else:
            capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # Explicitly disable if not needed
        
        frame_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))
        has_frame, frame = capture.read()

        # Only convert manually if color_correction is enabled but capture didn't handle it
        if has_frame and modules.globals.color_correction and frame is not None:
            frame_channels = frame.shape[2] if len(frame.shape) == 3 else 1
            if frame_channels == 3:  # Only convert if we have a color image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        capture.release()
        return frame if has_frame else None
    except Exception as e:
        logger.error(f"Error processing video frame: {str(e)}")
        return None


def get_video_frame_total(video_path: str) -> int:
    """
    Get the total number of frames in a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Total number of frames in the video
    """
    try:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            logger.error(f"Failed to open video for frame counting: {video_path}")
            return 0
            
        video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return video_frame_total
    except Exception as e:
        logger.error(f"Error counting video frames: {str(e)}")
        return 0