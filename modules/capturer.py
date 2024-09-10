from typing import Any
import cv2
import modules.globals  # Import the globals to check the color correction toggle

def list_available_cameras(max_tested: int = 10):
    """ List all available camera indices. """
    available_cameras = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def get_video_frame(video_source: Any, frame_number: int = 0, is_camera: bool = False) -> Any:
    """
    Capture a video frame from a camera or video file.
    
    :param video_source: The camera index or video file path.
    :param frame_number: Frame number to retrieve (only applicable for video files).
    :param is_camera: Flag to indicate if the source is a camera.
    :return: The captured frame.
    """
    capture = cv2.VideoCapture(video_source)

    # Set MJPEG format to ensure correct color space handling
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Only force RGB conversion if color correction is enabled
    if modules.globals.color_correction:
        capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    if not is_camera:
        frame_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))

    has_frame, frame = capture.read()

    if has_frame and modules.globals.color_correction:
        # Convert the frame color if necessary
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    capture.release()
    return frame if has_frame else None

def get_video_frame_total(video_path: str) -> int:
    """ Get total frame count of a video file. """
    capture = cv2.VideoCapture(video_path)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return video_frame_total
