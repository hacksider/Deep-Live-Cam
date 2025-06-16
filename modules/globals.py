import os
from typing import List, Dict, Any, Optional

ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR: str = os.path.join(ROOT_DIR, "workflow")

file_types: List[Any] = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

source_target_map: List[Dict[str, Any]] = []  # List of face mapping dicts
simple_map: Dict[str, Any] = {}  # Simplified face/embedding map

source_path: Optional[str] = None  # Path to source image
target_path: Optional[str] = None  # Path to target image or video
output_path: Optional[str] = None  # Path to output file or directory
frame_processors: List[str] = []  # List of enabled frame processors
keep_fps: bool = True  # Keep original FPS
keep_audio: bool = True  # Keep original audio
keep_frames: bool = False  # Keep temporary frames
many_faces: bool = False  # Process every face
map_faces: bool = False  # Map source/target faces
color_correction: bool = False  # Toggle for color correction
nsfw_filter: bool = False  # Toggle for NSFW filtering
video_encoder: Optional[str] = None  # Video encoder
video_quality: Optional[int] = None  # Video quality
live_mirror: bool = False  # Mirror webcam preview
live_resizable: bool = True  # Allow resizing webcam preview
max_memory: Optional[int] = None  # Max memory usage
execution_providers: List[str] = []  # ONNX/Torch execution providers
execution_threads: Optional[int] = None  # Number of threads
headless: Optional[bool] = None  # Headless mode
log_level: str = "error"  # Logging level
fp_ui: Dict[str, bool] = {"face_enhancer": False}  # UI state for frame processors
camera_input_combobox: Any = None  # Camera input combobox widget
webcam_preview_running: bool = False  # Webcam preview running state
show_fps: bool = False  # Show FPS overlay
mouth_mask: bool = False  # Enable mouth mask
show_mouth_mask_box: bool = False  # Show mouth mask box
mask_feather_ratio: int = 8  # Feather ratio for mask
mask_down_size: float = 0.50  # Downsize ratio for mask
mask_size: int = 1  # Mask size multiplier
