import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

souce_target_map = []
simple_map = {}

source_path = None
target_path = None
output_path = None
frame_processors: List[str] = []
keep_fps = True  # Initialize with default value
keep_audio = True  # Initialize with default value
keep_frames = False  # Initialize with default value
many_faces = False  # Initialize with default value
map_faces = False  # Initialize with default value
color_correction = False  # Initialize with default value
nsfw_filter = False  # Initialize with default value
video_encoder = None
video_quality = None
live_mirror = False  # Initialize with default value
live_resizable = False  # Initialize with default value
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = "error"
fp_ui: Dict[str, bool] = {"face_enhancer": False}  # Initialize with default value
camera_input_combobox = None
webcam_preview_running = False
show_fps = False  # Initialize with default value
mouth_mask = False
show_mouth_mask_box = False
mask_down_size = 0.5
mask_size = 1.0
mask_feather_ratio = 8
opacity_switch = False
face_opacity = 100
selected_camera = None
