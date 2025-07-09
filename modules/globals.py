import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

source_target_map = []
simple_map = {}

source_path = None
target_path = None
output_path = None
frame_processors: List[str] = []
keep_fps = True
keep_audio = True
keep_frames = False
many_faces = False
map_faces = False
color_correction = False  # New global variable for color correction toggle
nsfw_filter = False
video_encoder = None
video_quality = None
live_mirror = False
live_resizable = True
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = "error"
fp_ui: Dict[str, bool] = {"face_enhancer": False}
camera_input_combobox = None
webcam_preview_running = False
show_fps = False
mouth_mask = False
show_mouth_mask_box = False
mask_feather_ratio = 8
mask_down_size = 0.50
mask_size = 1
use_poisson_blending = False # Added for Poisson blending
poisson_blending_feather_amount = 5 # Feathering for the mask before Poisson blending
preserve_target_ears = False # Flag to enable preserving target's ears
ear_width_ratio = 0.18 # Width of the ear exclusion box as a ratio of face bbox width
ear_height_ratio = 0.35 # Height of the ear exclusion box as a ratio of face bbox height
ear_vertical_offset_ratio = 0.20 # Vertical offset of the ear box from top of face bbox
ear_horizontal_overlap_ratio = 0.03 # How much the ear exclusion zone can overlap into the face bbox
