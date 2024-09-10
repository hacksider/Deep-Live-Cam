import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, 'workflow')

file_types = [
    ('Image', ('*.png','*.jpg','*.jpeg','*.gif','*.bmp')),
    ('Video', ('*.mp4','*.mkv'))
]

souce_target_map = []
simple_map = {}

source_path = None
target_path = None
output_path = None
frame_processors: List[str] = []
keep_fps = None
keep_audio = None
keep_frames = None
many_faces = None
map_faces = None
color_correction = None  # New global variable for color correction toggle
nsfw_filter = None
video_encoder = None
video_quality = None
live_mirror = None
live_resizable = None
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = 'error'
fp_ui: Dict[str, bool] = {}
camera_input_combobox = None
webcam_preview_running = False