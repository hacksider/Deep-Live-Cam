# --- START OF FILE globals.py ---

import os
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

# Face Mapping Data
source_target_map: List[
    Dict[str, Any]
] = []  # Stores detailed map for image/video processing
simple_map: Dict[
    str, Any
] = {}  # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = False
keep_frames: bool = False
many_faces: bool = False  # Process all detected faces with default source
map_faces: bool = False  # Use source_target_map or simple_map for specific swaps
poisson_blend: bool = False  # Enable Poisson Blending for smoother face swaps
color_correction: bool = True  # Enable color correction (implementation specific)
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None  # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None  # Placeholder for UI element if needed
webcam_preview_running: bool = False
live_mode: bool = False
show_fps: bool = False
face_analyser_engine: str = "insightface"  # "insightface" | "mlx_uniface"
mlx_face_detector: str = "retinaface"  # "retinaface"

# System Configuration
max_memory: int | None = None  # Memory limit in GB? (Needs clarification)
execution_providers: List[
    str
] = []  # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None  # Number of threads for CPU execution
headless: bool | None = None  # Run without UI?
log_level: str = "error"  # Logging level (e.g., 'debug', 'info', 'warning', 'error')

# Face Processor UI Toggles (Example)
fp_ui: Dict[str, bool] = {
    "face_enhancer": False,
    "face_enhancer_gpen256": False,
    "face_enhancer_gpen512": False,
}

# Face Swapper Specific Options
face_swapper_enabled: bool = True
opacity: float = 0.9
sharpness: float = 0.0
color_correction: bool = False

# Mouth Mask Options
mouth_mask: bool = False  # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
mask_feather_ratio: int = (
    12  # Denominator for feathering calculation (higher = smaller feather)
)
mask_down_size: float = 0.1  # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0  # Expansion factor for upper lip mask (relative)

# --- START: Added for Frame Interpolation ---
enable_interpolation: bool = True  # Toggle temporal smoothing
interpolation_weight: float = (
    0  # Blend weight for current frame (0.0-1.0). Lower=smoother.
)
# --- END: Added for Frame Interpolation ---

# --- END OF FILE globals.py ---
