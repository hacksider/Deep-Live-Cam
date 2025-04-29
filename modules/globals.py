import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Core paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

# Default configuration settings
DEFAULT_SETTINGS = {
    'max_cluster_k': 10,
    'kmeans_init': 'k-means++',
    'nsfw_threshold': 0.85,
    'mask_feather_ratio': 8,
    'mask_down_size': 0.50,
    'mask_size': 1
}

# File type definitions
file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

# Runtime variables
source_target_map = []
simple_map = {}

# Paths and processing options
source_path = None
target_path = None
output_path = None
frame_processors: List[str] = []
keep_fps = True
keep_audio = True
keep_frames = False
many_faces = False
map_faces = False
color_correction = False
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

# Masking parameters - moved from hardcoded to configurable
mask_feather_ratio = DEFAULT_SETTINGS['mask_feather_ratio']
mask_down_size = DEFAULT_SETTINGS['mask_down_size']
mask_size = DEFAULT_SETTINGS['mask_size']

# Advanced parameters
max_cluster_k = DEFAULT_SETTINGS['max_cluster_k']
kmeans_init = DEFAULT_SETTINGS['kmeans_init']
nsfw_threshold = DEFAULT_SETTINGS['nsfw_threshold']

def load_settings() -> None:
    """
    Load user settings from config file
    """
    global mask_feather_ratio, mask_down_size, mask_size
    global max_cluster_k, kmeans_init, nsfw_threshold
    
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
                
            # Apply settings from config, falling back to defaults
            mask_feather_ratio = config.get('mask_feather_ratio', DEFAULT_SETTINGS['mask_feather_ratio'])
            mask_down_size = config.get('mask_down_size', DEFAULT_SETTINGS['mask_down_size'])
            mask_size = config.get('mask_size', DEFAULT_SETTINGS['mask_size'])
            max_cluster_k = config.get('max_cluster_k', DEFAULT_SETTINGS['max_cluster_k'])
            kmeans_init = config.get('kmeans_init', DEFAULT_SETTINGS['kmeans_init'])
            nsfw_threshold = config.get('nsfw_threshold', DEFAULT_SETTINGS['nsfw_threshold'])
            
            logger.info("Settings loaded from config file")
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")
        # Use defaults if loading fails

def save_settings() -> None:
    """
    Save current settings to config file
    """
    try:
        config = {
            'mask_feather_ratio': mask_feather_ratio,
            'mask_down_size': mask_down_size,
            'mask_size': mask_size,
            'max_cluster_k': max_cluster_k,
            'kmeans_init': kmeans_init,
            'nsfw_threshold': nsfw_threshold
        }
        
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info("Settings saved to config file")
    except Exception as e:
        logger.error(f"Error saving settings: {str(e)}")

# Load settings at module import time
load_settings()