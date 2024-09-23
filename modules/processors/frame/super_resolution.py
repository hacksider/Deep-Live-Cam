import threading
import traceback
from typing import Any, List
import cv2

import os

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.utilities import conditional_download, resolve_relative_path, is_image, is_video
import numpy as np

NAME = 'DLC.SUPER-RESOLUTION'
THREAD_SEMAPHORE = threading.Semaphore()

# Singleton class for Super-Resolution
class SuperResolutionModel:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, sr_model_path: str = 'ESPCN_x4.pb'):
        if SuperResolutionModel._instance is not None:
            raise Exception("This class is a singleton!")
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self.model_path = os.path.join(resolve_relative_path('../models'), sr_model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Super-resolution model not found at {self.model_path}")
        try:
            self.sr.readModel(self.model_path)
            self.sr.setModel("espcn", 4)  # Using ESPCN with 4x upscaling
        except Exception as e:
            print(f"Error during super-resolution model initialization: {e}")
            raise e

    @classmethod
    def get_instance(cls, sr_model_path: str = 'ESPCN_x4.pb'):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    try:
                        cls._instance = cls(sr_model_path)
                    except Exception as e:
                        print(f"Failed to initialize SuperResolutionModel: {e}")
                        return None
        return cls._instance


def pre_check() -> bool:
    """
    Checks and downloads necessary models before starting the face swapper.
    """
    download_directory_path = resolve_relative_path('../models')
    # Download the super-resolution model as well
    conditional_download(download_directory_path, [
        'https://huggingface.co/spaces/PabloGabrielSch/AI_Resolution_Upscaler_And_Resizer/resolve/bcd13b766a9499196e8becbe453c4a848673b3b6/models/ESPCN_x4.pb'
    ])
    return True

def pre_start() -> bool:
    if not is_image(modules.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(modules.globals.source_path)):
        update_status('No face detected in the source path.', NAME)
        return False
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def apply_super_resolution(image: np.ndarray) -> np.ndarray:
    """
    Applies super-resolution to the given image using the provided super-resolver.

    Args:
        image (np.ndarray): The input image to enhance.
        sr_model_path (str): ESPCN model path for super-resolution.

    Returns:
        np.ndarray: The super-resolved image.
    """
    with THREAD_SEMAPHORE:
        sr_model = SuperResolutionModel.get_instance()

        if sr_model is None:
            print("Super-resolution model is not initialized.")
            return image
        try:
            upscaled_image = sr_model.sr.upsample(image)
            return upscaled_image
        except Exception as e:
            print(f"Error during super-resolution: {e}")
            return image


def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Processes a single frame by swapping the source face into detected target faces.

    Args:

        frame (np.ndarray): The target frame image.

    Returns:
        np.ndarray: The processed frame with swapped faces.
    """

    # Apply super-resolution to the entire frame
    frame = apply_super_resolution(frame)

    return frame

def process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None:
    """
    Processes multiple frames by swapping the source face into each target frame.

    Args:
        source_path (str): Path to the source image.
        temp_frame_paths (List[str]): List of paths to target frame images.
        progress (Any, optional): Progress tracker. Defaults to None.
    """
    for idx, temp_frame_path in enumerate(temp_frame_paths):
        frame = cv2.imread(temp_frame_path)
        if frame is None:
            print(f"Failed to load frame from {temp_frame_path}")
            continue
        try:
            result = process_frame(frame)
            cv2.imwrite(temp_frame_path, result)
        except Exception as exception:
            traceback.print_exc()
            print(f"Error processing frame {temp_frame_path}: {exception}")
        if progress:
            progress.update(1)

def upscale_image(image: np.ndarray, scaling_factor: int = 2) -> np.ndarray:
    """
    Upscales the given image by the specified scaling factor.

    Args:
        image (np.ndarray): The input image to upscale.
        scaling_factor (int): The factor by which to upscale the image.

    Returns:
        np.ndarray: The upscaled image.
    """
    height, width = image.shape[:2]
    new_size = (width * scaling_factor, height * scaling_factor)
    upscaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    return upscaled_image

def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """
    Processes a single image by swapping the source face into the target image.

    Args:
        source_path (str): Path to the source image.
        target_path (str): Path to the target image.
        output_path (str): Path to save the output image.
    """
    source_image = cv2.imread(source_path)
    if source_image is None:
        print(f"Failed to load source image from {source_path}")
        return

    # Upscale the source image for better quality before face detection
    source_image_upscaled = upscale_image(source_image, scaling_factor=2)

    # Detect source face from the upscaled image
    source_face = get_one_face(source_image_upscaled)
    if source_face is None:
        print("No source face detected.")
        return

    target_frame = cv2.imread(target_path)
    if target_frame is None:
        print(f"Failed to load target image from {target_path}")
        return

    # Process the frame
    result = process_frame(target_frame)

    # Save the processed frame
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """
    Processes a video by swapping the source face into each frame.

    Args:
        source_path (str): Path to the source image.
        temp_frame_paths (List[str]): List of paths to video frame images.
    """
    modules.processors.frame.core.process_video(None, temp_frame_paths, process_frames)