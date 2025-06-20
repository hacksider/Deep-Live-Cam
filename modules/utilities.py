import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
from pathlib import Path
from typing import List, Any
from tqdm import tqdm

import modules.globals

TEMP_FILE = "temp.mp4"
TEMP_DIRECTORY = "temp"

# monkey patch ssl for mac
if platform.system().lower() == "darwin":
    ssl._create_default_https_context = ssl._create_unverified_context


def run_ffmpeg(args: List[str]) -> bool:
    commands = [
        "ffmpeg",
        "-hide_banner",
        "-hwaccel",
        "auto",
        "-loglevel",
        modules.globals.log_level,
    ]
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        pass
    return False


def detect_fps(target_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        target_path,
    ]
    output = subprocess.check_output(command).decode().strip().split("/")
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass
    return 30.0


def extract_frames(target_path: str, temp_directory_path: str) -> None: # Added temp_directory_path
    # temp_directory_path = get_temp_directory_path(target_path) # Original
    run_ffmpeg(
        [
            "-i",
            target_path,
            "-pix_fmt",
            "rgb24",
            os.path.join(temp_directory_path, "%04d.png"),
        ]
    )


# Accepts pattern for frames and explicit output path
def create_video(frames_pattern: str, fps: float, output_path: str, video_quality: int, video_encoder: str) -> bool:
    # temp_output_path = get_temp_output_path(target_path) # Original
    # temp_directory_path = get_temp_directory_path(target_path) # Original
    return run_ffmpeg( # Return boolean status
        [
            "-r",
            str(fps),
            "-i",
            frames_pattern, # Use pattern directly e.g. /path/to/temp/frames/%04d.png
            "-c:v",
            video_encoder, # Use passed encoder
            "-crf",
            str(video_quality), # Use passed quality
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "colorspace=bt709:iall=bt601-6-625:fast=1",
            "-y",
            output_path, # Use explicit output path
        ]
    )


# Accepts path to video without audio, path to original video (for audio), and final output path
def restore_audio(video_without_audio_path: str, original_audio_source_path: str, final_output_path: str) -> bool:
    # temp_output_path = get_temp_output_path(target_path) # Original
    # target_path was original_audio_source_path
    # output_path was final_output_path
    return run_ffmpeg( # Return boolean status
        [
            "-i",
            video_without_audio_path, # Video processed by frame processors
            "-i",
            original_audio_source_path, # Original video as audio source
            "-c:v",
            "copy",
            "-c:a", # Specify audio codec, e.g., aac or copy if sure
            "aac", # Or "copy" if the original audio is desired as is and compatible
            "-strict", # May be needed for some AAC versions
            "experimental", # May be needed for some AAC versions
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?", # Use ? to make mapping optional (if audio stream exists)
            "-y",
            final_output_path, # Final output path
        ]
    )
    # If ffmpeg fails to restore audio (e.g. no audio in source),
    # it will return False. The calling function should handle this,
    # for example by moving video_without_audio_path to final_output_path.
    # if not done:
    #     move_temp(target_path, output_path) # This logic will be handled in webapp.py


def get_temp_frame_paths(temp_directory_path: str) -> List[str]: # takes temp_directory_path
    # temp_directory_path = get_temp_directory_path(target_path) # This was incorrect
    return glob.glob((os.path.join(glob.escape(temp_directory_path), "*.png")))


def get_temp_directory_path(base_path: str, subfolder_name: str = None) -> str: # Made more generic
    # target_name, _ = os.path.splitext(os.path.basename(target_path)) # Original
    # target_directory_path = os.path.dirname(target_path) # Original
    # return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name) # Original
    if subfolder_name is None:
        subfolder_name, _ = os.path.splitext(os.path.basename(base_path))

    # Use a consistent top-level temp directory if possible, or one relative to base_path's dir
    # For webapp, a central temp might be better than next to the original file if uploads are far away
    # For now, keeping it relative to base_path's directory.
    base_dir = os.path.dirname(base_path)
    return os.path.join(base_dir, TEMP_DIRECTORY, subfolder_name)


# This function might not be needed if create_video directly uses output_path
# def get_temp_output_path(target_path: str) -> str:
#     temp_directory_path = get_temp_directory_path(target_path)
#     return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(target_path: str, output_dir: str, suffix: str) -> Any: # Changed signature
    # if source_path and target_path: # Original
    #     source_name, _ = os.path.splitext(os.path.basename(source_path)) # Original
    #     target_name, target_extension = os.path.splitext(os.path.basename(target_path)) # Original
    #     if os.path.isdir(output_path): # Original output_path was directory
    #         return os.path.join( # Original
    #             output_path, source_name + "-" + target_name + target_extension # Original
    #         ) # Original
    # return output_path # Original

    if target_path and output_dir:
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        # Suffix can be like "_processed" or "_temp_video"
        # Ensure suffix starts with underscore if not already, or handle it if it's part of the name
        if not suffix.startswith("_") and not suffix == "":
            suffix = "_" + suffix

        return os.path.join(output_dir, target_name + suffix + target_extension)
    return None


def create_temp(temp_directory_path: str) -> None: # Takes full temp_directory_path
    # temp_directory_path = get_temp_directory_path(target_path) # Original
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(temp_file_path: str, output_path: str) -> None: # Takes specific temp_file_path
    # temp_output_path = get_temp_output_path(target_path) # Original
    if os.path.isfile(temp_file_path): # Check temp_file_path directly
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_file_path, output_path)


def clean_temp(temp_directory_path: str) -> None: # Takes full temp_directory_path
    # temp_directory_path = get_temp_directory_path(target_path) # This was incorrect
    if not modules.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)

    # Attempt to clean up parent 'temp' directory if it's empty
    # Be cautious with this part to avoid removing unintended directories
    parent_directory_path = os.path.dirname(temp_directory_path)
    if os.path.basename(parent_directory_path) == TEMP_DIRECTORY: # Check if parent is 'temp'
        if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
            try:
                shutil.rmtree(parent_directory_path) # Remove the 'temp' folder itself if empty
                print(f"Cleaned empty temp parent directory: {parent_directory_path}")
            except OSError as e:
                print(f"Error removing temp parent directory {parent_directory_path}: {e}")
    # The duplicated functions below this point should be removed by this diff if they are identical to these.
    # If they are not, this diff might fail or have unintended consequences.
    # The goal is to have only one definition for each utility function.

# Duplicated functions from here are being removed by ensuring the SEARCH block spans them.
# This SEARCH block starts from the known good `has_image_extension` and goes to the end of the file.
def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(("png", "jpg", "jpeg"))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith("image/"))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith("video/"))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(
            download_directory_path, os.path.basename(url)
        )
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get("Content-Length", 0))
            with tqdm(
                total=total,
                desc="Downloading",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
# End of file, ensuring all duplicated content below the last 'SEARCH' block is removed.
