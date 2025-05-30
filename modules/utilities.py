import glob
import mimetypes
import os
import platform
import ssl
import subprocess
import cv2
import modules
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
    """Run an ffmpeg command with the given arguments."""
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
        subprocess.run(commands, check=True)
        return True
    except Exception as e:
        print(f"Error running ffmpeg: {e}")
        return False


def detect_fps(target_path: str) -> float:
    """Detect the FPS of a video file using ffprobe."""
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
    try:
        output = subprocess.check_output(command).decode().strip().split("/")
        if len(output) == 2:
            return float(output[0]) / float(output[1])
        return float(output[0])
    except Exception as e:
        print(f"Error detecting FPS: {e}")
        return 30.0


def extract_frames(target_path: str) -> None:
    """Extract frames from a video file to a temp directory."""
    temp_directory_path = get_temp_directory_path(target_path)
    run_ffmpeg(
        [
            "-i",
            target_path,
            "-pix_fmt",
            "rgb24",
            os.path.join(temp_directory_path, "%04d.png"),
        ]
    )


def create_video(target_path: str, fps: float = 30.0) -> None:
    """Create a video from frames in the temp directory."""
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    run_ffmpeg(
        [
            "-r",
            str(fps),
            "-i",
            os.path.join(temp_directory_path, "%04d.png"),
            "-c:v",
            modules.globals.video_encoder,
            "-crf",
            str(modules.globals.video_quality),
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "colorspace=bt709:iall=bt601-6-625:fast=1",
            "-y",
            temp_output_path,
        ]
    )


def restore_audio(target_path: str, output_path: str) -> None:
    """Restore audio from the original video to the output video."""
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg(
        [
            "-i",
            temp_output_path,
            "-i",
            target_path,
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-y",
            output_path,
        ]
    )
    if not done:
        print(f"Failed to restore audio for {output_path}")


def get_temp_frame_paths(target_path: str) -> List[str]:
    """Get all temp frame file paths for a given target path."""
    temp_directory_path = get_temp_directory_path(target_path)
    try:
        return sorted([
            str(p) for p in Path(temp_directory_path).glob("*.png")
        ])
    except Exception as e:
        print(f"Error getting temp frame paths: {e}")
        return []


def get_temp_directory_path(target_path: str) -> str:
    """Get the temp directory path for a given target path."""
    base = os.path.splitext(os.path.basename(target_path))[0]
    temp_dir = os.path.join(TEMP_DIRECTORY, base)
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def get_temp_output_path(target_path: str) -> str:
    """Get the temp output video path for a given target path."""
    base = os.path.splitext(os.path.basename(target_path))[0]
    return os.path.join(TEMP_DIRECTORY, f"{base}_out.mp4")


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    """Normalize the output path for saving results."""
    if not output_path:
        base = os.path.splitext(os.path.basename(target_path))[0]
        return os.path.join(TEMP_DIRECTORY, f"{base}_result.png")
    return output_path


def create_temp(target_path: str) -> None:
    """Create a temp directory for a given target path."""
    temp_directory_path = get_temp_directory_path(target_path)
    os.makedirs(temp_directory_path, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    """Move temp output to the final output path."""
    temp_output_path = get_temp_output_path(target_path)
    try:
        os.rename(temp_output_path, output_path)
    except Exception as e:
        print(f"Error moving temp output: {e}")


def clean_temp(target_path: str) -> None:
    """Remove temp directory and files for a given target path."""
    temp_directory_path = get_temp_directory_path(target_path)
    try:
        for p in Path(temp_directory_path).glob("*"):
            p.unlink()
        os.rmdir(temp_directory_path)
    except Exception as e:
        print(f"Error cleaning temp directory: {e}")


def has_image_extension(image_path: str) -> bool:
    """Check if a file has an image extension."""
    return os.path.splitext(image_path)[1].lower() in [
        ".png", ".jpg", ".jpeg", ".gif", ".bmp"
    ]


def is_image(image_path: str) -> bool:
    """Check if a file is an image."""
    return has_image_extension(image_path)


def is_video(video_path: str) -> bool:
    """Check if a file is a video."""
    return os.path.splitext(video_path)[1].lower() in [
        ".mp4", ".mkv"
    ]


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    """Download files from URLs if they do not exist in the directory."""
    import requests
    for url in urls:
        filename = os.path.basename(url)
        file_path = os.path.join(download_directory_path, filename)
        if not os.path.exists(file_path):
            try:
                print(f"Downloading {url}...")
                r = requests.get(url, stream=True)
                with open(file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")


def resolve_relative_path(path: str) -> str:
    """Resolve a relative path to an absolute path."""
    return os.path.abspath(path)
