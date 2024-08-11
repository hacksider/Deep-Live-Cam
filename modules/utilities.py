import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Any
from tqdm import tqdm

import modules.globals

TEMP_FILE = 'temp.mp4'
TEMP_DIRECTORY = 'temp'

# Monkey patch SSL for macOS to handle issues with some HTTPS requests
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context

def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-hwaccel', 'auto', '-loglevel', modules.globals.log_level]
    commands.extend(args)
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.output.decode()}")
    return False

def detect_fps(target_path: str) -> float:
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', 
        '-show_entries', 'stream=r_frame_rate', 
        '-of', 'default=noprint_wrappers=1:nokey=1', target_path
    ]
    try:
        output = subprocess.check_output(command).decode().strip().split('/')
        numerator, denominator = map(int, output)
        return numerator / denominator
    except (subprocess.CalledProcessError, ValueError):
        print("Failed to detect FPS, defaulting to 30.0 FPS.")
    return 30.0

def extract_frames(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    create_temp(target_path)
    run_ffmpeg(['-i', target_path, '-pix_fmt', 'rgb24', os.path.join(temp_directory_path, '%04d.png')])

def create_video(target_path: str, fps: float = 30.0) -> None:
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    run_ffmpeg([
        '-r', str(fps), '-i', os.path.join(temp_directory_path, '%04d.png'), 
        '-c:v', modules.globals.video_encoder, 
        '-crf', str(modules.globals.video_quality), 
        '-pix_fmt', 'yuv420p', 
        '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', 
        '-y', temp_output_path
    ])

def restore_audio(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    done = run_ffmpeg([
        '-i', temp_output_path, '-i', target_path, 
        '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y', output_path
    ])
    if not done:
        move_temp(target_path, output_path)

def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob(os.path.join(glob.escape(temp_directory_path), '*.png'))

def get_temp_directory_path(target_path: str) -> str:
    target_name = Path(target_path).stem
    target_directory_path = Path(target_path).parent
    return str(target_directory_path / TEMP_DIRECTORY / target_name)

def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return str(Path(temp_directory_path) / TEMP_FILE)

def normalize_output_path(source_path: str, target_path: str, output_path: str) -> str:
    if source_path and target_path and os.path.isdir(output_path):
        source_name = Path(source_path).stem
        target_name = Path(target_path).stem
        target_extension = Path(target_path).suffix
        return str(Path(output_path) / f"{source_name}-{target_name}{target_extension}")
    return output_path

def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)

def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        shutil.move(temp_output_path, output_path)

def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = Path(temp_directory_path).parent
    if not modules.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if parent_directory_path.exists() and not list(parent_directory_path.iterdir()):
        parent_directory_path.rmdir()

def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(('png', 'jpg', 'jpeg'))

def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return mimetype and mimetype.startswith('image/')
    return False

def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return mimetype and mimetype.startswith('video/')
    return False

def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    download_directory = Path(download_directory_path)
    download_directory.mkdir(parents=True, exist_ok=True)
    for url in urls:
        download_file_path = download_directory / Path(url).name
        if not download_file_path.exists():
            with urllib.request.urlopen(url) as request:
                total = int(request.headers.get('Content-Length', 0))
                with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                    urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

def resolve_relative_path(path: str) -> str:
    return str(Path(__file__).parent / path)
