import os
import sys
import importlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm
import cv2

import modules
import modules.globals

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_image',
    'process_video'
]


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f'modules.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                raise ImportError(f"Frame processor '{frame_processor}' missing method: {method_name}")
    except ImportError as e:
        raise ImportError(f"Frame processor '{frame_processor}' could not be loaded: {e}") from e
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    set_frame_processors_modules_from_ui(frame_processors)
    return FRAME_PROCESSORS_MODULES

def set_frame_processors_modules_from_ui(frame_processors: List[str]) -> None:
    global FRAME_PROCESSORS_MODULES
    current_processor_names = [proc.__name__.split('.')[-1] for proc in FRAME_PROCESSORS_MODULES]

    for frame_processor, state in modules.globals.fp_ui.items():
        if state == True and frame_processor not in current_processor_names:
            try:
                frame_processor_module = load_frame_processor_module(frame_processor)
                FRAME_PROCESSORS_MODULES.append(frame_processor_module)
                if frame_processor not in modules.globals.frame_processors:
                     modules.globals.frame_processors.append(frame_processor)
            except SystemExit:
                 print(f"Warning: Failed to load frame processor {frame_processor} requested by UI state.")
            except Exception as e:
                 print(f"Warning: Error loading frame processor {frame_processor} requested by UI state: {e}")

        elif state == False and frame_processor in current_processor_names:
            try:
                module_to_remove = next((mod for mod in FRAME_PROCESSORS_MODULES if mod.__name__.endswith(f'.{frame_processor}')), None)
                if module_to_remove:
                    FRAME_PROCESSORS_MODULES.remove(module_to_remove)
                if frame_processor in modules.globals.frame_processors:
                    modules.globals.frame_processors.remove(frame_processor)
            except Exception as e:
                 print(f"Warning: Error removing frame processor {frame_processor}: {e}")

def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], progress: Any = None) -> None:
    """Process video frames in parallel using ProcessPoolExecutor.

    Uses separate processes for video batch mode to bypass the GIL and fully
    utilise multiple CPU cores. Each worker process loads its own ONNX model
    (~2-5s startup overhead per worker), which is amortised over thousands of
    frames in a typical video.

    For live/webcam mode, use multi_process_frame_live() instead which uses
    ThreadPoolExecutor to avoid per-process model loading latency.
    """
    max_workers = modules.globals.execution_threads

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_frames, source_path, [path], None): path
            for path in temp_frame_paths
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing frame {futures[future]}: {e}")
            if progress:
                progress.update(1)


def multi_process_frame_live(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], progress: Any = None) -> None:
    """Process frames in parallel using ThreadPoolExecutor for live mode.

    ThreadPoolExecutor avoids the ~2-5s per-worker model loading overhead of
    ProcessPoolExecutor, which is critical for live/webcam mode where latency
    matters. ONNX Runtime releases the GIL during inference, so threads still
    get reasonable parallelism on the dominant cost (model inference).
    """
    max_workers = modules.globals.execution_threads

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_frames, source_path, [path], progress): path
            for path in temp_frame_paths
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing frame {futures[future]}: {e}")


def process_frames_io(
    temp_frame_paths: List[str],
    process_fn: Callable,
    progress: Any = None,
    jpeg_quality: int = 95,
) -> None:
    """Read/process/write loop shared by frame processors.

    ``process_fn`` receives a single frame (numpy array) and must return
    the processed frame.  Frames that cannot be read are skipped.
    """
    for path in temp_frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            if progress:
                progress.update(1)
            continue
        result = process_fn(frame)
        if result is None:
            result = frame
        cv2.imwrite(path, result, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if progress:
            progress.update(1)


def process_video(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        progress.set_postfix({'execution_providers': modules.globals.execution_providers, 'execution_threads': modules.globals.execution_threads, 'max_memory': modules.globals.max_memory})
        multi_process_frame(source_path, frame_paths, process_frames, progress)
