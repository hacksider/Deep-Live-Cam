import sys
import importlib
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

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

def load_frame_processor_module(frame_processor: str) -> ModuleType:
    try:
        frame_processor_module = importlib.import_module(f'modules.processors.frame.{frame_processor}')
        # Ensure all required methods are present
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                raise AttributeError(f"Missing required method {method_name} in {frame_processor} module.")
    except ImportError:
        print(f"Error: Frame processor '{frame_processor}' not found.")
        sys.exit(1)
    except AttributeError as e:
        print(e)
        sys.exit(1)
    
    return frame_processor_module

def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        FRAME_PROCESSORS_MODULES = [load_frame_processor_module(fp) for fp in frame_processors]
    
    set_frame_processors_modules_from_ui(frame_processors)
    return FRAME_PROCESSORS_MODULES

def set_frame_processors_modules_from_ui(frame_processors: List[str]) -> None:
    global FRAME_PROCESSORS_MODULES
    for frame_processor, state in modules.globals.fp_ui.items():
        if state and frame_processor not in frame_processors:
            module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(module)
            modules.globals.frame_processors.append(frame_processor)
        elif not state and frame_processor in frame_processors:
            module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.remove(module)
            modules.globals.frame_processors.remove(frame_processor)

def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], progress: Any = None) -> None:
    with ThreadPoolExecutor(max_workers=modules.globals.execution_threads) as executor:
        futures = [executor.submit(process_frames, source_path, [path], progress) for path in temp_frame_paths]
        for future in futures:
            future.result()

def process_video(source_path: str, frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        progress.set_postfix({
            'execution_providers': modules.globals.execution_providers, 
            'execution_threads': modules.globals.execution_threads, 
            'max_memory': modules.globals.max_memory
        })
        multi_process_frame(source_path, frame_paths, process_frames, progress)
