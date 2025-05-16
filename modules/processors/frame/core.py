import sys
import importlib
from typing import Set, Optional

# Define a whitelist of allowed modules that can be dynamically imported
ALLOWED_MODULES: Set[str] = {
    'modules.processors.frame.blur',
    'modules.processors.frame.censor',
    'modules.processors.frame.crop',
    'modules.processors.frame.resize',
    'modules.processors.frame.rotate',
    # Add all legitimate processor modules that should be importable
}

def safe_import_module(module_name: str) -> Optional[object]:
    """
    Safely import a module by checking against a whitelist.
    
    Args:
        module_name: The name of the module to import
        
    Returns:
        The imported module or None if the module is not in the whitelist
    """
    if module_name in ALLOWED_MODULES:
        return safe_import_module(module_name)
    else:
        # Log the attempt to import a non-whitelisted module
        print(f"Warning: Attempted to import non-whitelisted module: {module_name}")
        return None

import importlib
from typing import Set, Optional

# Define a whitelist of allowed modules that can be dynamically imported
ALLOWED_MODULES: Set[str] = {
    'modules.processors.frame.blur',
    'modules.processors.frame.censor',
    'modules.processors.frame.crop',
    'modules.processors.frame.resize',
    'modules.processors.frame.rotate',
    # Add all legitimate processor modules that should be importable
}

def safe_import_module(module_name: str) -> Optional[object]:
    """
    Safely import a module by checking against a whitelist.
    
    Args:
        module_name: The name of the module to import
        
    Returns:
        The imported module or None if the module is not in the whitelist
    """
    if module_name in ALLOWED_MODULES:
        return safe_import_module(module_name)
    else:
        # Log the attempt to import a non-whitelisted module
        print(f"Warning: Attempted to import non-whitelisted module: {module_name}")
        return None

from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import importlib
from typing import Set, Optional

# Define a whitelist of allowed modules that can be dynamically imported
ALLOWED_MODULES: Set[str] = {
    'modules.processors.frame.blur',
    'modules.processors.frame.censor',
    'modules.processors.frame.crop',
    'modules.processors.frame.resize',
    'modules.processors.frame.rotate',
    # Add all legitimate processor modules that should be importable
}

def safe_import_module(module_name: str) -> Optional[object]:
    """
    Safely import a module by checking against a whitelist.
    
    Args:
        module_name: The name of the module to import
        
    Returns:
        The imported module or None if the module is not in the whitelist
    """
    if module_name in ALLOWED_MODULES:
        return safe_import_module(module_name)
    else:
        # Log the attempt to import a non-whitelisted module
        print(f"Warning: Attempted to import non-whitelisted module: {module_name}")
        return None

import importlib
from typing import Set, Optional

# Define a whitelist of allowed modules that can be dynamically imported
ALLOWED_MODULES: Set[str] = {
    'modules.processors.frame.blur',
    'modules.processors.frame.censor',
    'modules.processors.frame.crop',
    'modules.processors.frame.resize',
    'modules.processors.frame.rotate',
    # Add all legitimate processor modules that should be importable
}

def safe_import_module(module_name: str) -> Optional[object]:
    """
    Safely import a module by checking against a whitelist.
    
    Args:
        module_name: The name of the module to import
        
    Returns:
        The imported module or None if the module is not in the whitelist
    """
    if module_name in ALLOWED_MODULES:
        return safe_import_module(module_name)
    else:
        # Log the attempt to import a non-whitelisted module
        print(f"Warning: Attempted to import non-whitelisted module: {module_name}")
        return None


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
        frame_processor_module = safe_import_module(f'modules.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                sys.exit()
    except ImportError:
        print(f"Frame processor {frame_processor} not found")
        sys.exit()
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
    with ThreadPoolExecutor(max_workers=modules.globals.execution_threads) as executor:
        futures = []
        for path in temp_frame_paths:
            future = executor.submit(process_frames, source_path, [path], progress)
            futures.append(future)
        for future in futures:
            future.result()


def process_video(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        progress.set_postfix({'execution_providers': modules.globals.execution_providers, 'execution_threads': modules.globals.execution_threads, 'max_memory': modules.globals.max_memory})
        multi_process_frame(source_path, frame_paths, process_frames, progress)
