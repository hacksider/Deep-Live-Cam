# --- START OF FILE core.py ---

import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv) and ('cuda' in sys.argv or 'rocm' in sys.argv):
    # Apply for CUDA or ROCm if explicitly mentioned
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List, Optional, Dict, Any # Added Dict, Any
import platform
import signal
import shutil
import argparse
import gc # Garbage Collector
import time # For timing performance

# Conditional PyTorch import for memory management
_torch_available = False
_torch_cuda_available = False
try:
    import torch
    _torch_available = True
    if torch.cuda.is_available():
        _torch_cuda_available = True
except ImportError:
    # No warning needed unless CUDA is explicitly selected later
    pass

import onnxruntime
import tensorflow
import cv2 # OpenCV is crucial here
import numpy as np # For frame manipulation

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules, load_frame_processor_module # Added load_frame_processor_module
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path
# Import necessary typing
from modules.typing import Frame

# Configuration for GPU Memory Limit (adjust as needed, e.g., 0.7-0.9)
GPU_MEMORY_LIMIT_FRACTION = 0.8 # Keep as default, user might adjust based on VRAM

# Global to hold active processor instances
FRAME_PROCESSORS_INSTANCES: List[Any] = []

# --- Argument Parsing and Setup (Mostly unchanged, but refined) ---

def parse_args() -> argparse.ArgumentParser: # Return parser for help message on error
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100, width=120)) # Improved formatter
    program.add_argument('-s', '--source', help='Select source image(s) or directory', dest='source_path', nargs='+') # Allow multiple sources
    program.add_argument('-t', '--target', help='Select target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='Select output file or directory', dest='output_path')
    # Frame Processors: Add all available processors to choices dynamically later if possible
    available_processors = [proc.NAME for proc in get_frame_processors_modules([])] # Get names dynamically
    program.add_argument('--frame-processor', help='Pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=available_processors, nargs='+')
    program.add_argument('--keep-fps', help='Keep original video fps', dest='keep_fps', action='store_true')
    program.add_argument('--keep-audio', help='Keep original video audio (requires --keep-fps for sync)', dest='keep_audio', action='store_true', default=True) # Keep True default
    program.add_argument('--keep-frames', help='Keep temporary frames after processing', dest='keep_frames', action='store_true')
    program.add_argument('--many-faces', help='Process all detected faces (specific processor behavior)', dest='many_faces', action='store_true')
    program.add_argument('--nsfw-filter', help='Enable NSFW prediction and skip if detected', dest='nsfw_filter', action='store_true')
    program.add_argument('--map-faces', help='Enable face mapping for video (requires target analysis)', dest='map_faces', action='store_true')
    program.add_argument('--color-correction', help='Enable color correction (specific processor behavior)', dest='color_correction', action='store_true') # Add color correction flag
    # Mouth mask is processor specific, maybe handled internally or via processor options? Keep it for now.
    program.add_argument('--mouth-mask', help='Enable mouth masking (specific processor behavior)', dest='mouth_mask', action='store_true')
    program.add_argument('--video-encoder', help='Output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc']) # Added NVIDIA HW encoders
    program.add_argument('--video-quality', help='Output video quality crf/qp (0-51 for sw, 0-? for hw, lower=better)', dest='video_quality', type=int, default=18) # Adjusted help text
    program.add_argument('-l', '--lang', help='UI language', default="en", choices=["en", "de", "es", "fr", "it", "pt", "ru", "zh"]) # Example languages
    program.add_argument('--live-mirror', help='Mirror live camera feed', dest='live_mirror', action='store_true')
    program.add_argument('--live-resizable', help='Make live camera window resizable', dest='live_resizable', action='store_true')
    program.add_argument('--max-memory', help='DEPRECATED: Use GPU memory fraction. Max CPU RAM limit (GB).', dest='max_memory', type=int) # Default removed, handled dynamically
    program.add_argument('--execution-provider', help='Execution provider(s) (cpu, cuda, rocm, dml, coreml)', dest='execution_provider', default=suggest_execution_providers(), nargs='+') # Use suggested default
    program.add_argument('--execution-threads', help='Number of threads for execution provider', dest='execution_threads', type=int, default=suggest_execution_threads()) # Use suggested default
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    # Check for ROCm selection early for PyTorch unloading
    _is_rocm_selected = any('rocm' in ep.lower() for ep in args.execution_provider)
    global _torch_available, _torch_cuda_available
    if _is_rocm_selected and _torch_available:
        print("[DLC.CORE] ROCm selected, unloading PyTorch.")
        del torch
        _torch_available = False
        _torch_cuda_available = False
        gc.collect()

    handle_deprecated_args(args) # Handle deprecated args after initial parsing

    # Assign to globals
    # Use the first source if multiple provided for single-source contexts, processors might handle multiple sources.
    modules.globals.source_path = args.source_path[0] if isinstance(args.source_path, list) else args.source_path
    # Store all sources if needed by processors
    modules.globals.source_paths = args.source_path if isinstance(args.source_path, list) else [args.source_path]
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)

    # Frame Processors: Store names, instances will be created later
    modules.globals.frame_processors = args.frame_processor

    modules.globals.headless = bool(args.source_path or args.target_path or args.output_path)
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.mouth_mask = args.mouth_mask # Pass to processors if they use it
    modules.globals.color_correction = args.color_correction # Pass to processors
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    # Set max_memory, use suggested if not provided by user
    modules.globals.max_memory = args.max_memory if args.max_memory is not None else suggest_max_memory()

    # Decode and validate execution providers
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    # Set execution threads, ensure it's positive
    modules.globals.execution_threads = max(1, args.execution_threads)
    modules.globals.lang = args.lang

    # Update derived globals for UI state etc.
    modules.globals.fp_ui['face_enhancer'] = 'face_enhancer' in modules.globals.frame_processors
    modules.globals.fp_ui['face_swapper'] = 'face_swapper' in modules.globals.frame_processors # Example
    # Add other processors as needed

    # Final checks and warnings
    if modules.globals.keep_audio and not modules.globals.keep_fps:
        print("\033[33mWarning: --keep-audio is enabled without --keep-fps. This may cause audio/video sync issues.\033[0m")
    if 'cuda' in modules.globals.execution_providers and not _torch_cuda_available:
         # Warning if CUDA provider selected but PyTorch CUDA not functional (for memory limiting)
         print("\033[33mWarning: CUDA provider selected, but torch.cuda.is_available() is False. PyTorch GPU memory limiting disabled.\033[0m")
    if ('h264_nvenc' in modules.globals.video_encoder or 'hevc_nvenc' in modules.globals.video_encoder) and 'cuda' not in modules.globals.execution_providers:
        # Check if ffmpeg build supports nvenc if needed
        print(f"\033[33mWarning: NVENC encoder ({modules.globals.video_encoder}) selected, but 'cuda' is not in execution providers. Ensure ffmpeg has NVENC support and drivers are installed.\033[0m")

    # Set ONNX Runtime logging level (0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal)
    try:
        onnxruntime.set_default_logger_severity(3) # Set to Error level to reduce verbose logs
    except AttributeError:
        print("\033[33mWarning: Could not set ONNX Runtime logger severity (might be an older version).\033[0m")

    return program # Return parser


def handle_deprecated_args(args: argparse.Namespace) -> None:
    """Handles deprecated arguments and updates corresponding new arguments if necessary."""
    # Source path
    if args.source_path_deprecated:
        print('\033[33mWarning: Argument -f/--face is deprecated. Use -s/--source instead.\033[0m')
        if not args.source_path:
            # Convert to list to match potential nargs='+'
            args.source_path = [args.source_path_deprecated]

    # Execution Threads
    if args.cpu_cores_deprecated is not None:
        print('\033[33mWarning: Argument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        # Only override if execution_threads wasn't explicitly set *and* cpu_cores was used
        if args.execution_threads == suggest_execution_threads(): # Check against default suggestion
             args.execution_threads = args.cpu_cores_deprecated

    if args.gpu_threads_deprecated is not None:
        print('\033[33mWarning: Argument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        # Override if gpu_threads was used, potentially overriding cpu_cores value if both were used
        # Check if execution_threads is still at default OR was set by cpu_cores_deprecated
        if args.execution_threads == suggest_execution_threads() or \
           (args.cpu_cores_deprecated is not None and args.execution_threads == args.cpu_cores_deprecated):
             args.execution_threads = args.gpu_threads_deprecated

    # Execution Provider from gpu_vendor
    if args.gpu_vendor_deprecated:
        # Only override if execution_provider is still the default suggested list
        suggested_providers_default = suggest_execution_providers()
        is_default_provider = sorted(args.execution_provider) == sorted(suggested_providers_default)

        if is_default_provider:
            provider_map = {
                'apple': ['coreml', 'cpu'],
                'nvidia': ['cuda', 'cpu'],
                'amd': ['rocm', 'cpu'],
                'intel': ['dml', 'cpu'] # Example for DirectML on Intel
            }
            vendor = args.gpu_vendor_deprecated.lower()
            if vendor in provider_map:
                print(f'\033[33mWarning: Argument --gpu-vendor {args.gpu_vendor_deprecated} is deprecated. Setting --execution-provider to {provider_map[vendor]}.\033[0m')
                args.execution_provider = provider_map[vendor]
            else:
                 print(f'\033[33mWarning: Unknown --gpu-vendor {args.gpu_vendor_deprecated}. Default execution providers kept.\033[0m')
        else:
             # User explicitly set execution providers, ignore deprecated vendor
             print(f'\033[33mWarning: --gpu-vendor {args.gpu_vendor_deprecated} is deprecated and ignored because --execution-provider was explicitly set to {args.execution_provider}.\033[0m')


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    """Converts ONNX Runtime provider names to lowercase short names."""
    return [ep.replace('ExecutionProvider', '').lower() for ep in execution_providers]


def decode_execution_providers(execution_providers_names: List[str]) -> List[str]:
    """Converts lowercase short names back to full ONNX Runtime provider names, preserving order and ensuring availability."""
    available_providers_full = onnxruntime.get_available_providers() # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
    available_providers_encoded = encode_execution_providers(available_providers_full) # e.g., ['cuda', 'cpu']
    decoded_providers = []
    requested_providers_lower = [name.lower() for name in execution_providers_names]

    # User's requested providers first, if available
    for req_name_lower in requested_providers_lower:
        try:
            idx = available_providers_encoded.index(req_name_lower)
            provider_full_name = available_providers_full[idx]
            if provider_full_name not in decoded_providers: # Avoid duplicates
                 decoded_providers.append(provider_full_name)
        except ValueError:
            print(f"\033[33mWarning: Requested execution provider '{req_name_lower}' is not available or not recognized by ONNX Runtime.\033[0m")

    # Ensure CPU is present if no other providers were valid or if it wasn't requested but is available
    cpu_provider_full = 'CPUExecutionProvider'
    if not decoded_providers or cpu_provider_full not in decoded_providers:
        if cpu_provider_full in available_providers_full:
            if cpu_provider_full not in decoded_providers: # Add CPU if missing
                decoded_providers.append(cpu_provider_full)
            print(f"[DLC.CORE] Ensuring '{cpu_provider_full}' is included as a fallback.")
        else:
             # This is critical - OR needs at least one provider
             print(f"\033[31mFatal Error: No valid execution providers found, and '{cpu_provider_full}' is not available in this ONNX Runtime build!\033[0m")
             sys.exit(1)

    # Filter list based on actual availability reported by ORT (double check)
    final_providers = [p for p in decoded_providers if p in available_providers_full]
    if len(final_providers) != len(decoded_providers):
        removed = set(decoded_providers) - set(final_providers)
        print(f"\033[33mWarning: Providers {list(removed)} were removed after final availability check.\033[0m")

    if not final_providers:
         print(f"\033[31mFatal Error: No available execution providers could be configured. Available: {available_providers_full}\033[0m")
         sys.exit(1)

    print(f"[DLC.CORE] Using execution providers: {final_providers}")
    return final_providers


def suggest_max_memory() -> int:
    """Suggests a default max CPU RAM limit in GB based on available memory (heuristic)."""
    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        # Suggest using roughly 50% of total RAM, capped at a reasonable upper limit (e.g., 64GB)
        # and a lower limit (e.g., 4GB)
        suggested_gb = max(4, min(int(total_memory_gb * 0.5), 64))
        # print(f"[DLC.CORE] Suggested max CPU memory (heuristic): {suggested_gb} GB")
        return suggested_gb
    except ImportError:
        print("\033[33mWarning: 'psutil' module not found. Cannot suggest dynamic max_memory. Using default (16GB).\033[0m")
        # Fallback to a static default if psutil is not available
        return 16
    except Exception as e:
        print(f"\033[33mWarning: Error getting system memory: {e}. Using default max_memory (16GB).\033[0m")
        return 16


def suggest_execution_providers() -> List[str]:
    """Suggests available execution providers as short names, prioritizing GPU if available."""
    available_providers_full = onnxruntime.get_available_providers()
    available_providers_encoded = encode_execution_providers(available_providers_full)

    # Prioritize GPU providers
    provider_priority = ['cuda', 'rocm', 'dml', 'coreml', 'cpu']
    suggested = []
    for provider in provider_priority:
        if provider in available_providers_encoded:
            suggested.append(provider)

    # Ensure CPU is always included as a fallback
    if 'cpu' not in suggested and 'cpu' in available_providers_encoded:
        suggested.append('cpu')

    # If only CPU is available, return that
    if not suggested and 'cpu' in available_providers_encoded:
         return ['cpu']
    elif not suggested:
         # Should not happen if ORT is installed correctly
         print("\033[31mError: No execution providers detected, including CPU!\033[0m")
         return ['cpu'] # Still return cpu as a placeholder

    return suggested


def suggest_execution_threads() -> int:
    """Suggests a sensible default number of execution threads based on logical CPU cores."""
    try:
        logical_cores = os.cpu_count()
        if logical_cores:
            # Heuristic: Use most cores, but leave some for OS/other tasks. Cap reasonably.
            # For systems with many cores (>16), maybe don't use all of them by default.
            threads = max(1, min(logical_cores - 2, 16)) if logical_cores > 4 else max(1, logical_cores - 1)
            return threads
    except NotImplementedError:
        pass # Fallback if os.cpu_count() fails
    except Exception as e:
        print(f"\033[33mWarning: Error getting CPU count: {e}. Using default threads (4).\033[0m")

    # Default fallback
    return 4


def limit_gpu_memory(fraction: float) -> None:
    """Attempts to limit GPU memory usage via PyTorch (for CUDA) or TensorFlow."""
    gpu_limited = False

    # 1. PyTorch (CUDA) Limit - Only if PyTorch CUDA is available
    if 'CUDAExecutionProvider' in modules.globals.execution_providers and _torch_cuda_available:
        try:
            # Ensure fraction is within valid range [0.0, 1.0]
            safe_fraction = max(0.1, min(1.0, fraction)) # Prevent setting 0%
            print(f"[DLC.CORE] Attempting to limit PyTorch CUDA memory fraction to {safe_fraction:.1%}")
            torch.cuda.set_per_process_memory_fraction(safe_fraction, 0) # Limit on default device (0)
            print(f"[DLC.CORE] PyTorch CUDA memory fraction limit set.")
            gpu_limited = True
            # Optional: Check memory post-limit (can be verbose)
            # total_mem = torch.cuda.get_device_properties(0).total_memory
            # reserved_mem = torch.cuda.memory_reserved(0)
            # allocated_mem = torch.cuda.memory_allocated(0)
            # print(f"[DLC.CORE] CUDA Device 0: Total={total_mem/1024**3:.2f}GB, Reserved={reserved_mem/1024**3:.2f}GB, Allocated={allocated_mem/1024**3:.2f}GB")
        except RuntimeError as e:
            print(f"\033[33mWarning: Failed to set PyTorch CUDA memory fraction (may already be initialized?): {e}\033[0m")
        except Exception as e:
            print(f"\033[33mWarning: An unexpected error occurred setting PyTorch CUDA memory fraction: {e}\033[0m")

    # 2. TensorFlow GPU Limit (Memory Growth) - Less direct limit, but essential
    try:
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tensorflow.config.experimental.set_memory_growth(gpu, True)
                    print(f"[DLC.CORE] Enabled TensorFlow memory growth for GPU: {gpu.name}")
                    gpu_limited = True # Considered a form of GPU resource management
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(f"\033[33mWarning: Could not set TensorFlow memory growth for {gpu.name} (may already be initialized?): {e}\033[0m")
                except Exception as e:
                    print(f"\033[33mWarning: An unexpected error occurred setting TensorFlow memory growth for {gpu.name}: {e}\033[0m")
        # else:
            # No TF GPUs detected, which is fine if not using TF-based models directly
            # print("[DLC.CORE] No TensorFlow physical GPUs detected.")
    except Exception as e:
        print(f"\033[33mWarning: Error configuring TensorFlow GPU settings: {e}\033[0m")

    # if not gpu_limited:
    #      print("[DLC.CORE] No GPU memory limits applied (GPU provider not used, or libraries unavailable/failed).")


def limit_resources() -> None:
    """Limits system resources like CPU RAM (best effort) and configures TF."""
    # 1. Limit CPU RAM (Best effort, platform dependent)
    if modules.globals.max_memory and modules.globals.max_memory > 0:
        limit_gb = modules.globals.max_memory
        limit_bytes = limit_gb * (1024 ** 3)
        try:
            if platform.system().lower() in ['linux', 'darwin']:
                import resource
                # RLIMIT_AS limits virtual memory size (includes RAM, swap, mappings)
                # Set both soft and hard limits
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
                print(f"[DLC.CORE] Limited process virtual memory (CPU RAM approximation) to ~{limit_gb} GB.")
            elif platform.system().lower() == 'windows':
                # Windows limiting is harder; SetProcessWorkingSetSizeEx is more of a hint
                # Using Job Objects is the robust way but complex to implement here
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetCurrentProcess()
                # Try setting min and max working set size
                # Note: Requires specific privileges, might fail silently or with error code
                # Use values slightly smaller than the limit for flexibility
                min_ws = 1024 * 1024 # Set a small minimum (e.g., 1MB)
                max_ws = limit_bytes
                if not kernel32.SetProcessWorkingSetSizeEx(handle, ctypes.c_size_t(min_ws), ctypes.c_size_t(max_ws), ctypes.c_ulong(0x1)): # QUOTA_LIMITS_HARDWS_ENABLE = 0x1
                     last_error = ctypes.get_last_error()
                     # Common error: 1314 (ERROR_PRIVILEGE_NOT_HELD)
                     if last_error == 1314:
                         print(f"\033[33mWarning: Failed to set process working set size limit on Windows (Error {last_error}). Try running as Administrator if limits are needed.\033[0m")
                     else:
                         print(f"\033[33mWarning: Failed to set process working set size limit on Windows (Error {last_error}).\033[0m")
                else:
                    print(f"[DLC.CORE] Requested process working set size limit (Windows memory hint) max ~{limit_gb} GB.")
            else:
                 print(f"\033[33mWarning: CPU RAM limiting not implemented for platform {platform.system()}. --max-memory ignored.\033[0m")
        except ImportError:
             print(f"\033[33mWarning: 'resource' module (Linux/macOS) or 'ctypes' (Windows) not available. Cannot limit CPU RAM.\033[0m")
        except Exception as e:
             print(f"\033[33mWarning: Failed to limit CPU RAM: {e}\033[0m")
    # else:
    #      print("[DLC.CORE] CPU RAM limit (--max-memory) not set.")

    # 2. Configure TensorFlow GPU memory growth (already done in limit_gpu_memory, but safe to call again)
    #    This ensures it's attempted even if limit_gpu_memory wasn't fully effective.
    try:
        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    if not tensorflow.config.experimental.get_memory_growth(gpu):
                         tensorflow.config.experimental.set_memory_growth(gpu, True)
                         # print(f"[DLC.CORE] Re-checked TF memory growth for {gpu.name}: Enabled.") # Avoid redundant logs
                except RuntimeError:
                     pass # Ignore if already initialized error
    except Exception:
        pass # Ignore errors here, primary attempt was in limit_gpu_memory


def release_resources() -> None:
    """Releases resources, especially GPU memory caches, and runs garbage collection."""
    # 1. Clear PyTorch CUDA cache (if applicable and available)
    if _torch_cuda_available: # Check if torch+cuda is loaded
        try:
            torch.cuda.empty_cache()
            # print("[DLC.CORE] Cleared PyTorch CUDA cache.") # Can be verbose
        except Exception as e:
             print(f"\033[33mWarning: Failed to clear PyTorch CUDA cache: {e}\033[0m")

    # 2. Potentially clear TensorFlow session / clear Keras backend session (less common need)
    # try:
    #     from tensorflow.keras import backend as K
    #     K.clear_session()
    #     print("[DLC.CORE] Cleared Keras backend session.")
    # except ImportError:
    #     pass # Keras might not be installed or used
    # except Exception as e:
    #     print(f"\033[33mWarning: Failed to clear Keras session: {e}\033[0m")

    # 3. Explicitly run garbage collection (important!)
    gc.collect()
    # print("[DLC.CORE] Ran garbage collection.") # Can be verbose


def pre_check() -> bool:
    """Performs essential pre-run checks for dependencies, versions, and paths."""
    update_status('Performing pre-flight checks...')
    checks_passed = True

    # Python version
    if sys.version_info < (3, 9):
        update_status('Error: Python 3.9 or higher is required.', 'ERROR')
        checks_passed = False

    # FFmpeg
    if not shutil.which('ffmpeg'):
        update_status('Error: ffmpeg command was not found in your system PATH. Please install ffmpeg.', 'ERROR')
        checks_passed = False

    # ONNX Runtime
    try:
        ort_version = onnxruntime.__version__
        update_status(f'ONNX Runtime version: {ort_version}')
    except Exception as e:
         update_status(f'Error: Failed to import or access ONNX Runtime: {e}', 'ERROR')
         checks_passed = False

    # TensorFlow (optional, but good to check)
    try:
        tf_version = tensorflow.__version__
        update_status(f'TensorFlow version: {tf_version}')
    except Exception as e:
        update_status(f'Warning: Could not import or access TensorFlow: {e}', 'WARN')
        # Decide if TF absence is critical based on potential processors
        # checks_passed = False

    # PyTorch (only if CUDA is selected for memory limiting)
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        if not _torch_available:
            update_status('Warning: CUDA provider selected, but PyTorch is not installed. GPU memory limiting via PyTorch is disabled.', 'WARN')
        elif not _torch_cuda_available:
            update_status('Warning: PyTorch installed, but torch.cuda.is_available() is False. Check PyTorch CUDA installation and drivers. GPU memory limiting via PyTorch is disabled.', 'WARN')
        else:
             update_status(f'PyTorch version: {torch.__version__} (CUDA available for memory limiting)')


    # Check source/target paths if in headless mode
    if modules.globals.headless:
        if not modules.globals.source_path:
            update_status("Error: Source path ('-s' or '--source') is required in headless mode.", 'ERROR')
            checks_passed = False
        # Check if source files exist
        elif isinstance(modules.globals.source_paths, list):
             for spath in modules.globals.source_paths:
                 if not os.path.exists(spath):
                      update_status(f"Error: Source file/directory not found: {spath}", 'ERROR')
                      checks_passed = False
        elif not os.path.exists(modules.globals.source_path):
            update_status(f"Error: Source file/directory not found: {modules.globals.source_path}", 'ERROR')
            checks_passed = False

        if not modules.globals.target_path:
            update_status("Error: Target path ('-t' or '--target') is required in headless mode.", 'ERROR')
            checks_passed = False
        elif not os.path.exists(modules.globals.target_path):
            update_status(f"Error: Target file not found: {modules.globals.target_path}", 'ERROR')
            checks_passed = False

        if not modules.globals.output_path:
             update_status("Error: Output path ('-o' or '--output') could not be determined or is missing.", 'ERROR')
             checks_passed = False

    update_status('Pre-flight checks completed.')
    return checks_passed


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    """Prints status messages and updates UI if not headless."""
    log_message = f'[{scope}] {message}'
    print(log_message)
    if not modules.globals.headless:
        try:
            # Check if ui module and function exist and are callable
            if hasattr(ui, 'update_status') and callable(ui.update_status):
                ui.update_status(message) # Pass original message to UI
        except Exception as e:
            print(f"[DLC.CORE] Error updating UI status: {e}")


# --- Main Processing Logic ---

def start() -> None:
    """Main processing logic for images and videos."""
    start_time = time.time()
    update_status(f'Processing started at {time.strftime("%Y-%m-%d %H:%M:%S")}')

    # --- Load and Prepare Frame Processors ---
    global FRAME_PROCESSORS_INSTANCES
    FRAME_PROCESSORS_INSTANCES = [] # Clear previous instances if any
    processors_ready = True
    for processor_name in modules.globals.frame_processors:
        update_status(f'Loading frame processor: {processor_name}...')
        module = load_frame_processor_module(processor_name)
        if module:
            # Pass necessary global options to the processor's constructor or setup method if needed
            # Example: instance = module.Processor(many_faces=modules.globals.many_faces, ...)
            instance = module # Assuming module itself might have necessary functions
            FRAME_PROCESSORS_INSTANCES.append(instance)
            if not instance.pre_start(): # Call pre_start after loading
                 update_status(f'Initialization failed for {processor_name}. Aborting.', 'ERROR')
                 processors_ready = False
                 break # Stop loading further processors
        else:
            update_status(f'Could not load frame processor module: {processor_name}. Aborting.', 'ERROR')
            processors_ready = False
            break

    if not processors_ready or not FRAME_PROCESSORS_INSTANCES:
        update_status('Frame processor setup failed. Cannot start processing.', 'ERROR')
        return

    # Simplify face map for faster lookups if needed
    if modules.globals.map_faces and ('face_swapper' in modules.globals.frame_processors): # Example condition
        update_status("Simplifying face map for processing...", "Face Analyser")
        from modules.face_analyser import simplify_maps # Import locally
        simplify_maps()
        # Verify map content after simplification (optional debug)
        # if modules.globals.simple_map:
        #      print(f"[DEBUG] Simple map: {len(modules.globals.simple_map['source_faces'])} sources, {len(modules.globals.simple_map['target_embeddings'])} targets")
        # else:
        #      print("[DEBUG] Simple map is empty.")


    # --- Target is Image ---
    if has_image_extension(modules.globals.target_path) and is_image(modules.globals.target_path):
        process_image_to_image()

    # --- Target is Video ---
    elif is_video(modules.globals.target_path):
        process_video()

    # --- Invalid Target ---
    else:
        if modules.globals.target_path:
            update_status(f"Target path '{modules.globals.target_path}' is not a recognized image or video file.", "ERROR")
        else:
            update_status("Target path not specified or invalid.", "ERROR")

    # --- Processing Finished ---
    end_time = time.time()
    total_time = end_time - start_time
    update_status(f'Processing finished in {total_time:.2f} seconds.')


def process_image_to_image():
    """Handles the image-to-image processing workflow."""
    update_status('Processing image: {}'.format(os.path.basename(modules.globals.target_path)))

    # --- NSFW Check ---
    if modules.globals.nsfw_filter:
        update_status("Checking target image for NSFW content...", "NSFW")
        from modules.predicter import predict_image # Import locally
        try:
            is_nsfw = predict_image(modules.globals.target_path)
            if is_nsfw:
                update_status("NSFW content detected in target image. Skipping processing.", "NSFW")
                if not modules.globals.headless:
                     ui.show_error("NSFW content detected. Processing skipped.", title="NSFW Detected")
                # Consider deleting output placeholder if it exists? Risky.
                # if os.path.exists(modules.globals.output_path): os.remove(modules.globals.output_path)
                return # Stop processing
            else:
                 update_status("NSFW check passed.", "NSFW")
        except Exception as e:
             update_status(f"Error during NSFW check for image: {e}. Continuing processing.", "NSFW")

    # --- Process ---
    try:
        # Create output directory if needed
        output_dir = os.path.dirname(modules.globals.output_path)
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir, exist_ok=True)
             print(f"[DLC.CORE] Created output directory: {output_dir}")

        # Read target image using OpenCV (consistent with video frames)
        target_frame: Frame = cv2.imread(modules.globals.target_path)
        if target_frame is None:
            update_status(f'Error: Could not read target image file: {modules.globals.target_path}', 'ERROR')
            return

        # --- Apply Processors Sequentially ---
        processed_frame = target_frame.copy() # Start with a copy
        for processor in FRAME_PROCESSORS_INSTANCES:
            processor_name = getattr(processor, 'NAME', 'UnknownProcessor') # Get name safely
            update_status(f'Applying {processor_name}...', processor_name)
            try:
                # Processors should accept a frame (numpy array) and return a processed frame
                # Pass global options if needed by the process_frame method
                start_proc_time = time.time()
                # Pass source path(s) and the frame to be processed
                processor_params = {
                     "source_paths": modules.globals.source_paths, # Pass list of source paths
                     "target_frame": processed_frame,
                     "many_faces": modules.globals.many_faces,
                     "color_correction": modules.globals.color_correction,
                     "mouth_mask": modules.globals.mouth_mask,
                     # Add other relevant globals if processors need them
                 }
                # Filter params based on what the processor's process_frame expects (optional advanced)

                processed_frame = processor.process_frame(processor_params)

                if processed_frame is None:
                     update_status(f'Error: Processor {processor_name} returned None. Aborting processing for this image.', 'ERROR')
                     return # Stop processing this image

                end_proc_time = time.time()
                update_status(f'{processor_name} applied in {end_proc_time - start_proc_time:.2f} seconds.', processor_name)
                release_resources() # Release memory after each processor

            except Exception as e:
                update_status(f'Error applying processor {processor_name}: {e}', 'ERROR')
                import traceback
                traceback.print_exc()
                return # Stop processing on error

        # --- Save Processed Image ---
        update_status(f'Saving processed image to: {modules.globals.output_path}')
        try:
            # Use OpenCV to save the final frame
            # Quality parameters can be added for formats like JPG
            # Example: cv2.imwrite(modules.globals.output_path, processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            save_success = cv2.imwrite(modules.globals.output_path, processed_frame)
            if not save_success:
                 update_status('Error: Failed to save the processed image.', 'ERROR')
            elif os.path.exists(modules.globals.output_path) and is_image(modules.globals.output_path):
                 update_status('Image processing finished successfully.')
            else:
                 update_status('Error: Output image file not found or invalid after saving.', 'ERROR')

        except Exception as e:
            update_status(f'Error saving processed image: {e}', 'ERROR')

    except Exception as e:
        update_status(f'An unexpected error occurred during image processing: {e}', 'ERROR')
        import traceback
        traceback.print_exc()


def process_video():
    """Handles the video processing workflow with optimized frame handling."""
    update_status('Processing video: {}'.format(os.path.basename(modules.globals.target_path)))

    # --- NSFW Check (Basic - Check first frame or predict_video) ---
    if modules.globals.nsfw_filter:
        update_status("Checking video for NSFW content (sampling)...", "NSFW")
        from modules.predicter import predict_video # Import locally
        try:
            # Use the library's video prediction (may not use optimal providers)
            # Or implement custom frame sampling here using predict_frame
            is_nsfw = predict_video(modules.globals.target_path)
            if is_nsfw:
                update_status("NSFW content detected in video (based on sampling). Skipping processing.", "NSFW")
                if not modules.globals.headless:
                     ui.show_error("NSFW content detected. Processing skipped.", title="NSFW Detected")
                return # Stop processing
            else:
                 update_status("NSFW check passed (based on sampling).", "NSFW")
        except Exception as e:
             update_status(f"Error during NSFW check for video: {e}. Continuing processing.", "NSFW")

    # --- Prepare Temp Environment ---
    temp_output_video_path = None # For intermediate video file
    video_fps = 30.0 # Default FPS

    try:
        # Setup temp directory and frame extraction (if not mapping faces, which might pre-extract)
        # If map_faces is enabled, face_analyser.get_unique_faces_from_target_video handles extraction.
        if not modules.globals.map_faces:
            update_status('Creating temporary resources...', 'Temp')
            clean_temp(modules.globals.target_path) # Clean first
            create_temp(modules.globals.target_path)
            update_status('Extracting video frames...', 'FFmpeg')
            extract_frames(modules.globals.target_path, modules.globals.keep_fps) # Pass keep_fps hint
            update_status('Frame extraction complete.', 'FFmpeg')
        # else: Handled by face mapper


        # Get paths to frames (must exist either way)
        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
        if not temp_frame_paths:
            update_status('Error: No frames found to process. Check temp folder or extraction step.', 'ERROR')
            destroy(to_quit=False) # Clean up temp
            return

        num_frames = len(temp_frame_paths)
        update_status(f'Processing {num_frames} frames...')

        # Determine Target FPS
        if modules.globals.keep_fps:
            update_status('Detecting target video FPS...', 'FFmpeg')
            detected_fps = detect_fps(modules.globals.target_path)
            if detected_fps:
                video_fps = detected_fps
                update_status(f'Using detected FPS: {video_fps:.2f}')
            else:
                update_status("Warning: Could not detect FPS, using default 30.", "WARN")
                video_fps = 30.0 # Fallback fps
        else:
            video_fps = 30.0 # Use default fps if not keeping original
            update_status(f'Using fixed FPS: {video_fps:.2f}')
        modules.globals.video_fps = video_fps # Store globally if needed elsewhere

        # --- OPTIMIZED Frame Processing Loop ---
        update_status('Starting frame processing loop...')
        # Use tqdm for progress bar
        frame_iterator = tqdm(enumerate(temp_frame_paths), total=num_frames, desc="Processing Frames", unit="frame")

        for frame_index, frame_path in frame_iterator:
            try:
                # 1. Read Frame
                target_frame: Frame = cv2.imread(frame_path)
                if target_frame is None:
                    update_status(f'Warning: Could not read frame {frame_path}. Skipping.', 'WARN')
                    continue

                # Frame dimensions for potential checks later
                # height, width = target_frame.shape[:2]

                # 2. Apply Processors Sequentially to this Frame
                processed_frame = target_frame # Start with the original frame for this iteration
                for processor in FRAME_PROCESSORS_INSTANCES:
                    processor_name = getattr(processor, 'NAME', 'UnknownProcessor')
                    try:
                        # Pass necessary parameters to the processor's process_frame method
                        processor_params = {
                            "source_paths": modules.globals.source_paths,
                            "target_frame": processed_frame, # Pass the current state of the frame
                            "many_faces": modules.globals.many_faces,
                            "color_correction": modules.globals.color_correction,
                            "mouth_mask": modules.globals.mouth_mask,
                            "frame_index": frame_index, # Pass frame index if needed
                            "total_frames": num_frames, # Pass total frames if needed
                            # Pass simple_map if face mapping is active
                            "simple_map": modules.globals.simple_map if modules.globals.map_faces else None,
                        }
                        # Filter params or use **kwargs if processor accepts them

                        temp_frame = processor.process_frame(processor_params)

                        if temp_frame is None:
                             update_status(f'Warning: Processor {processor_name} returned None for frame {frame_index}. Using previous frame state.', 'WARN')
                             # Keep processed_frame as it was before this processor
                        else:
                             processed_frame = temp_frame # Update frame state for the next processor

                        # Optimization: Conditional resource release inside loop if memory is tight
                        # if frame_index % 50 == 0: release_resources()

                    except Exception as proc_e:
                        update_status(f'Error applying processor {processor_name} on frame {frame_index}: {proc_e}', 'ERROR')
                        # Option: Skip frame vs. Abort entirely
                        # For now, we continue processing the frame with subsequent processors, using the last valid state
                        pass # Continue with next processor on this frame

                # 3. Write Processed Frame back to temp location (overwrite original temp frame)
                # This ensures create_video reads the modified frames
                save_success = cv2.imwrite(frame_path, processed_frame)
                if not save_success:
                     update_status(f'Warning: Failed to save processed frame {frame_path}. Video might contain unprocessed frame.', 'WARN')

                # 4. Release resources periodically (e.g., every N frames or based on time)
                if frame_index % 25 == 0 or frame_index == num_frames - 1: # Release every 25 frames and on the last frame
                    release_resources()

            except Exception as frame_e:
                update_status(f'Error processing frame {frame_index} at path {frame_path}: {frame_e}', 'ERROR')
                import traceback
                traceback.print_exc()
                # Option: Continue to next frame or abort? Continue for robustness.

        update_status('Frame processing loop finished.')

        # --- Create Video from Processed Frames ---
        update_status('Creating video from processed frames...')
        # Define temp output path before audio restoration
        temp_output_dir = get_temp_directory_path(modules.globals.target_path) # Get base temp dir
        if not temp_output_dir: temp_output_dir = os.path.dirname(modules.globals.output_path) # Fallback
        temp_output_video_path = os.path.join(temp_output_dir, f"temp_{os.path.basename(modules.globals.output_path)}")

        create_success = create_video(modules.globals.target_path, video_fps, temp_output_video_path)
        if not create_success:
             update_status('Error: Failed to create video from processed frames.', 'ERROR')
             # Cleanup might still run in finally block
             return # Stop here

        # --- Handle Audio Restoration ---
        final_output_path = modules.globals.output_path
        if modules.globals.keep_audio:
            update_status('Restoring audio...', 'FFmpeg')
            if not modules.globals.keep_fps:
                update_status('Warning: Audio restoration enabled without --keep-fps. Sync issues may occur.', 'WARN')

            # Ensure final output directory exists
            final_output_dir = os.path.dirname(final_output_path)
            if final_output_dir and not os.path.exists(final_output_dir): os.makedirs(final_output_dir)

            # Restore audio from original target to the temp video, outputting to final path
            audio_success = restore_audio(modules.globals.target_path, temp_output_video_path, final_output_path)
            if audio_success:
                update_status('Audio restoration complete.')
            else:
                 update_status('Error: Audio restoration failed. Video saved without audio.', 'ERROR')
                 # As a fallback, move the no-audio video to the final path
                 try:
                      if os.path.exists(final_output_path): os.remove(final_output_path)
                      shutil.move(temp_output_video_path, final_output_path)
                      update_status(f'Fallback: Saved video without audio to {final_output_path}')
                      temp_output_video_path = None # Prevent deletion in finally
                 except Exception as move_e:
                      update_status(f'Error moving temporary video after failed audio restore: {move_e}', 'ERROR')

        else:
            # No audio requested, move the temp video to the final output path
            update_status('Moving temporary video to final output path (no audio).')
            try:
                # Ensure final output directory exists
                final_output_dir = os.path.dirname(final_output_path)
                if final_output_dir and not os.path.exists(final_output_dir): os.makedirs(final_output_dir)

                if os.path.abspath(temp_output_video_path) != os.path.abspath(final_output_path):
                     if os.path.exists(final_output_path):
                         os.remove(final_output_path) # Remove existing destination file first
                     shutil.move(temp_output_video_path, final_output_path)
                     temp_output_video_path = None # Prevent deletion in finally block
                else:
                     update_status("Temporary video path is same as final output path. No move needed.", "WARN")
                     temp_output_video_path = None # Still prevent deletion

            except Exception as move_e:
                 update_status(f'Error moving temporary video to final destination: {move_e}', 'ERROR')


        # --- Validation ---
        if os.path.exists(final_output_path) and is_video(final_output_path):
            update_status('Video processing finished successfully.')
        else:
             update_status('Error: Final output video file not found or invalid after processing.', 'ERROR')

    except Exception as e:
        update_status(f'An unexpected error occurred during video processing: {e}', 'ERROR')
        import traceback
        traceback.print_exc()

    finally:
        # --- Clean Up Temporary Resources ---
        if not modules.globals.keep_frames:
             update_status("Cleaning temporary frame files...", "Temp")
             clean_temp(modules.globals.target_path)
        else:
             update_status("Keeping temporary frame files (--keep-frames enabled).", "Temp")

        # Remove intermediate temp video file if it exists and wasn't moved
        if temp_output_video_path and os.path.exists(temp_output_video_path):
             try:
                 os.remove(temp_output_video_path)
                 update_status(f"Removed intermediate video file: {temp_output_video_path}", "Temp")
             except OSError as e:
                 update_status(f"Warning: Could not remove intermediate video file {temp_output_video_path}: {e}", "WARN")
        # Final resource release
        release_resources()


def destroy(to_quit: bool = True) -> None:
    """Cleans up temporary files, releases resources, and optionally exits."""
    update_status("Initiating shutdown sequence...", "CLEANUP")

    # Clean temp files only if target_path was set and keep_frames is false
    if hasattr(modules.globals, 'target_path') and modules.globals.target_path and \
       hasattr(modules.globals, 'keep_frames') and not modules.globals.keep_frames:
        update_status("Cleaning temporary files (if any)...", "CLEANUP")
        clean_temp(modules.globals.target_path)

    # Release models and GPU memory
    update_status("Releasing resources...", "CLEANUP")
    release_resources()

    # Explicitly clear processor instances (helps GC)
    global FRAME_PROCESSORS_INSTANCES
    if FRAME_PROCESSORS_INSTANCES:
        # Call destroy method on processors if they have one
        for processor in FRAME_PROCESSORS_INSTANCES:
            if hasattr(processor, 'destroy') and callable(processor.destroy):
                try:
                    processor.destroy()
                except Exception as e:
                     print(f"\033[33mWarning: Error destroying processor {getattr(processor, 'NAME', '?')}: {e}\033[0m")
        FRAME_PROCESSORS_INSTANCES.clear()

    # Clear other potentially large global variables explicitly (optional)
    if hasattr(modules.globals, 'source_target_map'): modules.globals.source_target_map = []
    if hasattr(modules.globals, 'simple_map'): modules.globals.simple_map = {}
    # Clear analyser cache (if it holds significant data)
    global FACE_ANALYSER
    FACE_ANALYSER = None # Allow GC to collect it
    global _ort_session # For NSFW predictor
    _ort_session = None

    gc.collect() # Final GC run

    update_status("Cleanup complete.", "CLEANUP")
    if to_quit:
        print("Exiting application.")
        os._exit(0) # Use os._exit for a more forceful exit if needed, sys.exit(0) is generally preferred


def run() -> None:
    """Parses arguments, sets up environment, and starts processing or UI."""
    # Set TERM environment variable for tqdm on Windows (helps with progress bar rendering)
    if platform.system().lower() == 'windows':
         os.environ['TERM'] = 'xterm' # Or 'vt100'

    parser = parse_args() # Parse arguments first to set globals

    # Apply GPU Memory Limit early, requires execution_providers to be set
    limit_gpu_memory(GPU_MEMORY_LIMIT_FRACTION)

    # Perform pre-checks (dependencies, versions, paths)
    if not pre_check():
        # Display help if critical checks fail in headless mode (e.g., missing paths)
        if modules.globals.headless:
             print("\033[31mCritical pre-check failed. Please review errors above.\033[0m")
             parser.print_help()
        destroy(to_quit=True)
        return # Exit if pre-checks fail

    # Limit other resources (CPU RAM, TF GPU options)
    limit_resources()

    # --- Processor Requirements Check ---
    # Moved after parse_args and resource limits
    active_processor_modules = get_frame_processors_modules(modules.globals.frame_processors)
    all_processors_ready = True
    if not active_processor_modules:
         update_status('Error: No valid frame processors specified or found.', 'ERROR')
         all_processors_ready = False
    else:
        for processor_module in active_processor_modules:
            processor_name = getattr(processor_module, 'NAME', 'UnknownProcessor')
            update_status(f'Checking requirements for {processor_name}...')
            try:
                 if not processor_module.pre_check():
                     update_status(f'Requirements check failed for {processor_name}.', 'ERROR')
                     all_processors_ready = False
                     # Don't break early, report all failed checks
                 else:
                     update_status(f'Requirements met for {processor_name}.')
            except Exception as e:
                 update_status(f'Error during requirements check for {processor_name}: {e}', 'ERROR')
                 all_processors_ready = False

    if not all_processors_ready:
         update_status('One or more frame processors failed requirement checks. Please review messages above.', 'ERROR')
         destroy(to_quit=True)
         return

    # --- Run Mode ---
    if modules.globals.headless:
        update_status('Running in headless mode.')
        # Face mapping requires specific setup before starting the main processing
        if modules.globals.map_faces:
            update_status("Mapping faces enabled, analyzing target...", "Face Analyser")
            if is_video(modules.globals.target_path):
                 from modules.face_analyser import get_unique_faces_from_target_video
                 get_unique_faces_from_target_video()
            elif is_image(modules.globals.target_path):
                 from modules.face_analyser import get_unique_faces_from_target_image
                 get_unique_faces_from_target_image()
            else:
                 update_status("Map faces requires a valid target image or video.", "ERROR")
                 destroy(to_quit=True)
                 return
            update_status("Target analysis for face mapping complete.", "Face Analyser")

        start() # Run the main processing function
        destroy(to_quit=True) # Exit after headless processing
    else:
        # Launch UI
        update_status('Launching graphical user interface...')
        # Ensure destroy is callable without arguments for the UI close button
        destroy_wrapper = lambda: destroy(to_quit=True)
        try:
            window = ui.init(start, destroy_wrapper, modules.globals.lang)
            window.mainloop()
        except Exception as e:
             print(f"\033[31mFatal Error initializing or running the UI: {e}\033[0m")
             import traceback
             traceback.print_exc()
             destroy(to_quit=True) # Attempt cleanup and exit even if UI fails


# --- Main execution entry point ---
if __name__ == "__main__":
    # Add project root to Python path (if core.py is not at the very top level)
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(script_dir) # Adjust if structure differs
    # if project_root not in sys.path:
    #      sys.path.insert(0, project_root)

    run()
# --- END OF FILE core.py ---