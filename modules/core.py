# --- START OF FILE core.py ---

import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
# Check if CUDAExecutionProvider is likely intended
_cuda_intended = False
if '--execution-provider' in sys.argv:
    try:
        providers_index = sys.argv.index('--execution-provider')
        # Check subsequent arguments until the next option (starts with '-') or end of list
        for i in range(providers_index + 1, len(sys.argv)):
            if sys.argv[i].startswith('-'):
                break
            if 'cuda' in sys.argv[i].lower():
                _cuda_intended = True
                break
    except ValueError:
        pass # --execution-provider not found
# Less precise check if the above fails or isn't used (e.g. deprecated --gpu-vendor nvidia)
if not _cuda_intended and any('cuda' in arg.lower() or 'nvidia' in arg.lower() for arg in sys.argv):
     _cuda_intended = True

if _cuda_intended:
    print("[DLC.CORE] CUDA execution provider detected or inferred, setting OMP_NUM_THREADS=1.")
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List, Optional
import platform
import signal
import shutil
import argparse
import gc # Garbage Collector

# --- ONNX Runtime Version Check ---
# Ensure ONNX Runtime is imported and check version compatibility if needed.
# As of onnxruntime 1.19, the core APIs used here (get_available_providers, InferenceSession config)
# remain stable. No specific code changes are required *in this file* for 1.19 compatibility,
# assuming frame processors use standard SessionOptions/InferenceSession creation.
try:
    import onnxruntime
    # print(f"[DLC.CORE] Using ONNX Runtime version: {onnxruntime.__version__}") # Optional: uncomment for debug
    # Example future check:
    # from packaging import version
    # if version.parse(onnxruntime.__version__) < version.parse("1.19.0"):
    #     print(f"Warning: ONNX Runtime version {onnxruntime.__version__} is older than 1.19. Some features might differ.")
except ImportError:
    print("\033[31m[DLC.CORE] Error: ONNX Runtime is not installed. Please install it (e.g., `pip install onnxruntime` or `pip install onnxruntime-gpu`).\033[0m")
    sys.exit(1)

# --- PyTorch Conditional Import ---
_torch_available = False
_torch_cuda_available = False
try:
    import torch
    _torch_available = True
    if torch.cuda.is_available():
        _torch_cuda_available = True
except ImportError:
    # Warning only if CUDA EP might be used, otherwise PyTorch is optional
    if _cuda_intended:
        print("[DLC.CORE] Warning: PyTorch not found or CUDA not available. GPU memory limiting via Torch is disabled.")
    pass # Keep torch=None or handle appropriately

# --- TensorFlow Conditional Import (for resource limiting) ---
_tensorflow_available = False
try:
    import tensorflow
    _tensorflow_available = True
except ImportError:
    print("[DLC.CORE] Info: TensorFlow not found. GPU memory growth configuration for TensorFlow will be skipped.")
    pass

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

# Configuration for GPU Memory Limit (0.8 = 80%)
GPU_MEMORY_LIMIT_FRACTION = 0.8

# Check if ROCM is chosen early, before parse_args if possible, or handle after
_is_rocm_selected = False
# A simple check; parse_args will give the definitive list later
if any('rocm' in arg.lower() for arg in sys.argv):
    _is_rocm_selected = True

if _is_rocm_selected and _torch_available:
    # If ROCM is selected, torch might interfere or not be needed.
    # Let's keep the behavior of unloading it for safety, as ROCm support in PyTorch can be complex.
    print("[DLC.CORE] ROCM detected or selected, unloading PyTorch to prevent potential conflicts.")
    del torch
    _torch_available = False
    _torch_cuda_available = False
    gc.collect() # Try to explicitly collect garbage


warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=40)) # Wider help
    program.add_argument('-s', '--source', help='Path to the source image file', dest='source_path')
    program.add_argument('-t', '--target', help='Path to the target image or video file', dest='target_path')
    program.add_argument('-o', '--output', help='Path for the output file or directory', dest='output_path')
    # Frame processors - Updated choices might be needed if new processors are added
    available_processors = ['face_swapper', 'face_enhancer'] # Dynamically get these if possible in future
    program.add_argument('--frame-processor', help='Pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=available_processors, nargs='+')
    program.add_argument('--keep-fps', help='Keep the original frames per second (FPS) of the target video', dest='keep_fps', action='store_true')
    program.add_argument('--keep-audio', help='Keep the original audio of the target video (requires --keep-fps for perfect sync)', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='Keep the temporary extracted frames after processing', dest='keep_frames', action='store_true')
    program.add_argument('--many-faces', help='Process all detected faces in the target, not just the most similar', dest='many_faces', action='store_true')
    program.add_argument('--nsfw-filter', help='Enable NSFW content filtering (experimental, image-only currently)', dest='nsfw_filter', action='store_true')
    program.add_argument('--map-faces', help='EXPERIMENTAL: Map source faces to target faces based on order or index. Requires manual setup or specific naming conventions.', dest='map_faces', action='store_true')
    program.add_argument('--mouth-mask', help='Apply a mask over the mouth region during processing (specific to certain processors)', dest='mouth_mask', action='store_true')
    program.add_argument('--video-encoder', help='Encoder for the output video', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc']) # Added NVENC options
    program.add_argument('--video-quality', help='Quality for the output video (lower value means higher quality, range depends on encoder)', dest='video_quality', type=int, default=18, metavar='[0-51 for x264/x265, 0-63 for vp9]') # Adjusted range note
    program.add_argument('-l', '--lang', help='User interface language code (e.g., "en", "es")', default="en")
    program.add_argument('--live-mirror', help='Mirror the live camera preview (like a webcam)', dest='live_mirror', action='store_true')
    program.add_argument('--live-resizable', help='Allow resizing the live camera preview window', dest='live_resizable', action='store_true')
    program.add_argument('--max-memory', help='DEPRECATED (use with caution): Approx. maximum CPU RAM in GB. Less effective than GPU limits.', dest='max_memory', type=int) # Removed default, let suggest_max_memory handle it dynamically if needed
    # Execution Provider - Updated based on ONNX Runtime 1.19 common providers
    program.add_argument('--execution-provider', help='Execution provider(s) to use (e.g., cuda, cpu, rocm, dml, coreml). Order determines priority.', dest='execution_provider', default=suggest_execution_providers(), choices=get_available_execution_providers_short(), nargs='+')
    program.add_argument('--execution-threads', help='Number of threads for the execution provider', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version} (ONNX Runtime: {onnxruntime.__version__})') # Added ORT version

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated', choices=['apple', 'nvidia', 'amd'])
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    # Set default for max_memory if not provided
    if args.max_memory is None:
        args.max_memory = suggest_max_memory()

    # Process deprecated args first
    handle_deprecated_args(args)

    # Assign to globals
    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    # Headless mode is determined by the presence of CLI args for paths
    modules.globals.headless = bool(args.source_path or args.target_path or args.output_path)
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio # Note: keep_audio without keep_fps can cause sync issues
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory # Still set, but primarily for CPU RAM limit now
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider) # Decode selected short names
    modules.globals.execution_threads = args.execution_threads
    modules.globals.lang = args.lang

    # Update derived globals
    modules.globals.fp_ui = {proc: (proc in modules.globals.frame_processors) for proc in available_processors} # Simplified UI state init

    # Validate keep_audio / keep_fps combination
    if modules.globals.keep_audio and not modules.globals.keep_fps and not modules.globals.headless:
         # Only warn in interactive mode, CLI users are expected to know
        print("\033[33mWarning: --keep-audio is enabled but --keep-fps is disabled. This might cause audio/video synchronization issues.\033[0m")
    elif modules.globals.keep_audio and not modules.globals.target_path:
         print("\033[33mWarning: --keep-audio is enabled but no target video path is provided. Audio cannot be kept.\033[0m")
         modules.globals.keep_audio = False

def handle_deprecated_args(args: argparse.Namespace) -> None:
    """Handles deprecated arguments and updates corresponding new arguments if necessary."""
    if args.source_path_deprecated:
        print('\033[33mArgument -f/--face is deprecated. Use -s/--source instead.\033[0m')
        if not args.source_path: # Only override if --source wasn't set
            args.source_path = args.source_path_deprecated
            # Re-evaluate output path based on deprecated source (normalize_output_path handles this later)

    # Track if execution_threads was explicitly set by the user via --execution-threads
    # This requires checking sys.argv as argparse doesn't directly expose this.
    threads_explicitly_set = '--execution-threads' in sys.argv

    if args.cpu_cores_deprecated is not None:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        # Only override if --execution-threads wasn't explicitly set
        if not threads_explicitly_set:
            args.execution_threads = args.cpu_cores_deprecated
            threads_explicitly_set = True # Mark as set now

    if args.gpu_threads_deprecated is not None:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        # Only override if --execution-threads wasn't explicitly set (by user or cpu-cores)
        if not threads_explicitly_set:
             args.execution_threads = args.gpu_threads_deprecated
             threads_explicitly_set = True # Mark as set

    # Handle --gpu-vendor deprecation by modifying execution_provider list *if not explicitly set*
    ep_explicitly_set = '--execution-provider' in sys.argv

    if args.gpu_vendor_deprecated:
        print(f'\033[33mArgument --gpu-vendor {args.gpu_vendor_deprecated} is deprecated. Use --execution-provider instead.\033[0m')
        if not ep_explicitly_set:
            provider_map = {
                # Map vendor to preferred execution provider short names
                'apple': ['coreml', 'cpu'], # CoreML first
                'nvidia': ['cuda', 'cpu'],  # CUDA first
                'amd': ['rocm', 'cpu']      # ROCm first
                # 'intel': ['openvino', 'cpu'] # Example if OpenVINO support is relevant
            }
            if args.gpu_vendor_deprecated in provider_map:
                suggested_providers = provider_map[args.gpu_vendor_deprecated]
                print(f"Mapping deprecated --gpu-vendor {args.gpu_vendor_deprecated} to --execution-provider {' '.join(suggested_providers)}")
                args.execution_provider = suggested_providers # Set the list of short names
            else:
                 print(f'\033[33mWarning: Unknown --gpu-vendor {args.gpu_vendor_deprecated}. Default execution providers will be used.\033[0m')
        else:
             print(f'\033[33mWarning: --gpu-vendor {args.gpu_vendor_deprecated} is ignored because --execution-provider was explicitly set.\033[0m')

def get_available_execution_providers_full() -> List[str]:
    """Returns the full names of available ONNX Runtime execution providers."""
    try:
        return onnxruntime.get_available_providers()
    except AttributeError:
        # Fallback for very old versions or unexpected issues
        print("\033[33mWarning: Could not dynamically get available providers. Falling back to common defaults.\033[0m")
        # Provide a reasonable guess
        defaults = ['CPUExecutionProvider']
        if _cuda_intended: defaults.insert(0, 'CUDAExecutionProvider')
        if _is_rocm_selected: defaults.insert(0, 'ROCMExecutionProvider')
        # Add others based on platform if needed
        return defaults

def get_available_execution_providers_short() -> List[str]:
    """Returns the short names (lowercase) of available ONNX Runtime execution providers."""
    full_names = get_available_execution_providers_full()
    return [name.replace('ExecutionProvider', '').lower() for name in full_names]

def decode_execution_providers(selected_short_names: List[str]) -> List[str]:
    """Converts selected short names back to full ONNX Runtime provider names, preserving order and checking availability."""
    available_full_names = get_available_execution_providers_full()
    available_short_map = {name.replace('ExecutionProvider', '').lower(): name for name in available_full_names}
    decoded_providers = []
    valid_short_names_found = []

    for short_name in selected_short_names:
        name_lower = short_name.lower()
        if name_lower in available_short_map:
            full_name = available_short_map[name_lower]
            if full_name not in decoded_providers: # Avoid duplicates
                decoded_providers.append(full_name)
                valid_short_names_found.append(name_lower)
        else:
            print(f"\033[33mWarning: Requested execution provider '{short_name}' is not available or not recognized. Skipping.\033[0m")

    if not decoded_providers:
        print("\033[33mWarning: No valid execution providers selected or available. Falling back to CPU.\033[0m")
        if 'CPUExecutionProvider' in available_full_names:
            decoded_providers = ['CPUExecutionProvider']
            valid_short_names_found.append('cpu')
        else:
             print("\033[31mError: CPUExecutionProvider is not available in this build of ONNX Runtime. Cannot proceed.\033[0m")
             sys.exit(1) # Critical error

    print(f"[DLC.CORE] Using execution providers: {valid_short_names_found} (Full names: {decoded_providers})")
    return decoded_providers


def suggest_max_memory() -> int:
    """Suggests a default max CPU RAM limit in GB. Less critical now with GPU limits."""
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        # Suggest slightly less than half of total RAM, capped at a reasonable upper limit (e.g., 64GB)
        # and a minimum (e.g., 4GB)
        suggested = max(4, min(int(total_ram_gb * 0.4), 64))
        # print(f"[DLC.CORE] Auto-suggesting max_memory: {suggested} GB (based on total system RAM: {total_ram_gb:.1f} GB)")
        return suggested
    except (ImportError, OSError):
        print("[DLC.CORE] Info: psutil not found or failed. Using fallback default for max_memory suggestion (16 GB).")
        # Fallback defaults similar to original code
        if platform.system().lower() == 'darwin':
            return 8 # Increased macOS default slightly
        return 16 # Keep higher default for Linux/Windows


def suggest_execution_providers() -> List[str]:
    """Suggests a default list of execution providers based on availability and platform."""
    available_short = get_available_execution_providers_short()
    preferred_providers = []

    # Prioritize GPU providers if available
    if 'cuda' in available_short:
        preferred_providers.append('cuda')
    elif 'rocm' in available_short:
        preferred_providers.append('rocm')
    elif 'dml' in available_short and platform.system().lower() == 'windows':
         preferred_providers.append('dml') # DirectML on Windows
    elif 'coreml' in available_short and platform.system().lower() == 'darwin':
         preferred_providers.append('coreml') # CoreML on macOS

    # Always include CPU as a fallback
    if 'cpu' in available_short:
        preferred_providers.append('cpu')
    elif available_short: # If CPU is somehow missing, add the first available one
        preferred_providers.append(available_short[0])

    # If list is empty (shouldn't happen if get_available works), default to cpu
    if not preferred_providers:
        return ['cpu']

    # print(f"[DLC.CORE] Suggested execution providers: {preferred_providers}") # Optional debug info
    return preferred_providers


def suggest_execution_threads() -> int:
    """Suggests a sensible default number of execution threads based on CPU cores."""
    try:
        logical_cores = os.cpu_count() or 4 # Default to 4 if cpu_count fails
        # Use slightly fewer threads than logical cores, capped.
        # Good balance between parallelism and overhead.
        suggested_threads = max(1, min(logical_cores - 1 if logical_cores > 1 else 1, 16))
        # Don't suggest 1 for CUDA/ROCm implicitly here, let user override or frame processors decide.
        # The SessionOptions in the processors should handle provider-specific thread settings if needed.
        # print(f"[DLC.CORE] Auto-suggesting execution_threads: {suggested_threads} (based on {logical_cores} logical cores)")
        return suggested_threads
    except NotImplementedError:
        print("[DLC.CORE] Warning: os.cpu_count() not implemented. Using fallback default for execution_threads (4).")
        return 4 # Fallback


def limit_gpu_memory(fraction: float) -> None:
    """Attempts to limit GPU memory usage, primarily via PyTorch if CUDA is used."""
    # Check if CUDAExecutionProvider is in the *actually selected* providers
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        if _torch_cuda_available:
            try:
                # Ensure CUDA is initialized if needed (might not be necessary, but safe)
                if not torch.cuda.is_initialized():
                     torch.cuda.init()

                device_count = torch.cuda.device_count()
                if device_count > 0:
                    # Limit memory on the default device (usually device 0)
                    # Note: This limits PyTorch's allocation pool. ONNX Runtime might manage
                    # its CUDA memory somewhat separately, but this can still help prevent
                    # PyTorch from grabbing everything.
                    print(f"[DLC.CORE] Attempting to limit PyTorch CUDA memory fraction to {fraction:.1%} on device 0")
                    torch.cuda.set_per_process_memory_fraction(fraction, 0)
                    # Optional: Check memory after setting limit
                    total_mem = torch.cuda.get_device_properties(0).total_memory
                    reserved_mem = torch.cuda.memory_reserved(0)
                    allocated_mem = torch.cuda.memory_allocated(0)
                    print(f"[DLC.CORE] PyTorch CUDA memory limit hint set. Device 0 Total: {total_mem / 1024**3:.2f} GB. "
                          f"PyTorch Reserved: {reserved_mem / 1024**3:.2f} GB, Allocated: {allocated_mem / 1024**3:.2f} GB.")
                else:
                    print("\033[33mWarning: PyTorch reports no CUDA devices available, cannot set memory limit.\033[0m")

            except RuntimeError as e:
                 print(f"\033[33mWarning: PyTorch CUDA runtime error during memory limit setting (may already be initialized?): {e}\033[0m")
            except Exception as e:
                print(f"\033[33mWarning: Failed to set PyTorch CUDA memory fraction: {e}\033[0m")
        else:
            # Only warn if PyTorch CUDA specifically isn't available, but CUDA EP was chosen.
            if _cuda_intended: # Check original intent
                print("\033[33mWarning: CUDAExecutionProvider selected, but PyTorch CUDA is not available. Cannot apply PyTorch memory limit.\033[0m")
    # Add future limits for other providers if ONNX Runtime API supports it directly
    # Example placeholder for potential future ONNX Runtime API:
    # elif 'ROCMExecutionProvider' in modules.globals.execution_providers:
    #     try:
    #         # Hypothetical ONNX Runtime API
    #         ort_options = onnxruntime.SessionOptions()
    #         ort_options.add_provider_options('rocm', {'gpu_mem_limit': str(int(total_mem_bytes * fraction))})
    #         print("[DLC.CORE] Note: ROCm memory limit set via ONNX Runtime provider options (if API exists).")
    #     except Exception as e:
    #         print(f"\033[33mWarning: Failed to set ROCm memory limit via hypothetical ORT options: {e}\033[0m")
    # else:
    #     print("[DLC.CORE] GPU memory limit not applied (PyTorch CUDA not used or unavailable).")


def limit_resources() -> None:
    """Limits system resources like CPU RAM (best effort) and sets TensorFlow GPU options."""
    # 1. Limit CPU RAM (Best-effort, OS-dependent)
    if modules.globals.max_memory and modules.globals.max_memory > 0:
        limit_gb = modules.globals.max_memory
        limit_bytes = limit_gb * (1024 ** 3)
        current_system = platform.system().lower()

        try:
            if current_system == 'linux' or current_system == 'darwin':
                import resource
                # RLIMIT_AS (virtual memory) is often more effective than RLIMIT_DATA
                try:
                    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                    # Set soft limit; hard limit usually requires root. Don't exceed current hard limit.
                    new_soft = min(limit_bytes, hard)
                    resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
                    print(f"[DLC.CORE] Limited process virtual memory (CPU RAM approximation) soft limit towards ~{limit_gb} GB.")
                except (ValueError, resource.error) as e:
                    print(f"\033[33mWarning: Failed to set virtual memory limit (RLIMIT_AS): {e}\033[0m")
                    # Fallback attempt using RLIMIT_DATA (less effective for total memory)
                    try:
                         soft_data, hard_data = resource.getrlimit(resource.RLIMIT_DATA)
                         new_soft_data = min(limit_bytes, hard_data)
                         resource.setrlimit(resource.RLIMIT_DATA, (new_soft_data, hard_data))
                         print(f"[DLC.CORE] Limited process data segment (partial CPU RAM) soft limit towards ~{limit_gb} GB.")
                    except (ValueError, resource.error) as e_data:
                         print(f"\033[33mWarning: Failed to set data segment limit (RLIMIT_DATA): {e_data}\033[0m")

            elif current_system == 'windows':
                # Windows memory limiting is complex. SetProcessWorkingSetSizeEx is more of a suggestion.
                # Job Objects are the robust way but much more involved. Keep the hint for now.
                import ctypes
                kernel32 = ctypes.windll.kernel32
                process_handle = kernel32.GetCurrentProcess()
                # Flags: QUOTA_LIMITS_HARDWS_ENABLE (1) requires special privileges, use 0 for min/max hint only
                # Using min=1MB, max=limit_bytes. Returns non-zero on success.
                min_ws = ctypes.c_size_t(1024 * 1024)
                max_ws = ctypes.c_size_t(limit_bytes)
                if not kernel32.SetProcessWorkingSetSizeEx(process_handle, min_ws, max_ws, 0):
                    error_code = ctypes.get_last_error()
                    print(f"\033[33mWarning: Failed to set process working set size hint (Windows). Error code: {error_code}. This limit may not be enforced.\033[0m")
                else:
                    print(f"[DLC.CORE] Requested process working set size hint (Windows memory guidance) max ~{limit_gb} GB.")
            else:
                 print(f"\033[33mWarning: CPU RAM limiting not implemented for platform {current_system}. --max-memory ignored.\033[0m")

        except ImportError:
             print(f"\033[33mWarning: 'resource' module (Unix) not available. Cannot limit CPU RAM via setrlimit.\033[0m")
        except Exception as e:
             print(f"\033[33mWarning: An unexpected error occurred during CPU RAM limiting: {e}\033[0m")
    # else:
    #     print("[DLC.CORE] Info: CPU RAM limit (--max-memory) not set or disabled.")


    # 2. Configure TensorFlow GPU memory (if TensorFlow is installed)
    if _tensorflow_available:
        try:
            gpus = tensorflow.config.experimental.list_physical_devices('GPU')
            if gpus:
                configured_gpus = 0
                for gpu in gpus:
                    try:
                        # Allow memory growth instead of pre-allocating everything
                        tensorflow.config.experimental.set_memory_growth(gpu, True)
                        # print(f"[DLC.CORE] Enabled TensorFlow memory growth for GPU: {gpu.name}")
                        configured_gpus += 1
                    except RuntimeError as e:
                        # Memory growth must be set before GPUs have been initialized
                        print(f"\033[33mWarning: Could not set TensorFlow memory growth for {gpu.name} (may already be initialized): {e}\033[0m")
                    except Exception as e_inner: # Catch other potential TF config errors
                         print(f"\033[33mWarning: Error configuring TensorFlow memory growth for {gpu.name}: {e_inner}\033[0m")
                if configured_gpus > 0:
                     print(f"[DLC.CORE] Enabled TensorFlow memory growth for {configured_gpus} GPU(s).")
            # else:
            #     print("[DLC.CORE] No TensorFlow physical GPUs detected.")
        except Exception as e:
            print(f"\033[33mWarning: Error listing or configuring TensorFlow GPU devices: {e}\033[0m")
    # else:
    #     print("[DLC.CORE] TensorFlow not available, skipping TF GPU configuration.")


def release_resources() -> None:
    """Releases resources, especially GPU memory caches."""
    # Clear PyTorch CUDA cache if applicable and PyTorch CUDA is available
    if 'CUDAExecutionProvider' in modules.globals.execution_providers and _torch_cuda_available:
        try:
            torch.cuda.empty_cache()
            # print("[DLC.CORE] Cleared PyTorch CUDA cache.") # Optional: uncomment for verbose logging
        except Exception as e:
             print(f"\033[33mWarning: Failed to clear PyTorch CUDA cache: {e}\033[0m")

    # Add potential cleanup for other frameworks or ONNX Runtime sessions if needed
    # (Usually session objects going out of scope and gc.collect() is sufficient for ORT C++ backend)

    # Explicitly run garbage collection
    # This helps release Python-level objects, which might then trigger
    # the release of underlying resources (like ONNX Runtime session memory)
    gc.collect()
    # print("[DLC.CORE] Ran garbage collector.") # Optional: uncomment for verbose logging


def pre_check() -> bool:
    """Performs essential pre-run checks for dependencies and versions."""
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to Python 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg command not found in PATH. Please install ffmpeg and ensure it is accessible.')
        return False

    # ONNX Runtime was checked at import time, but double check here if needed.
    # The import would have failed earlier if it's not installed.
    # print(f"[DLC.CORE] Using ONNX Runtime version: {onnxruntime.__version__}")

    # TensorFlow check (optional, only issue warning if unavailable)
    if not _tensorflow_available:
        update_status('TensorFlow not found. Some features like GPU memory growth setting will be skipped.', scope='INFO')
        # Decide if TF is strictly required by any processor. If so, change to error and return False.
        # Currently, it seems only used for optional resource limiting.

    # Check PyTorch availability *only if* CUDA EP is selected
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
         if not _torch_available:
             update_status('CUDAExecutionProvider selected, but PyTorch is not installed. Install PyTorch with CUDA support (see PyTorch website).', scope='ERROR')
             return False
         if not _torch_cuda_available:
             update_status('CUDAExecutionProvider selected, but torch.cuda.is_available() is False. Check PyTorch CUDA installation, GPU drivers, and CUDA toolkit compatibility.', scope='ERROR')
             return False

    # Check if selected video encoder potentially requires specific hardware/drivers (e.g., NVENC)
    if modules.globals.video_encoder in ['h264_nvenc', 'hevc_nvenc']:
        # This check is basic. FFmpeg needs to be compiled with NVENC support,
        # and NVIDIA drivers must be installed. We can't easily verify this from Python.
        # Just issue an informational note.
        update_status(f"Selected video encoder '{modules.globals.video_encoder}' requires an NVIDIA GPU and correctly configured FFmpeg/drivers.", scope='INFO')
        if 'CUDAExecutionProvider' not in modules.globals.execution_providers:
             update_status(f"Warning: NVENC encoder selected, but CUDAExecutionProvider is not active. Ensure FFmpeg can access the GPU independently.", scope='WARN')

    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    """Prints status messages and updates UI if not headless."""
    formatted_message = f'[{scope}] {message}'
    print(formatted_message)
    if not modules.globals.headless:
        # Ensure ui module and update_status function exist and are callable
        if hasattr(ui, 'update_status') and callable(ui.update_status):
            try:
                # Use a mechanism that's safe for cross-thread UI updates if necessary
                # (e.g., queue or wx.CallAfter if using wxPython)
                # Assuming direct call is okay for now based on original structure.
                ui.update_status(message) # Pass the original message without scope prefix
            except Exception as e:
                # Avoid crashing core process for UI update errors
                print(f"[DLC.CORE] Error updating UI status: {e}")
        # else:
        #      print("[DLC.CORE] UI or ui.update_status not available for status update.")


def start() -> None:
    """Main processing logic: routes to image or video processing."""
    # Ensure frame processors are ready (this also initializes them)
    try:
        active_processors = get_frame_processors_modules(modules.globals.frame_processors)
        if not active_processors:
            update_status("No valid frame processors selected or loaded. Aborting.", "ERROR")
            return

        all_processors_initialized = True
        for frame_processor in active_processors:
            update_status(f'Initializing frame processor: {getattr(frame_processor, "NAME", "UnknownProcessor")}...')
            # The pre_start method should handle model loading and initial setup.
            # It might raise exceptions or return False on failure.
            if not hasattr(frame_processor, 'pre_start') or not callable(frame_processor.pre_start):
                 update_status(f'Processor {getattr(frame_processor, "NAME", "UnknownProcessor")} lacks a pre_start method.', 'WARN')
                 continue # Or treat as failure?

            if not frame_processor.pre_start():
                update_status(f'Initialization failed for {getattr(frame_processor, "NAME", "UnknownProcessor")}. Aborting.', 'ERROR')
                all_processors_initialized = False
                break # Stop initialization if one fails

        if not all_processors_initialized:
            return # Abort if any processor failed to initialize

    except Exception as e:
        update_status(f"Error during frame processor initialization: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return

    # --- Route based on target type ---
    if not modules.globals.target_path or not os.path.exists(modules.globals.target_path):
        update_status(f"Target path '{modules.globals.target_path}' not found or not specified.", "ERROR")
        return

    if has_image_extension(modules.globals.target_path) and is_image(modules.globals.target_path):
        process_image_target(active_processors)
    elif is_video(modules.globals.target_path):
        process_video_target(active_processors)
    else:
        update_status(f"Target path '{modules.globals.target_path}' is not a recognized image or video file.", "ERROR")


def process_image_target(active_processors: List) -> None:
    """Handles processing when the target is an image."""
    update_status('Processing image target...')
    # NSFW check (basic, for image only)
    if modules.globals.nsfw_filter:
         update_status('Checking image for NSFW content...', 'NSFW')
         # Assuming ui.check_and_ignore_nsfw is suitable for this
         if ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
             update_status('NSFW content detected and processing skipped.', 'NSFW')
             return # Stop processing

    try:
        # Ensure source path exists if needed by processors
        if not modules.globals.source_path or not os.path.exists(modules.globals.source_path):
             # Face swapping requires a source, enhancer might not. Check processor needs?
             if any(proc.NAME == 'face_swapper' for proc in active_processors): # Example check
                 update_status(f"Source image path '{modules.globals.source_path}' not found or not specified, required for face swapping.", "ERROR")
                 return

        # Ensure output directory exists
        output_dir = os.path.dirname(modules.globals.output_path)
        if output_dir and not os.path.exists(output_dir):
             try:
                 os.makedirs(output_dir, exist_ok=True)
                 print(f"[DLC.CORE] Created output directory: {output_dir}")
             except OSError as e:
                 update_status(f"Error creating output directory '{output_dir}': {e}", "ERROR")
                 return

        # Copy target to output path first to preserve metadata if possible and safe
        final_output_path = modules.globals.output_path
        temp_output_path = None # Use a temp path if overwriting source/target directly

        # Avoid overwriting input files directly during processing if they are the same as output
        if os.path.abspath(modules.globals.target_path) == os.path.abspath(final_output_path) or \
           (modules.globals.source_path and os.path.abspath(modules.globals.source_path) == os.path.abspath(final_output_path)):
            temp_output_path = os.path.join(output_dir, f"temp_image_{os.path.basename(final_output_path)}")
            print(f"[DLC.CORE] Output path conflicts with input, using temporary file: {temp_output_path}")
            shutil.copy2(modules.globals.target_path, temp_output_path)
            current_processing_file = temp_output_path
        else:
            # Copy target to final destination to start
            shutil.copy2(modules.globals.target_path, final_output_path)
            current_processing_file = final_output_path


        # Apply processors sequentially to the current file path
        source_for_processing = modules.globals.source_path
        output_for_processing = current_processing_file # Processors modify this file

        for frame_processor in active_processors:
            processor_name = getattr(frame_processor, "NAME", "UnknownProcessor")
            update_status(f'Applying {processor_name}...', processor_name)
            try:
                # Pass source, input_path (current state), output_path (same as input for in-place modification)
                frame_processor.process_image(source_for_processing, output_for_processing, output_for_processing)
                release_resources() # Release memory after each processor step
            except Exception as e:
                 update_status(f'Error during {processor_name} processing: {e}', 'ERROR')
                 import traceback
                 traceback.print_exc()
                 # Optionally clean up temp file and abort
                 if temp_output_path and os.path.exists(temp_output_path): os.remove(temp_output_path)
                 return

        # If a temporary file was used, move it to the final destination
        if temp_output_path:
            try:
                shutil.move(temp_output_path, final_output_path)
                print(f"[DLC.CORE] Moved temporary result to final output: {final_output_path}")
            except Exception as e:
                 update_status(f"Error moving temporary file to final output: {e}", "ERROR")
                 # Temp file might still exist, leave it for inspection?
                 return

        # Final check if output exists and is an image
        if os.path.exists(final_output_path) and is_image(final_output_path):
             update_status('Processing image finished successfully.')
        else:
             update_status('Processing image failed: Output file not found or invalid after processing.', 'ERROR')

    except Exception as e:
        update_status(f'An unexpected error occurred during image processing: {e}', 'ERROR')
        import traceback
        traceback.print_exc()
        # Clean up potentially corrupted output/temp file? Be cautious.
        # if temp_output_path and os.path.exists(temp_output_path): os.remove(temp_output_path)
        # if os.path.exists(final_output_path) and current_processing_file == final_output_path: # Careful not to delete original if copy failed
              # Consider what to do on failure - delete potentially corrupt output?


def process_video_target(active_processors: List) -> None:
    """Handles processing when the target is a video."""
    update_status('Processing video target...')

    # Basic check for source if needed (similar to image processing)
    if not modules.globals.source_path or not os.path.exists(modules.globals.source_path):
         if any(proc.NAME == 'face_swapper' for proc in active_processors):
             update_status(f"Source image path '{modules.globals.source_path}' not found or not specified, required for face swapping.", "ERROR")
             return

    # NSFW Check (Could be enhanced to sample frames, currently basic/skipped for video)
    if modules.globals.nsfw_filter:
        update_status('NSFW check for video is basic/experimental. Checking first frame...', 'NSFW')
        # Consider implementing frame sampling for a more robust check if needed
        # if ui.check_and_ignore_nsfw(modules.globals.target_path, destroy): # This might not work well for video
        #     update_status('NSFW content potentially detected (based on first frame check). Skipping.', 'NSFW')
        #     return
        update_status('NSFW check passed or skipped for video.', 'NSFW INFO')

    temp_output_video_path = None
    temp_frame_dir = None # Keep track of temp frame directory

    try:
        # --- Frame Extraction ---
        # map_faces might imply frames are already extracted or handled differently
        if not modules.globals.map_faces:
            update_status('Creating temporary resources for video frames...')
            # create_temp should return the path to the temp directory created
            temp_frame_dir = create_temp(modules.globals.target_path)
            if not temp_frame_dir:
                 update_status("Failed to create temporary directory for frames.", "ERROR")
                 return

            update_status('Extracting video frames...')
            # extract_frames needs the temp directory path
            # It should also ideally set modules.globals.video_fps based on the extracted video
            extract_frames(modules.globals.target_path, temp_frame_dir) # Pass temp dir
            update_status('Frame extraction complete.')
        else:
             update_status('Skipping frame extraction due to --map-faces flag.', 'INFO')
             # Assuming frames are already in the expected temp location or handled by processors
             temp_frame_dir = os.path.join(modules.globals.TEMP_DIRECTORY, os.path.basename(modules.globals.target_path)) # Need consistent temp path logic


        # Get paths to frames (extracted or pre-existing)
        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path) # This needs to know the temp dir structure
        if not temp_frame_paths:
            update_status('No frames found to process. Check temp folder or extraction step.', 'ERROR')
            # Clean up if temp dir was created
            if temp_frame_dir and not modules.globals.keep_frames: clean_temp(modules.globals.target_path)
            return

        update_status(f'Processing {len(temp_frame_paths)} frames...')

        # --- Frame Processing ---
        source_for_processing = modules.globals.source_path
        for frame_processor in active_processors:
            processor_name = getattr(frame_processor, "NAME", "UnknownProcessor")
            update_status(f'Applying {processor_name}...', processor_name)
            try:
                # process_video should modify frames in-place in the temp directory
                # It needs the source path and the list of frame paths
                frame_processor.process_video(source_for_processing, temp_frame_paths)
                release_resources() # Release memory after each processor completes its pass
            except Exception as e:
                 update_status(f'Error during {processor_name} frame processing: {e}', 'ERROR')
                 import traceback
                 traceback.print_exc()
                 # Abort processing
                 # Clean up temp frames if not keeping them
                 if temp_frame_dir and not modules.globals.keep_frames: clean_temp(modules.globals.target_path)
                 return

        # --- Video Creation ---
        update_status('Reconstructing video from processed frames...')
        fps = modules.globals.video_fps # Should be set by extract_frames or detected earlier

        if modules.globals.keep_fps:
            # Use the FPS detected during extraction (should be stored in globals.video_fps)
            if fps is None:
                update_status('Original FPS not detected during extraction, attempting fallback detection...', 'WARN')
                detected_fps = detect_fps(modules.globals.target_path)
                if detected_fps is not None:
                    fps = detected_fps
                    modules.globals.video_fps = fps # Store it back
                    update_status(f'Using fallback detected FPS: {fps:.2f}')
                else:
                    fps = 30.0 # Ultimate fallback
                    update_status("Could not detect FPS, using default 30.", "WARN")
            else:
                 update_status(f'Using original detected FPS: {fps:.2f}')
        else:
            fps = 30.0 # Use default fps if not keeping original
            update_status(f'Using fixed FPS: {fps:.2f}')

        # Define a temporary path for the video created *without* audio
        output_dir = os.path.dirname(modules.globals.output_path)
        if not output_dir: output_dir = '.' # Handle case where output is in current dir
        temp_output_video_filename = f"temp_{os.path.basename(modules.globals.output_path)}"
        # Ensure the temp filename doesn't clash if multiple runs happen concurrently (less likely in this app)
        temp_output_video_path = os.path.join(output_dir, temp_output_video_filename)

        # create_video needs the target path (for context?), fps, and the *temp* output path
        # It internally uses get_temp_frame_paths based on the target_path context.
        create_video(modules.globals.target_path, fps, temp_output_video_path)

        # --- Audio Handling ---
        final_output_path = modules.globals.output_path
        if modules.globals.keep_audio:
            update_status('Restoring audio...')
            if not modules.globals.keep_fps:
                update_status('Audio restoration may cause sync issues as FPS was not kept.', 'WARN')

            # restore_audio needs: original video (with audio), temp video (no audio), final output path
            restore_success = restore_audio(modules.globals.target_path, temp_output_video_path, final_output_path)

            if restore_success:
                update_status('Audio restoration complete.')
                # Remove the intermediate temp video *after* successful audio merge
                if os.path.exists(temp_output_video_path):
                    try: os.remove(temp_output_video_path)
                    except OSError as e: print(f"\033[33mWarning: Could not remove intermediate video file {temp_output_video_path}: {e}\033[0m")
                temp_output_video_path = None # Mark as removed
            else:
                update_status('Audio restoration failed. The output video will be silent.', 'ERROR')
                # Audio failed, move the silent video to the final path as a fallback?
                update_status('Moving silent video to final output path as fallback.')
                try:
                    shutil.move(temp_output_video_path, final_output_path)
                    temp_output_video_path = None # Mark as moved
                except Exception as e:
                     update_status(f"Error moving silent video to final output: {e}", "ERROR")
                     # Both audio failed and move failed, temp video might still exist

        else:
            # No audio requested, move the temp video to the final output path
            update_status('Moving temporary video to final output path (no audio).')
            try:
                if os.path.abspath(temp_output_video_path) == os.path.abspath(final_output_path):
                     update_status("Temporary path is the same as final path, no move needed.", "INFO")
                     temp_output_video_path = None # No deletion needed later
                else:
                    # Ensure target directory exists (should already, but double check)
                    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
                    shutil.move(temp_output_video_path, final_output_path)
                    temp_output_video_path = None # Mark as moved successfully
            except Exception as e:
                 update_status(f"Error moving temporary video to final output: {e}", "ERROR")
                 # The temp video might still exist

        # --- Validation ---
        if os.path.exists(final_output_path) and is_video(final_output_path):
            update_status('Processing video finished successfully.')
        else:
             update_status('Processing video failed: Output file not found or invalid after processing.', 'ERROR')

    except Exception as e:
        update_status(f'An unexpected error occurred during video processing: {e}', 'ERROR')
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging

    finally:
        # --- Cleanup ---
        # Clean up temporary frames if they exist and keep_frames is false
        if temp_frame_dir and os.path.exists(temp_frame_dir) and not modules.globals.keep_frames:
             update_status("Cleaning up temporary frames...")
             clean_temp(modules.globals.target_path) # clean_temp uses target_path context to find the dir

        # Clean up intermediate temp video file if it still exists (e.g., audio failed and move failed)
        if temp_output_video_path and os.path.exists(temp_output_video_path):
             try:
                 os.remove(temp_output_video_path)
                 print(f"[DLC.CORE] Removed intermediate temporary video file: {temp_output_video_path}")
             except OSError as e:
                 print(f"\033[33mWarning: Could not remove intermediate temporary video file {temp_output_video_path}: {e}\033[0m")


def destroy(to_quit: bool = True) -> None:
    """Cleans up temporary files, releases resources, and optionally exits."""
    update_status("Cleaning up temporary resources...", "CLEANUP")
    # Use the context of target_path to find the temp directory
    if modules.globals.target_path and not modules.globals.keep_frames:
        clean_temp(modules.globals.target_path)
    release_resources() # Final resource release (GPU cache, GC)
    update_status("Cleanup complete.", "CLEANUP")
    if to_quit:
        print("[DLC.CORE] Exiting application.")
        os._exit(0) # Use os._exit for a more forceful exit if sys.exit hangs (e.g., due to threads)
        # sys.exit(0) # Standard exit


def run() -> None:
    """Parses arguments, sets up the environment, performs checks, and starts processing or UI."""
    try:
        parse_args() # Parse arguments first to set globals like execution_providers, paths, etc.

        # Apply GPU Memory Limit early, requires execution_providers to be set by parse_args
        limit_gpu_memory(GPU_MEMORY_LIMIT_FRACTION)

        # Limit other resources (CPU RAM approximation, TF GPU options)
        # Call this *after* potential PyTorch limit and TensorFlow import check
        limit_resources()

        # Perform pre-checks (dependencies like Python version, ffmpeg, libraries, provider checks)
        update_status("Performing pre-run checks...")
        if not pre_check():
            update_status("Pre-run checks failed. Please see messages above.", "ERROR")
            # destroy(to_quit=True) # Don't call destroy here, let the main try/finally handle it
            return # Exit run() function

        update_status("Pre-run checks passed.")

        # Pre-check frame processors (model downloads, requirements within processors)
        # This needs globals to be set by parse_args and should happen before starting work.
        active_processor_modules = get_frame_processors_modules(modules.globals.frame_processors)
        all_processors_reqs_met = True
        for frame_processor_module in active_processor_modules:
            processor_name = getattr(frame_processor_module, "NAME", "UnknownProcessor")
            update_status(f'Checking requirements for {processor_name}...')
            if hasattr(frame_processor_module, 'pre_check') and callable(frame_processor_module.pre_check):
                if not frame_processor_module.pre_check():
                    update_status(f'Requirements check failed for {processor_name}. See processor messages for details.', 'ERROR')
                    all_processors_reqs_met = False
                    # Don't break early, check all processors to report all issues
            else:
                update_status(f'Processor {processor_name} does not have a pre_check method. Assuming requirements met.', 'WARN')

        if not all_processors_reqs_met:
             update_status('Some frame processors failed requirement checks. Please resolve the issues and retry.', 'ERROR')
             # destroy(to_quit=True) # Let finally handle cleanup
             return

        update_status("All frame processor requirements met.")

        # --- Start processing (headless) or launch UI ---
        if modules.globals.headless:
            # Check for essential paths in headless mode
            if not modules.globals.source_path:
                 update_status("Error: Headless mode requires --source argument.", "ERROR")
                 # program.print_help() # Can't access program object here easily
                 print("Use -h or --help for usage details.")
                 return
            if not modules.globals.target_path:
                 update_status("Error: Headless mode requires --target argument.", "ERROR")
                 print("Use -h or --help for usage details.")
                 return
            if not modules.globals.output_path:
                 update_status("Error: Headless mode requires --output argument.", "ERROR")
                 print("Use -h or --help for usage details.")
                 return

            update_status('Running in headless mode.')
            start() # Execute the main processing logic
            # destroy() will be called by the finally block

        else:
            # --- Launch UI ---
            update_status('Launching graphical user interface...')
            # Ensure destroy is callable without arguments for the UI close button
            destroy_wrapper = lambda: destroy(to_quit=True)
            try:
                # Pass start (processing function) and destroy (cleanup) to the UI
                window = ui.init(start, destroy_wrapper, modules.globals.lang)
                if window:
                    window.mainloop() # Start the UI event loop
                else:
                    update_status("UI initialization failed.", "ERROR")
            except Exception as e:
                 update_status(f"Error initializing or running the UI: {e}", "FATAL")
                 import traceback
                 traceback.print_exc()
                 # Attempt cleanup even if UI fails
                 # destroy(to_quit=True) # Let finally handle it

    except Exception as e:
         # Catch any unexpected errors during setup or execution
         update_status(f"A critical error occurred: {e}", "FATAL")
         import traceback
         traceback.print_exc()

    finally:
         # Ensure cleanup happens regardless of success or failure
         destroy(to_quit=True) # Clean up and exit


# --- Main execution entry point ---
if __name__ == "__main__":
    # This ensures 'run()' is called only when the script is executed directly
    run()

# --- END OF FILE core.py ---