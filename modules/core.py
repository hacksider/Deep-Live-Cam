import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow

# modules.globals should be imported first to ensure variables are initialized with defaults
# before any command-line parsing or other logic attempts to modify them.
import modules.globals
import modules.metadata
# import modules.ui as ui # UI import removed
from modules.processors.frame.core import get_frame_processors_modules
# utilities import needs to be after globals for some path normalizations if they were to use globals
from modules.utilities import (
    has_image_extension, is_image, is_video, detect_fps, create_video,
    extract_frames, get_temp_frame_paths, restore_audio, create_temp,
    move_temp, clean_temp, normalize_output_path, get_temp_directory_path # Added get_temp_directory_path
)


if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None: # For CLI use
    # Default values in modules.globals are set when modules.globals is imported.
    # parse_args will overwrite them if CLI arguments are provided.
    signal.signal(signal.SIGINT, lambda signal_number, frame: cleanup_temp_files(quit_app=True)) # Pass quit_app for CLI context
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--nsfw-filter', help='filter the NSFW image or video', dest='nsfw_filter', action='store_true', default=False)
    program.add_argument('--map-faces', help='map source target faces', dest='map_faces', action='store_true', default=False)
    program.add_argument('--mouth-mask', help='mask the mouth region', dest='mouth_mask', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('-l', '--lang', help='Ui language', default="en")
    program.add_argument('--live-mirror', help='The live camera display as you see it in the front-facing camera frame', dest='live_mirror', action='store_true', default=False)
    program.add_argument('--live-resizable', help='The live camera frame is resizable', dest='live_resizable', action='store_true', default=False)
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads
    modules.globals.lang = args.lang

    #for ENHANCER tumbler:
    if 'face_enhancer' in args.frame_processor:
        modules.globals.fp_ui['face_enhancer'] = True
    else:
        modules.globals.fp_ui['face_enhancer'] = False

    # translate deprecated args
    if args.source_path_deprecated:
        print('\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path, args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    # limit memory usage
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = modules.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool: # For CLI and WebApp
    if sys.version_info < (3, 9):
        print('DLC.CORE: Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        print('DLC.CORE: ffmpeg is not installed.')
        return False
    # Potentially add other checks, like if source/target paths are set (for CLI context)
    # For webapp, these will be set by the app itself.
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None: # For CLI and WebApp (prints to console)
    print(f'[{scope}] {message}')
    # UI update removed:
    # if not modules.globals.headless:
    #     ui.update_status(message)

# Renamed from start()
def process_media() -> dict: # Returns a status dictionary
    # Ensure required paths are set in modules.globals
    if not modules.globals.source_path or not os.path.exists(modules.globals.source_path):
        return {'success': False, 'error': 'Source path not set or invalid.'}
    if not modules.globals.target_path or not os.path.exists(modules.globals.target_path):
        return {'success': False, 'error': 'Target path not set or invalid.'}
    if not modules.globals.output_path: # Output path must be determined by caller (e.g. webapp or CLI parse_args)
        return {'success': False, 'error': 'Output path not set.'}

    active_processors = get_frame_processors_modules(modules.globals.frame_processors)
    if not active_processors:
        return {'success': False, 'error': f"No valid frame processors could be initialized for: {modules.globals.frame_processors}. Check if they are installed and configured."}

    for frame_processor in active_processors:
        if hasattr(frame_processor, 'pre_start') and callable(frame_processor.pre_start):
            if not frame_processor.pre_start(): # Some processors might have pre-start checks
                return {'success': False, 'error': f"Pre-start check failed for processor: {frame_processor.NAME if hasattr(frame_processor, 'NAME') else 'Unknown'}"}

    update_status('Processing...')

    # process image to image
    if is_image(modules.globals.target_path): # Use is_image from utilities
        # NSFW Check (temporarily commented out)
        # if modules.globals.nsfw_filter and predict_nsfw(modules.globals.target_path): # Assuming a predict_nsfw utility
        #     return {'success': False, 'error': 'NSFW content detected in target image.', 'nsfw': True}

        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(modules.globals.output_path), exist_ok=True)
            shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        except Exception as e:
            return {'success': False, 'error': f"Error copying target file: {str(e)}"}

        for frame_processor in active_processors:
            update_status(f"Progressing with {frame_processor.NAME if hasattr(frame_processor, 'NAME') else 'Unknown Processor'}")
            try:
                if modules.globals.map_faces and modules.globals.simple_map and hasattr(frame_processor, 'process_image_v2'):
                    # For mapped faces, process_image_v2 might only need the target and output paths,
                    # as mappings are in Globals.simple_map.
                    # The specific signature depends on processor implementation.
                    # Assuming (target_path, output_path) for v2 for now.
                    frame_processor.process_image_v2(modules.globals.output_path, modules.globals.output_path)
                elif hasattr(frame_processor, 'process_image'):
                    # Standard processing if not map_faces or if processor lacks v2
                    frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
                else:
                    update_status(f"Processor {frame_processor.NAME} has no suitable process_image or process_image_v2 method.")
                    # Decide if this should be an error or just a skip
                release_resources()
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {'success': False, 'error': f"Error during image processing with {frame_processor.NAME if hasattr(frame_processor, 'NAME') else 'Unknown Processor'}: {str(e)}"}

        if os.path.exists(modules.globals.output_path): # Check if output file was actually created
            update_status('Processing to image succeed!')
            return {'success': True, 'output_path': modules.globals.output_path}
        else:
            update_status('Processing to image failed! Output file not found.')
            return {'success': False, 'error': 'Output image file not found after processing.'}

    # process video
    if is_video(modules.globals.target_path): # Use is_video from utilities
        # NSFW Check (temporarily commented out)
        # if modules.globals.nsfw_filter and predict_nsfw(modules.globals.target_path): # Assuming a predict_nsfw utility
        #     return {'success': False, 'error': 'NSFW content detected in target video.', 'nsfw': True}

        update_status('Creating temp resources...')
        # temp_frames_dir should be based on the target_path filename to ensure uniqueness
        temp_frames_dir = get_temp_directory_path(modules.globals.target_path)
        create_temp(temp_frames_dir) # Create the specific directory for frames

        update_status('Extracting frames...')
        extract_frames(modules.globals.target_path, temp_frames_dir) # Pass explicit temp_frames_dir

        processed_temp_frame_paths = get_temp_frame_paths(temp_frames_dir) # Get paths from the correct temp dir
        if not processed_temp_frame_paths:
            clean_temp(temp_frames_dir)
            return {'success': False, 'error': 'Failed to extract frames from video.'}

        for frame_processor in active_processors:
            update_status(f"Progressing with {frame_processor.NAME if hasattr(frame_processor, 'NAME') else 'Unknown Processor'}")
            try:
                if modules.globals.map_faces and modules.globals.simple_map and hasattr(frame_processor, 'process_video_v2'):
                    # For mapped faces, process_video_v2 might only need the frame paths,
                    # as mappings are in Globals.simple_map.
                    # The specific signature depends on processor implementation.
                    # Assuming (list_of_frame_paths) for v2 for now.
                    frame_processor.process_video_v2(processed_temp_frame_paths)
                elif hasattr(frame_processor, 'process_video'):
                     # Standard processing if not map_faces or if processor lacks v2
                    frame_processor.process_video(modules.globals.source_path, processed_temp_frame_paths)
                else:
                    update_status(f"Processor {frame_processor.NAME} has no suitable process_video or process_video_v2 method.")
                     # Decide if this should be an error or just a skip
                release_resources()
            except Exception as e:
                import traceback
                traceback.print_exc()
                clean_temp(temp_frames_dir)
                return {'success': False, 'error': f"Error during video processing with {frame_processor.NAME if hasattr(frame_processor, 'NAME') else 'Unknown Processor'}: {str(e)}"}

        video_fps = detect_fps(modules.globals.target_path) if modules.globals.keep_fps else 30.0
        update_status(f'Creating video with {video_fps} fps...')

        # Temp video output path for video without audio
        # output_path is the final destination, temp_video_output_path is intermediate
        temp_video_output_path = normalize_output_path(modules.globals.target_path, os.path.dirname(modules.globals.output_path), '_temp_novideoaudio')
        if not temp_video_output_path:
            clean_temp(temp_frames_dir)
            return {'success': False, 'error': 'Could not normalize temporary video output path.'}

        frames_pattern = os.path.join(temp_frames_dir, "%04d.png")
        if not create_video(frames_pattern, video_fps, temp_video_output_path, modules.globals.video_quality, modules.globals.video_encoder):
            clean_temp(temp_frames_dir)
            if os.path.exists(temp_video_output_path): os.remove(temp_video_output_path)
            return {'success': False, 'error': 'Failed to create video from processed frames.'}

        if modules.globals.keep_audio:
            update_status('Restoring audio...')
            if not restore_audio(temp_video_output_path, modules.globals.target_path, modules.globals.output_path):
                update_status('Audio restoration failed. Moving video without new audio to output.')
                shutil.move(temp_video_output_path, modules.globals.output_path) # Fallback: move the no-audio video
            else: # Audio restored, temp_video_output_path was used as source, now remove it if it still exists
                 if os.path.exists(temp_video_output_path) and temp_video_output_path != modules.globals.output_path :
                     os.remove(temp_video_output_path)
        else:
            shutil.move(temp_video_output_path, modules.globals.output_path)

        clean_temp(temp_frames_dir)

        if os.path.exists(modules.globals.output_path):
            update_status('Processing to video succeed!')
            return {'success': True, 'output_path': modules.globals.output_path}
        else:
            update_status('Processing to video failed! Output file not found.')
            return {'success': False, 'error': 'Output video file not found after processing.'}

    return {'success': False, 'error': 'Target file type not supported (not image or video).'}


# Renamed from destroy()
def cleanup_temp_files(quit_app: bool = False) -> None: # quit_app is for CLI context
    if modules.globals.target_path: # Check if target_path was ever set
        temp_frames_dir = get_temp_directory_path(modules.globals.target_path)
        if os.path.exists(temp_frames_dir): # Check if temp_frames_dir exists before cleaning
             clean_temp(temp_frames_dir)
    if quit_app:
        sys.exit() # Use sys.exit for a cleaner exit than quit()


def run() -> None: # CLI focused run
    parse_args() # Sets globals from CLI args
    if not pre_check():
        cleanup_temp_files(quit_app=True)
        return

    # Initialize processors and check their specific pre-requisites
    # This was implicitly part of the old start() before iterating
    active_processors = get_frame_processors_modules(modules.globals.frame_processors)
    if not active_processors:
        update_status(f"Failed to initialize frame processors: {modules.globals.frame_processors}. Exiting.")
        cleanup_temp_files(quit_app=True)
        return

    all_processors_ready = True
    for frame_processor in active_processors:
        if hasattr(frame_processor, 'pre_check') and callable(frame_processor.pre_check):
            if not frame_processor.pre_check():
                all_processors_ready = False
                # Processor should print its own error message via update_status or print
                break
    if not all_processors_ready:
        cleanup_temp_files(quit_app=True)
        return

    limit_resources()

    # modules.globals.headless is set by parse_args if CLI args are present
    # This run() is now CLI-only, so headless is effectively always true in this context
    if modules.globals.headless:
        processing_result = process_media()
        if processing_result['success']:
            update_status(f"CLI processing finished successfully. Output: {processing_result.get('output_path', 'N/A')}")
        else:
            update_status(f"CLI processing failed: {processing_result.get('error', 'Unknown error')}")
            if processing_result.get('nsfw'):
                update_status("NSFW content was detected and processing was halted.")
    else:
        # This block should ideally not be reached if parse_args correctly sets headless
        # or if run() is only called in a CLI context.
        # For safety, we can print a message.
        update_status("Warning: core.run() called in a mode that seems non-headless, but UI is disabled. Processing will not start.")

    cleanup_temp_files(quit_app=True) # Cleanup and exit for CLI
