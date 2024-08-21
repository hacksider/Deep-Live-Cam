import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Force TensorFlow to use Metal
os.environ['TENSORFLOW_METAL'] = '1'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow
import cv2

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=True)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=True)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--nsfw-filter', help='filter the NSFW image or video', dest='nsfw_filter', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('--live-mirror', help='The live camera display as you see it in the front-facing camera frame', dest='live_mirror', action='store_true', default=False)
    program.add_argument('--live-resizable', help='The live camera frame is resizable', dest='live_resizable', action='store_true', default=False)
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['coreml'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

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
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = ['CoreMLExecutionProvider']  # Force CoreML
    modules.globals.execution_threads = args.execution_threads

    if 'face_enhancer' in args.frame_processor:
        modules.globals.fp_ui['face_enhancer'] = True
    else:
        modules.globals.fp_ui['face_enhancer'] = False


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 6
    return 4


def suggest_execution_providers() -> List[str]:
    return ['coreml']  # Only suggest CoreML


def suggest_execution_threads() -> int:
    if platform.system().lower() == 'darwin':
        return 12
    return 4
    


def limit_resources() -> None:
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 6
        import resource
        resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    pass  # No need to release CUDA resources


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')
    if not modules.globals.headless:
        ui.update_status(message)

def start() -> None:
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    # process image to image
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw == False:
            from modules.predicter import predict_image
            if predict_image(modules.globals.target_path):
                destroy()
        shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
            release_resources()
        if is_image(modules.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    # process image to videos
    if modules.globals.nsfw == False:
        from modules.predicter import predict_video
        if predict_video(modules.globals.target_path):
            destroy()
    update_status('Creating temp resources...')
    create_temp(modules.globals.target_path)
    update_status('Extracting frames...')
    extract_frames(modules.globals.target_path)
    temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status('Progressing...', frame_processor.NAME)
        frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
    if modules.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(modules.globals.target_path)
        update_status(f'Creating video with {fps} fps...')
        create_video(modules.globals.target_path, fps)
    else:
        update_status('Creating video with 30.0 fps...')
        create_video(modules.globals.target_path)
    if modules.globals.keep_audio:
        if modules.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)
    clean_temp(modules.globals.target_path)
    if is_video(modules.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    if to_quit: quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    print(f"ONNX Runtime version: {onnxruntime.__version__}")
    print(f"Available execution providers: {onnxruntime.get_available_providers()}")
    print(f"Selected execution provider: CoreMLExecutionProvider (with CPU fallback for face detection)")
    
    # Configure ONNX Runtime to use CoreML
    onnxruntime.set_default_logger_severity(3)  # Set to WARNING level
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Add CoreML-specific options
    options.add_session_config_entry("session.coreml.force_precision", "FP32")
    options.add_session_config_entry("session.coreml.enable_on_subgraph", "1")
    
    # Update insightface model loading to use CPU for face detection
    from insightface.utils import face_align
    def custom_session(model_file, providers):
        if 'det_model.onnx' in model_file:
            return onnxruntime.InferenceSession(model_file, providers=['CPUExecutionProvider'])
        else:
            return onnxruntime.InferenceSession(model_file, options, providers=['CoreMLExecutionProvider'])
    face_align.Session = custom_session
    
    # Configure TensorFlow to use Metal
    try:
        tf_devices = tensorflow.config.list_physical_devices()
        print("TensorFlow devices:", tf_devices)
        if any('GPU' in device.name for device in tf_devices):
            print("TensorFlow is using GPU (Metal)")
        else:
            print("TensorFlow is not using GPU")
    except Exception as e:
        print(f"Error configuring TensorFlow: {str(e)}")
    
    # Configure PyTorch to use MPS (Metal Performance Shaders)
    try:
        if torch.backends.mps.is_available():
            print("PyTorch is using MPS (Metal Performance Shaders)")
            torch.set_default_device('mps')
        else:
            print("PyTorch MPS is not available")
    except Exception as e:
        print(f"Error configuring PyTorch: {str(e)}")
    
    if modules.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()

def get_one_face(frame):
    # Resize the frame to the expected input size
    frame_resized = cv2.resize(frame, (112, 112))  # Resize to (112, 112) for recognition model
    face = get_face_analyser().get(frame_resized)
    return face

# Ensure to use the CPUExecutionProvider if CoreML fails
def run_model_with_cpu_fallback(model_file, providers):
    try:
        return onnxruntime.InferenceSession(model_file, providers=['CoreMLExecutionProvider'])
    except Exception as e:
        print(f"CoreML execution failed: {e}. Falling back to CPU.")
        return onnxruntime.InferenceSession(model_file, providers=['CPUExecutionProvider'])

# Update the face analysis function to use the fallback
def get_face_analyser():
    # Load your model here with the fallback
    return run_model_with_cpu_fallback('/path/to/your/model.onnx', ['CoreMLExecutionProvider', 'CPUExecutionProvider'])