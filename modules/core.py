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

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

def get_system_memory() -> int:
    """
    Get the total system memory in GB.
    
    Returns:
        int: Total system memory in GB.
    """
    if platform.system().lower() == 'darwin':
        try:
            import psutil
            return psutil.virtual_memory().total // (1024 ** 3)
        except ImportError:
            # If psutil is not available, return a default value
            return 16  # Assuming 16GB as a default for macOS
    else:
        # For other systems, we can use psutil if available, or implement system-specific methods
        try:
            import psutil
            return psutil.virtual_memory().total // (1024 ** 3)
        except ImportError:
            # If psutil is not available, return a default value
            return 8  # Assuming 8GB as a default for other systems

def suggest_max_memory() -> int:
    """
    Suggest the maximum memory to use based on the system's total memory.
    
    Returns:
        int: Suggested maximum memory in GB.
    """
    total_memory = get_system_memory()
    # Suggest using 70% of total memory, but not more than 64GB
    suggested_memory = min(int(total_memory * 0.7), 64)
    return max(suggested_memory, 4)  # Ensure at least 4GB is suggested

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
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libvpx-vp9', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=1, choices=range(52), metavar='[0-51]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['coreml'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('--video-processor', help='video processor to use', dest='video_processor', default='cv2', choices=['cv2', 'ffmpeg'])
    program.add_argument('--model', help='model to use for face swapping', dest='model', default='inswapper_128.onnx')
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
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = ['CoreMLExecutionProvider']  # Force CoreML
    modules.globals.execution_threads = args.execution_threads
    modules.globals.video_processor = args.video_processor
    modules.globals.model = args.model

    if 'face_enhancer' in args.frame_processor:
        modules.globals.fp_ui['face_enhancer'] = True
    else:
        modules.globals.fp_ui['face_enhancer'] = False
    
    modules.globals.nsfw = False


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
    if has_image_extension(modules.globals.target_path):
        process_image()
    else:
        process_video()


def process_image():
    if modules.globals.nsfw == False:
        from modules.predicter import predict_image
        if predict_image(modules.globals.target_path):
            destroy()
    shutil.copy2(modules.globals.target_path, modules.globals.output_path)
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status('Progressing...', frame_processor.NAME)
        frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
    if is_image(modules.globals.target_path):
        update_status('Processing to image succeed!')
    else:
        update_status('Processing to image failed!')


def process_video():
    if modules.globals.nsfw == False:
        from modules.predicter import predict_video
        if predict_video(modules.globals.target_path):
            destroy()
    update_status('Creating temp resources...')
    create_temp(modules.globals.target_path)
    update_status('Extracting frames...')
    if modules.globals.video_processor == 'cv2':
        extract_frames_cv2(modules.globals.target_path)
    else:
        extract_frames_ffmpeg(modules.globals.target_path)
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


def extract_frames_cv2(target_path: str) -> None:
    import cv2
    capture = cv2.VideoCapture(target_path)
    frame_num = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        cv2.imwrite(f'{get_temp_frame_paths(target_path)}/%04d.png' % frame_num, frame)
        frame_num += 1
    capture.release()


def extract_frames_ffmpeg(target_path: str) -> None:
    import ffmpeg
    (
        ffmpeg
        .input(target_path)
        .output(f'{get_temp_frame_paths(target_path)}/%04d.png', start_number=0)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )


def destroy() -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    quit()


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
    print(f"Selected execution provider: CoreMLExecutionProvider")
    
    # Configure ONNX Runtime to use only CoreML
    onnxruntime.set_default_logger_severity(3)  # Set to WARNING level
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Test CoreML with a dummy model
    try:
        import numpy as np
        from onnx import helper, TensorProto
        
        # Create a simple ONNX model
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])
        node = helper.make_node('Identity', ['input'], ['output'])
        graph = helper.make_graph([node], 'test_model', [X], [Y])
        model = helper.make_model(graph)
        
        # Save the model
        model_path = 'test_model.onnx'
        with open(model_path, 'wb') as f:
            f.write(model.SerializeToString())
        
        # Create a CoreML session
        session = onnxruntime.InferenceSession(model_path, options, providers=['CoreMLExecutionProvider'])
        
        # Run inference
        input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        output = session.run(None, {'input': input_data})
        
        print("CoreML init successful and being used")
        print(f"Input shape: {input_data.shape}, Output shape: {output[0].shape}")
        
        # Clean up
        os.remove(model_path)
    except Exception as e:
        print(f"Error testing CoreML: {str(e)}")
        print("The application may not be able to use GPU acceleration")
    
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
