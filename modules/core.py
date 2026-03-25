import os
import sys

# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith("--execution-provider") for arg in sys.argv):
    os.environ["OMP_NUM_THREADS"] = "1"
# avoid unstable OpenCV OBSENSOR probing on some macOS setups
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")
# reduce tensorflow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
from typing import List, Optional
import platform
import signal
import shutil
import argparse
import importlib.metadata
import threading

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import onnxruntime

try:
    import tensorflow
except Exception:
    tensorflow = None

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
    detect_fps,
    create_video,
    extract_frames,
    get_temp_frame_paths,
    restore_audio,
    create_temp,
    move_temp,
    clean_temp,
    normalize_output_path,
)

if HAS_TORCH and "ROCMExecutionProvider" in modules.globals.execution_providers:
    del torch

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
if HAS_TORCH:
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument(
        "-s", "--source", help="select an source image", dest="source_path"
    )
    program.add_argument(
        "-t", "--target", help="select an target image or video", dest="target_path"
    )
    program.add_argument(
        "-o", "--output", help="select output file or directory", dest="output_path"
    )
    program.add_argument(
        "--frame-processor",
        help="pipeline of frame processors",
        dest="frame_processor",
        default=["face_swapper"],
        choices=[
            "face_swapper",
            "face_enhancer",
            "face_enhancer_gpen256",
            "face_enhancer_gpen512",
        ],
        nargs="+",
    )
    program.add_argument(
        "--keep-fps",
        help="keep original fps",
        dest="keep_fps",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--keep-audio",
        help="keep original audio",
        dest="keep_audio",
        action="store_true",
        default=True,
    )
    program.add_argument(
        "--keep-frames",
        help="keep temporary frames",
        dest="keep_frames",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--many-faces",
        help="process every face",
        dest="many_faces",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--nsfw-filter",
        help="filter the NSFW image or video",
        dest="nsfw_filter",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--map-faces",
        help="map source target faces",
        dest="map_faces",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--mouth-mask",
        help="mask the mouth region",
        dest="mouth_mask",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--video-encoder",
        help="adjust output video encoder",
        dest="video_encoder",
        default="libx264",
        choices=["libx264", "libx265", "libvpx-vp9"],
    )
    program.add_argument(
        "--video-quality",
        help="adjust output video quality",
        dest="video_quality",
        type=int,
        default=18,
        choices=range(52),
        metavar="[0-51]",
    )
    program.add_argument("-l", "--lang", help="Ui language", default="en")
    program.add_argument(
        "--live-mirror",
        help="The live camera display as you see it in the front-facing camera frame",
        dest="live_mirror",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--live-resizable",
        help="The live camera frame is resizable",
        dest="live_resizable",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--max-memory",
        help="maximum amount of RAM in GB",
        dest="max_memory",
        type=int,
        default=suggest_max_memory(),
    )
    program.add_argument(
        "--execution-provider",
        help="execution provider",
        dest="execution_provider",
        default=suggest_default_execution_providers(),
        choices=suggest_execution_providers(),
        nargs="+",
    )
    program.add_argument(
        "--execution-threads",
        help="number of execution threads",
        dest="execution_threads",
        type=int,
    )
    program.add_argument(
        "--face-analyser-engine",
        help="face analyser backend",
        dest="face_analyser_engine",
        default=modules.globals.face_analyser_engine,
        choices=["insightface", "mlx_uniface"],
    )
    program.add_argument(
        "--mlx-face-detector",
        help="detector model when MLX backend is enabled",
        dest="mlx_face_detector",
        default=modules.globals.mlx_face_detector,
        choices=["retinaface"],
    )
    program.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"{modules.metadata.name} {modules.metadata.version}",
    )

    # register deprecated args
    program.add_argument(
        "-f", "--face", help=argparse.SUPPRESS, dest="source_path_deprecated"
    )
    program.add_argument(
        "--cpu-cores", help=argparse.SUPPRESS, dest="cpu_cores_deprecated", type=int
    )
    program.add_argument(
        "--gpu-vendor", help=argparse.SUPPRESS, dest="gpu_vendor_deprecated"
    )
    program.add_argument(
        "--gpu-threads", help=argparse.SUPPRESS, dest="gpu_threads_deprecated", type=int
    )

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(
        modules.globals.source_path, modules.globals.target_path, args.output_path
    )
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = bool(
        args.source_path or args.target_path or args.output_path
    )
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
    modules.globals.execution_providers = decode_execution_providers(
        args.execution_provider
    )
    modules.globals.execution_threads = (
        args.execution_threads
        or suggest_execution_threads(modules.globals.execution_providers)
    )
    modules.globals.face_analyser_engine = args.face_analyser_engine
    modules.globals.mlx_face_detector = args.mlx_face_detector
    modules.globals.lang = args.lang

    # for ENHANCER tumblers:
    for enhancer_key in (
        "face_enhancer",
        "face_enhancer_gpen256",
        "face_enhancer_gpen512",
    ):
        modules.globals.fp_ui[enhancer_key] = enhancer_key in args.frame_processor

    # translate deprecated args
    if args.source_path_deprecated:
        print(
            "\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m"
        )
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(
            args.source_path_deprecated, modules.globals.target_path, args.output_path
        )
    if args.cpu_cores_deprecated:
        print(
            "\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m"
        )
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == "apple":
        print(
            "\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m"
        )
        modules.globals.execution_providers = decode_execution_providers(["coreml"])
    if args.gpu_vendor_deprecated == "nvidia":
        print(
            "\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m"
        )
        modules.globals.execution_providers = decode_execution_providers(["cuda"])
    if args.gpu_vendor_deprecated == "amd":
        print(
            "\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m"
        )
        modules.globals.execution_providers = decode_execution_providers(["rocm"])
    if args.gpu_threads_deprecated:
        print(
            "\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m"
        )
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [
        execution_provider.replace("ExecutionProvider", "").lower()
        for execution_provider in execution_providers
    ]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    available = onnxruntime.get_available_providers()
    decoded = [
        provider
        for provider, encoded_execution_provider in zip(
            available, encode_execution_providers(available)
        )
        if any(
            execution_provider in encoded_execution_provider
            for execution_provider in execution_providers
        )
    ]
    if (
        decoded
        and "CPUExecutionProvider" in available
        and "CPUExecutionProvider" not in decoded
    ):
        decoded.append("CPUExecutionProvider")
    if decoded:
        return decoded
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return available[:1]


def suggest_max_memory() -> int:
    if platform.system().lower() == "darwin":
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    supported = {
        "cpu",
        "cuda",
        "rocm",
        "directml",
        "dml",
        "openvino",
        "coreml",
        "tensorrt",
    }
    return [
        provider
        for provider in encode_execution_providers(
            onnxruntime.get_available_providers()
        )
        if provider in supported
    ]


def suggest_default_execution_providers() -> List[str]:
    available = suggest_execution_providers()
    if platform.system().lower() == "darwin" and "coreml" in available:
        return ["coreml"]
    if "cpu" in available:
        return ["cpu"]
    return available[:1] if available else ["cpu"]


def suggest_execution_threads(execution_providers: Optional[List[str]] = None) -> int:
    """Suggest optimal thread count based on hardware and execution provider."""
    cpu_count = os.cpu_count() or 4
    providers = (
        execution_providers
        if execution_providers is not None
        else modules.globals.execution_providers
    )

    if "DmlExecutionProvider" in providers:
        return 1
    if "ROCMExecutionProvider" in providers:
        return 1
    if "CUDAExecutionProvider" in providers:
        return min(cpu_count, 16)
    if "CoreMLExecutionProvider" in providers:
        if platform.system().lower() == "darwin" and platform.machine() == "arm64":
            return max(2, min(cpu_count // 2, 4))
        return max(2, min(cpu_count // 2, 8))

    if platform.system().lower() == "darwin" and platform.machine() == "arm64":
        return max(2, min(cpu_count - 2, 6))

    return max(2, min(cpu_count - 2, 16))


def limit_resources() -> None:
    # prevent tensorflow memory leak
    if tensorflow is not None:
        try:
            gpus = tensorflow.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tensorflow.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    # limit memory usage
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024**3
        if platform.system().lower() == "windows":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(
                -1, ctypes.c_size_t(memory), ctypes.c_size_t(memory)
            )
        else:
            import resource

            # macOS is more reliable with RLIMIT_AS than RLIMIT_DATA here.
            limit_kind = (
                resource.RLIMIT_AS
                if platform.system().lower() == "darwin"
                else resource.RLIMIT_DATA
            )
            try:
                resource.setrlimit(limit_kind, (memory, memory))
            except Exception:
                pass


def release_resources() -> None:
    if "CUDAExecutionProvider" in modules.globals.execution_providers and HAS_TORCH:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status(
            "Python version is not supported - please upgrade to 3.9 or higher."
        )
        return False
    if platform.system().lower() == "darwin":
        # Warn for known unstable dependency combinations on macOS.
        try:
            numpy_version = importlib.metadata.version("numpy")
            if int(numpy_version.split(".")[0]) >= 2:
                update_status(
                    'Warning: NumPy 2.x detected on macOS. For best stability use "numpy<2".'
                )
        except Exception:
            pass
        try:
            importlib.metadata.version("opencv-python-headless")
            update_status(
                "Warning: opencv-python-headless is installed. Webcam may fail; uninstall it on macOS."
            )
        except Exception:
            pass
    if not shutil.which("ffmpeg"):
        if modules.globals.headless:
            update_status("ffmpeg is not installed.")
            return False
        update_status(
            "Warning: ffmpeg is not installed. Live mode will work, but video export will fail."
        )
    return True


def update_status(message: str, scope: str = "DLC.CORE") -> None:
    print(f"[{scope}] {message}")
    if modules.globals.headless or getattr(ui, "status_label", None) is None:
        return
    try:
        if threading.current_thread() is threading.main_thread():
            ui.update_status(message)
    except Exception:
        # UI status updates are best-effort only; never break processing threads.
        pass


def start() -> None:
    """Start processing with performance monitoring."""
    import time

    start_time = time.time()

    for frame_processor in get_frame_processors_modules(
        modules.globals.frame_processors
    ):
        if not frame_processor.pre_start():
            return
    update_status("Processing...")

    # process image to image
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(
            modules.globals.target_path, destroy
        ):
            return
        try:
            shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        except Exception as e:
            print("Error copying file:", str(e))
        for frame_processor in get_frame_processors_modules(
            modules.globals.frame_processors
        ):
            update_status("Progressing...", frame_processor.NAME)
            frame_processor.process_image(
                modules.globals.source_path,
                modules.globals.output_path,
                modules.globals.output_path,
            )
            release_resources()
        if is_image(modules.globals.target_path):
            elapsed = time.time() - start_time
            update_status(f"Processing to image succeed! (Time: {elapsed:.2f}s)")
        else:
            update_status("Processing to image failed!")
        return

    # process image to videos
    if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(
        modules.globals.target_path, destroy
    ):
        return

    extraction_start = time.time()
    if not modules.globals.map_faces:
        update_status("Creating temp resources...")
        create_temp(modules.globals.target_path)
        update_status("Extracting frames...")
        extract_frames(modules.globals.target_path)
    extraction_time = time.time() - extraction_start
    update_status(f"Frame extraction completed in {extraction_time:.2f}s")

    temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
    total_frames = len(temp_frame_paths)
    update_status(
        f"Processing {total_frames} frames with {modules.globals.execution_threads} threads..."
    )

    processing_start = time.time()
    for frame_processor in get_frame_processors_modules(
        modules.globals.frame_processors
    ):
        update_status("Progressing...", frame_processor.NAME)
        frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
        release_resources()
    processing_time = time.time() - processing_start
    fps_processing = total_frames / processing_time if processing_time > 0 else 0
    update_status(
        f"Frame processing completed in {processing_time:.2f}s ({fps_processing:.2f} fps)"
    )

    # handles fps
    encoding_start = time.time()
    if modules.globals.keep_fps:
        update_status("Detecting fps...")
        fps = detect_fps(modules.globals.target_path)
        update_status(f"Creating video with {fps} fps...")
        create_video(modules.globals.target_path, fps)
    else:
        update_status("Creating video with 30.0 fps...")
        create_video(modules.globals.target_path)
    encoding_time = time.time() - encoding_start
    update_status(f"Video encoding completed in {encoding_time:.2f}s")

    # handle audio
    if modules.globals.keep_audio:
        if modules.globals.keep_fps:
            update_status("Restoring audio...")
        else:
            update_status("Restoring audio might cause issues as fps are not kept...")
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)

    # clean and validate
    clean_temp(modules.globals.target_path)

    total_time = time.time() - start_time
    if is_video(modules.globals.target_path):
        update_status(f"Processing to video succeed! Total time: {total_time:.2f}s")
    else:
        update_status("Processing to video failed!")


def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    if to_quit:
        quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(
        modules.globals.frame_processors
    ):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if modules.globals.headless:
        start()
    else:
        window = ui.init(start, destroy, modules.globals.lang)
        window.mainloop()
