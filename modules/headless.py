"""Programmatic, display-free entry point for Deep-Live-Cam.

This is what cloud GPU deployments (Modal, RunPod, Docker, ...) call instead
of the Qt UI or the argparse CLI. It sets the same globals the CLI sets,
runs the same pre-checks, and then drives ``modules.core.start``.

Example::

    from modules.headless import process
    process("face.jpg", "clip.mp4", "out.mp4", frame_processors=["face_swapper", "face_enhancer"])

It can also be run as a script, mirroring the CLI flags that matter off-desktop::

    python -m modules.headless --download-models
    python -m modules.headless -s face.jpg -t clip.mp4 -o out.mp4 --frame-processor face_swapper face_enhancer
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional

FRAME_PROCESSORS = ("face_swapper", "face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512")
ENHANCERS = ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512")
VIDEO_ENCODERS = ("libx264", "libx265", "libvpx-vp9")
DET_SIZES = (160, 320, 640)


def preload_cuda_libraries() -> None:
    """Make pip-installed CUDA/cuDNN wheels visible to onnxruntime.

    ``run.py`` does this for the desktop app; programmatic callers skip
    ``run.py``, so do it here. A no-op when the libraries come from the
    system (e.g. an ``nvidia/cuda`` base image) or no GPU build is installed.
    """
    try:
        import onnxruntime
        preload = getattr(onnxruntime, "preload_dlls", None)
        if preload is not None:
            preload()
    except Exception as error:  # pragma: no cover - best effort only
        print(f"[DLC.HEADLESS] CUDA library preload skipped: {error}", file=sys.stderr)


def download_models(frame_processors: Iterable[str] = FRAME_PROCESSORS) -> None:
    """Fetch every model the given processors need (plus insightface's).

    Run it once at image-build time or against a persistent volume so that
    cold starts don't spend minutes downloading ~2 GB of weights.
    """
    import modules.globals
    from modules.model_downloader import ensure_insightface_pack
    from modules.processors.frame.core import load_frame_processor_module

    modules.globals.headless = True
    if not ensure_insightface_pack("buffalo_l"):
        raise RuntimeError("Could not download the insightface buffalo_l models")
    for name in frame_processors:
        if not load_frame_processor_module(name).pre_check():
            raise RuntimeError(f"Could not download the models for {name}")


def _reset_per_job_state(previous_det_size: int) -> None:
    """Clear state a warm container would otherwise carry between jobs.

    Loaded models stay cached (that is the point of a warm container); only
    per-job selections and inter-frame state are reset.
    """
    import modules.globals
    from modules.processors.frame import core as frame_core

    # The processor list is cached on first use; rebuild it per job so a
    # job without face_swapper doesn't inherit it from the previous one.
    frame_core.FRAME_PROCESSORS_MODULES.clear()

    # Temporal smoothing keeps the previous frame between calls; don't blend
    # one job's last frame into the next.
    swapper = sys.modules.get("modules.processors.frame.face_swapper")
    if swapper is not None:
        swapper.PREVIOUS_FRAME_RESULT = None

    # insightface fixes the detector input shape at prepare() time.
    if modules.globals.det_size != previous_det_size:
        from modules.face_analyser import reset_face_analyser
        reset_face_analyser()


def process(
    source_path: str,
    target_path: str,
    output_path: str,
    *,
    frame_processors: Iterable[str] = ("face_swapper",),
    many_faces: bool = False,
    mouth_mask: bool = False,
    keep_fps: bool = True,
    keep_audio: bool = True,
    keep_frames: bool = False,
    nsfw_filter: bool = False,
    video_encoder: str = "libx264",
    video_quality: int = 18,
    det_size: Optional[int] = None,
    execution_provider: Optional[str] = None,
    execution_threads: Optional[int] = None,
    max_memory: Optional[int] = None,
) -> str:
    """Swap the face from ``source_path`` onto ``target_path`` (image or video).

    Returns ``output_path`` on success and raises ``RuntimeError`` otherwise.
    ``max_memory`` (GB) is off by default: cloud containers enforce their own
    limits and a process-wide RLIMIT_DATA can starve the CUDA runtime.
    """
    frame_processors = list(frame_processors)
    if not frame_processors:
        raise ValueError(f"At least one frame processor is required: {FRAME_PROCESSORS}")
    unknown = [name for name in frame_processors if name not in FRAME_PROCESSORS]
    if unknown:
        raise ValueError(f"Unknown frame processor(s): {unknown}. Choose from {FRAME_PROCESSORS}.")
    if video_encoder not in VIDEO_ENCODERS:
        raise ValueError(f"video_encoder must be one of {VIDEO_ENCODERS}")
    if not 0 <= video_quality <= 51:
        raise ValueError("video_quality must be between 0 and 51")
    if det_size is not None and det_size not in DET_SIZES:
        raise ValueError(f"det_size must be one of {DET_SIZES}")
    for label, path in (("source", source_path), ("target", target_path)):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    preload_cuda_libraries()

    import modules.globals
    from modules import core
    from modules.processors.frame.core import get_frame_processors_modules

    g = modules.globals
    previous_det_size = g.det_size
    g.headless = True
    g.source_path = source_path
    g.target_path = target_path
    g.output_path = output_path
    g.frame_processors = frame_processors
    g.many_faces = many_faces
    g.mouth_mask = mouth_mask
    g.map_faces = False
    g.keep_fps = keep_fps
    g.keep_audio = keep_audio
    g.keep_frames = keep_frames
    g.nsfw_filter = nsfw_filter
    g.video_encoder = video_encoder
    g.video_quality = video_quality
    g.det_size = det_size or g.DEFAULT_DET_SIZE
    g.max_memory = max_memory
    g.execution_providers = core.decode_execution_providers(
        [execution_provider or core.suggest_default_execution_provider()]
    )
    if not g.execution_providers:
        print(f"[DLC.HEADLESS] Execution provider {execution_provider!r} unavailable; using CPU.")
        g.execution_providers = ["CPUExecutionProvider"]
    g.execution_threads = execution_threads or core.suggest_execution_threads()
    for enhancer in ENHANCERS:
        g.fp_ui[enhancer] = enhancer in frame_processors

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    _reset_per_job_state(previous_det_size)

    if not core.pre_check():
        raise RuntimeError("Pre-check failed (is ffmpeg installed and on PATH?)")
    for frame_processor in get_frame_processors_modules(frame_processors):
        if not frame_processor.pre_check():
            raise RuntimeError(f"{frame_processor.NAME} pre-check failed (model download?)")
    core.limit_resources()
    core.start()

    if not os.path.isfile(output_path):
        raise RuntimeError("Processing finished without producing an output file; see log above.")
    return output_path


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Deep-Live-Cam headless runner")
    parser.add_argument("--download-models", action="store_true", help="only fetch models, then exit")
    parser.add_argument("-s", "--source", dest="source_path")
    parser.add_argument("-t", "--target", dest="target_path")
    parser.add_argument("-o", "--output", dest="output_path")
    parser.add_argument("--frame-processor", dest="frame_processors", nargs="+",
                        default=["face_swapper"], choices=FRAME_PROCESSORS)
    parser.add_argument("--many-faces", action="store_true")
    parser.add_argument("--mouth-mask", action="store_true")
    parser.add_argument("--no-keep-fps", dest="keep_fps", action="store_false")
    parser.add_argument("--no-keep-audio", dest="keep_audio", action="store_false")
    parser.add_argument("--video-encoder", default="libx264", choices=VIDEO_ENCODERS)
    parser.add_argument("--video-quality", type=int, default=18)
    parser.add_argument("--det-size", type=int, default=None, choices=DET_SIZES)
    parser.add_argument("--execution-provider", default=None, help="cuda, cpu, ... (default: best available)")
    args = parser.parse_args(argv)

    if args.download_models:
        preload_cuda_libraries()
        download_models(args.frame_processors if "--frame-processor" in (argv or sys.argv) else FRAME_PROCESSORS)
        return 0
    if not (args.source_path and args.target_path and args.output_path):
        parser.error("-s, -t and -o are required unless --download-models is given")
    process(
        args.source_path,
        args.target_path,
        args.output_path,
        frame_processors=args.frame_processors,
        many_faces=args.many_faces,
        mouth_mask=args.mouth_mask,
        keep_fps=args.keep_fps,
        keep_audio=args.keep_audio,
        video_encoder=args.video_encoder,
        video_quality=args.video_quality,
        det_size=args.det_size,
        execution_provider=args.execution_provider,
    )
    print(f"[DLC.HEADLESS] Wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
