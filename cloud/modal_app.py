"""Run Deep-Live-Cam on Modal (https://modal.com) GPUs.

One-time setup (on your own machine):

    pip install modal && modal setup
    modal run cloud/modal_app.py::download_models      # fill the model volume once

Swap a face from the command line (files are uploaded, the result is saved locally):

    modal run cloud/modal_app.py --source face.jpg --target clip.mp4
    modal run cloud/modal_app.py --source face.jpg --target clip.mp4 \\
        --output out.mp4 --processors face_swapper,face_enhancer --many-faces

Deploy a persistent HTTPS API (same routes as cloud/server.py):

    modal deploy cloud/modal_app.py
    curl -F source=@face.jpg -F target=@clip.mp4 https://<workspace>--deep-live-cam-deeplivecam-web.modal.run/swap -o out.mp4

Environment knobs (read when the app is defined):
    DLC_MODAL_GPU     GPU type, default "L4" (e.g. T4, A10G, L40S, A100, H100)
    DLC_MODAL_VOLUME  model volume name, default "deep-live-cam-models"
"""

from __future__ import annotations

import os
import pathlib

import modal

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/app"
MODELS_MOUNT = "/models"

GPU = os.environ.get("DLC_MODAL_GPU", "L4")
VOLUME_NAME = os.environ.get("DLC_MODAL_VOLUME", "deep-live-cam-models")

app = modal.App("deep-live-cam")
models_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    # CUDA 12 + cuDNN 9 are what onnxruntime-gpu's CUDA provider links against.
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("ffmpeg", "build-essential", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements-cloud.txt"))
    .env(
        {
            "DLC_MODELS_DIR": f"{MODELS_MOUNT}/dlc",
            "DLC_INSIGHTFACE_ROOT": f"{MODELS_MOUNT}/insightface",
            "PYTHONPATH": REMOTE_ROOT,
            "PYTHONUNBUFFERED": "1",
        }
    )
    # Source last so code edits don't rebuild the dependency layers.
    .add_local_dir(
        REPO_ROOT,
        REMOTE_ROOT,
        ignore=[
            ".git", ".git/**", "**/__pycache__/**", "venv/**", ".venv/**",
            "models/**", "media/**", "temp/**", "**/*.onnx", "**/*.pth",
            "**/*.mp4", "**/*.mkv",
        ],
    )
)


@app.function(image=image, volumes={MODELS_MOUNT: models_volume}, timeout=60 * 30)
def download_models() -> None:
    """Populate the model volume. Runs on CPU; only needed once."""
    os.chdir(REMOTE_ROOT)
    from modules.headless import download_models as fetch

    fetch()
    models_volume.commit()
    print("Models are in the volume; GPU containers will start without downloading.")


@app.cls(
    image=image,
    gpu=GPU,
    volumes={MODELS_MOUNT: models_volume},
    timeout=60 * 60,
    scaledown_window=5 * 60,
)
class DeepLiveCam:
    @modal.enter()
    def setup(self) -> None:
        os.chdir(REMOTE_ROOT)
        from modules.headless import download_models as fetch, preload_cuda_libraries

        preload_cuda_libraries()
        # A no-op when the volume is already populated; otherwise the first
        # container downloads and persists the models for the next ones.
        fetch(["face_swapper"])
        models_volume.commit()

        import onnxruntime

        providers = onnxruntime.get_available_providers()
        print(f"[DLC.MODAL] onnxruntime providers: {providers}")
        if "CUDAExecutionProvider" not in providers:
            print("[DLC.MODAL] WARNING: CUDA provider missing, falling back to CPU.")

    @modal.method()
    def swap(
        self,
        source_bytes: bytes,
        source_name: str,
        target_bytes: bytes,
        target_name: str,
        **options,
    ) -> bytes:
        """Run one job and return the output file's bytes."""
        import tempfile

        from cloud.server import run_job

        with tempfile.TemporaryDirectory(prefix="dlc-") as workdir:
            source_file = os.path.join(workdir, "source" + pathlib.Path(source_name).suffix.lower())
            target_suffix = pathlib.Path(target_name).suffix.lower()
            target_file = os.path.join(workdir, "target" + target_suffix)
            output_file = os.path.join(workdir, "output" + target_suffix)
            pathlib.Path(source_file).write_bytes(source_bytes)
            pathlib.Path(target_file).write_bytes(target_bytes)

            if any(p != "face_swapper" for p in options.get("frame_processors", [])):
                from modules.headless import download_models as fetch

                fetch(options["frame_processors"])
                models_volume.commit()

            run_job(source_file, target_file, output_file, **options)
            return pathlib.Path(output_file).read_bytes()

    @modal.asgi_app()
    def web(self):
        from cloud.server import app as fastapi_app

        return fastapi_app


@app.local_entrypoint()
def main(
    source: str,
    target: str,
    output: str = "",
    processors: str = "face_swapper",
    many_faces: bool = False,
    mouth_mask: bool = False,
    keep_fps: bool = True,
    keep_audio: bool = True,
    video_encoder: str = "libx264",
    video_quality: int = 18,
    det_size: int = 640,
) -> None:
    source_path = pathlib.Path(source)
    target_path = pathlib.Path(target)
    output_path = pathlib.Path(output) if output else target_path.with_name(
        f"{target_path.stem}_swapped{target_path.suffix}"
    )

    result = DeepLiveCam().swap.remote(
        source_path.read_bytes(),
        source_path.name,
        target_path.read_bytes(),
        target_path.name,
        frame_processors=[p.strip() for p in processors.split(",") if p.strip()],
        many_faces=many_faces,
        mouth_mask=mouth_mask,
        keep_fps=keep_fps,
        keep_audio=keep_audio,
        video_encoder=video_encoder,
        video_quality=video_quality,
        det_size=det_size,
    )
    output_path.write_bytes(result)
    print(f"Saved {output_path} ({len(result) / 1024 / 1024:.1f} MB)")
