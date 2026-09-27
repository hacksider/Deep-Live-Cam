"""Minimal HTTP API around ``modules.headless.process``.

Runs anywhere with a GPU and Docker (RunPod, Vast.ai, Lambda, Paperspace, a
bare EC2/GCP VM ...) and is also what the Modal app serves as its web endpoint.

    uvicorn cloud.server:app --host 0.0.0.0 --port 8000

    curl -F source=@face.jpg -F target=@clip.mp4 \
         -F frame_processors=face_swapper,face_enhancer \
         http://HOST:8000/swap -o out.mp4
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from starlette.background import BackgroundTask  # noqa: E402

from modules.globals import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS  # noqa: E402

# Deep-Live-Cam keeps its settings in module globals, so one process can only
# run one job at a time. Scale out with more containers, not more threads.
_JOB_LOCK = threading.Lock()

app = FastAPI(title="Deep-Live-Cam", version="1.0")


def _suffix(upload: UploadFile, allowed: tuple) -> str:
    suffix = os.path.splitext(upload.filename or "")[1].lower()
    if suffix not in allowed:
        raise HTTPException(400, f"{upload.filename!r}: expected one of {', '.join(allowed)}")
    return suffix


def run_job(source_file: str, target_file: str, output_file: str, **options) -> str:
    """Serialised call into the pipeline; shared by the HTTP API and Modal."""
    from modules.headless import process

    with _JOB_LOCK:
        return process(source_file, target_file, output_file, **options)


@app.get("/health")
def health() -> dict:
    import onnxruntime

    return {"status": "ok", "providers": onnxruntime.get_available_providers()}


@app.post("/swap")
async def swap(
    source: UploadFile = File(..., description="Image containing the face to use"),
    target: UploadFile = File(..., description="Image or video to put the face on"),
    frame_processors: str = Form("face_swapper"),
    many_faces: bool = Form(False),
    mouth_mask: bool = Form(False),
    keep_fps: bool = Form(True),
    keep_audio: bool = Form(True),
    video_encoder: str = Form("libx264"),
    video_quality: int = Form(18),
    det_size: int = Form(640),
):
    source_suffix = _suffix(source, IMAGE_EXTENSIONS)
    target_suffix = _suffix(target, IMAGE_EXTENSIONS + VIDEO_EXTENSIONS)

    workdir = tempfile.mkdtemp(prefix="dlc-")
    cleanup = BackgroundTask(shutil.rmtree, workdir, ignore_errors=True)
    try:
        source_file = os.path.join(workdir, "source" + source_suffix)
        target_file = os.path.join(workdir, "target" + target_suffix)
        output_file = os.path.join(workdir, "output" + target_suffix)
        for upload, path in ((source, source_file), (target, target_file)):
            with open(path, "wb") as handle:
                shutil.copyfileobj(upload.file, handle)

        await run_in_threadpool(
            run_job,
            source_file,
            target_file,
            output_file,
            frame_processors=[p.strip() for p in frame_processors.split(",") if p.strip()],
            many_faces=many_faces,
            mouth_mask=mouth_mask,
            keep_fps=keep_fps,
            keep_audio=keep_audio,
            video_encoder=video_encoder,
            video_quality=video_quality,
            det_size=det_size,
        )
    except ValueError as error:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(400, str(error))
    except Exception as error:
        shutil.rmtree(workdir, ignore_errors=True)
        raise HTTPException(500, str(error))

    download_name = os.path.splitext(target.filename or "output")[0] + "_swapped" + target_suffix
    return FileResponse(output_file, filename=download_name, background=cleanup)
