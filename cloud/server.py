"""HTTP + WebSocket API around the headless pipeline.

Runs anywhere with a GPU and Docker (RunPod, Vast.ai, Lambda, Paperspace, a
bare EC2/GCP VM ...) and is also what the Modal app serves as its web endpoint.

    uvicorn cloud.server:app --host 0.0.0.0 --port 8000

    # files: swap a face into an image or video
    curl -F source=@face.jpg -F target=@clip.mp4 \
         -F frame_processors=face_swapper,face_enhancer \
         http://HOST:8000/swap -o out.mp4

    # live webcam: stream frames over the /live WebSocket
    python cloud/live_client.py --server ws://HOST:8000 --source face.jpg

Set DLC_API_TOKEN to require ``Authorization: Bearer <token>`` (or
``?token=<token>`` on /live). Strongly recommended on any public address:
anyone who can reach an open endpoint can spend your GPU time.

/live protocol (the reference client is cloud/live_client.py):
  1. client -> text  {"type": "start", "source": <base64 image>, "options": {...}}
     server -> text  {"type": "ready", ...}  or  {"type": "error", "message": ...}
  2. client -> bytes 4-byte big-endian frame id + JPEG frame
     server -> bytes the same frame id + swapped JPEG frame (in order)
  3. client -> text  {"type": "update", "options": {...}, "source"?: <base64>}
     server -> text  {"type": "updated"} or {"type": "error", ...}
     (an update replaces all options; omitted ones return to their defaults)
  options: frame_processors (list), many_faces, mouth_mask, det_size,
           detect_every, opacity, jpeg_quality
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fastapi import (  # noqa: E402
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from starlette.background import BackgroundTask  # noqa: E402

from modules.globals import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS  # noqa: E402

# Deep-Live-Cam keeps its settings in module globals, so one process can only
# run one job at a time. Scale out with more containers, not more threads.
_JOB_LOCK = threading.Lock()

API_TOKEN = os.environ.get("DLC_API_TOKEN") or None
# A live session that sends no frames for this long is closed so an
# abandoned connection doesn't keep a paid GPU busy.
LIVE_IDLE_TIMEOUT = float(os.environ.get("DLC_LIVE_IDLE_TIMEOUT", "60"))
LIVE_MAX_PIXELS = 1920 * 1080
FRAME_HEADER = struct.Struct(">I")

app = FastAPI(title="Deep-Live-Cam", version="1.1")


def _token_ok(presented: str | None) -> bool:
    if API_TOKEN is None:
        return True
    return presented is not None and hmac.compare_digest(presented, API_TOKEN)


def require_token(authorization: str | None = Header(default=None)) -> None:
    presented = authorization[7:] if authorization and authorization.startswith("Bearer ") else None
    if not _token_ok(presented):
        raise HTTPException(401, "Missing or invalid API token")


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


@app.post("/swap", dependencies=[Depends(require_token)])
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


# ─── live webcam sessions ─────────────────────────────────────────────────


def _live_options(message: dict) -> tuple[dict, int]:
    options = dict(message.get("options") or {})
    jpeg_quality = int(options.pop("jpeg_quality", 85))
    if not 30 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 30 and 100")
    allowed = {"frame_processors", "many_faces", "mouth_mask", "det_size", "detect_every", "opacity"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown option(s): {sorted(unknown)}")
    return options, jpeg_quality


def _decode_source(message: dict) -> bytes | None:
    source = message.get("source")
    if source is None:
        return None
    try:
        return base64.b64decode(source, validate=True)
    except (ValueError, TypeError):
        raise ValueError("source must be a base64-encoded image")


def _process_live_frame(session, payload: bytes, jpeg_quality: int) -> bytes:
    from modules.live import decode_image, encode_jpeg

    frame = decode_image(payload)
    height, width = frame.shape[:2]
    if height * width > LIVE_MAX_PIXELS:
        raise ValueError(f"Frame {width}x{height} is larger than 1920x1080")
    return encode_jpeg(session.process(frame), jpeg_quality)


async def _reject(websocket: WebSocket, message: str, code: int) -> None:
    """Report an error and close; the reason is repeated in the close frame
    because a client may see the close before it reads the error message."""
    await _send_error(websocket, message)
    await websocket.close(code=code, reason=message.encode()[:120].decode(errors="ignore"))


async def _send_error(websocket: WebSocket, message: str, frame_id: int | None = None) -> None:
    payload = {"type": "error", "message": message}
    if frame_id is not None:
        payload["frame_id"] = frame_id  # lets the client stop waiting for that frame
    await websocket.send_text(json.dumps(payload))


@app.websocket("/live")
async def live(websocket: WebSocket) -> None:
    await websocket.accept()
    header = websocket.headers.get("authorization") or ""
    presented = websocket.query_params.get("token") or (header[7:] if header.startswith("Bearer ") else None)
    if not _token_ok(presented):
        await _reject(websocket, "Missing or invalid API token", 1008)
        return
    if not _JOB_LOCK.acquire(blocking=False):
        await _reject(websocket, "This GPU is busy with another session or job; try again shortly.", 1013)
        return

    # One dedicated thread per session: GPU work stays off the event loop,
    # and every frame runs on the same thread, like the desktop app's worker.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dlc-live")
    loop = asyncio.get_running_loop()
    try:
        from modules.live import LiveSession
        import onnxruntime

        session = LiveSession()
        try:
            start = json.loads(await asyncio.wait_for(websocket.receive_text(), LIVE_IDLE_TIMEOUT))
            if start.get("type") != "start":
                raise ValueError('The first message must be {"type": "start", ...}')
            options, jpeg_quality = _live_options(start)
            source = _decode_source(start)
            await loop.run_in_executor(executor, lambda: session.configure(source, **options))
        except asyncio.TimeoutError:
            await _reject(websocket, "No start message received; closing session.", 1000)
            return
        except (ValueError, RuntimeError, KeyError, TypeError, AttributeError) as error:
            await _reject(websocket, str(error) or "Invalid start message", 1003)
            return

        await websocket.send_text(json.dumps({
            "type": "ready",
            "providers": onnxruntime.get_available_providers(),
            "idle_timeout": LIVE_IDLE_TIMEOUT,
        }))

        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), LIVE_IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                await _reject(websocket, f"No frames for {LIVE_IDLE_TIMEOUT:.0f}s; closing session.", 1000)
                return
            if message["type"] == "websocket.disconnect":
                return

            if message.get("bytes") is not None:
                payload = message["bytes"]
                if len(payload) <= FRAME_HEADER.size:
                    await _send_error(websocket, "Frame message too short")
                    continue
                frame_id = payload[:FRAME_HEADER.size]
                try:
                    result = await loop.run_in_executor(
                        executor, _process_live_frame, session, payload[FRAME_HEADER.size:], jpeg_quality
                    )
                except ValueError as error:
                    await _send_error(websocket, str(error), FRAME_HEADER.unpack(frame_id)[0])
                    continue
                await websocket.send_bytes(frame_id + result)
                continue

            try:
                update = json.loads(message.get("text") or "")
                if update.get("type") != "update":
                    raise ValueError(f"Unexpected message type {update.get('type')!r}")
                options, jpeg_quality = _live_options(update)
                source = _decode_source(update)
                await loop.run_in_executor(executor, lambda: session.configure(source, **options))
                await websocket.send_text(json.dumps({"type": "updated"}))
            except (ValueError, RuntimeError, TypeError, AttributeError) as error:
                await _send_error(websocket, str(error))
    except WebSocketDisconnect:
        pass
    finally:
        executor.shutdown(wait=True)
        _JOB_LOCK.release()
