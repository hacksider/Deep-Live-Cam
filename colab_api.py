from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import mimetypes
import queue
import subprocess
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

import colab_batch

DRIVE_ROOT = Path("/content/drive/MyDrive/DeepLiveCamRemote")
SOURCE_DIR = DRIVE_ROOT / "source"
PHOTOS_DIR = DRIVE_ROOT / "photos"
VIDEOS_DIR = DRIVE_ROOT / "videos"
OUTPUT_PHOTOS_DIR = DRIVE_ROOT / "outputs" / "photos"
OUTPUT_VIDEOS_DIR = DRIVE_ROOT / "outputs" / "videos"

LOCAL_ROOT = Path("/content/inputs")
LOCAL_SOURCE_DIR = LOCAL_ROOT / "source"
LOCAL_PHOTOS_DIR = LOCAL_ROOT / "photos"
LOCAL_VIDEOS_DIR = LOCAL_ROOT / "videos"
LOCAL_OUTPUT_PHOTOS_DIR = Path("/content/outputs/photos")
LOCAL_OUTPUT_VIDEOS_DIR = Path("/content/outputs/videos")
ZIP_OUTPUT_DIR = Path("/content/outputs/downloads")
ARCHIVE_DIR = Path("/content/archive")

OUTPUT_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
OUTPUT_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
API_VERSION = "live-fast-detect-v7"
LIVE_FACE_MODEL_PACKS = {"buffalo_l", "buffalo_m", "buffalo_s"}
LIVE_SWAPPER_PRECISIONS = {"fp32", "fp16"}
LIVE_FRAME_CODECS = {"jpeg", "webp"}


def bool_config(config: dict[str, Any], name: str, default: bool) -> bool:
    value = config.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


class JobRequest(BaseModel):
    source_face: str = Field(default=str(SOURCE_DIR / "source.png"))
    input_dir: str | None = None
    output_dir: str | None = None
    recursive: bool = True
    overwrite: bool = False
    skip_processed: bool = True
    many_faces: bool = False
    enhancer: str = "none"
    opacity: float = 1.0
    sharpness: float = 0.0
    mouth_mask_size: float = 0.0
    poisson_blend: bool = False
    color_correction: bool = False
    interpolation_weight: float = 0.0
    max_fps: float = 30.0
    max_width: int = 420
    quality: int = 18
    encoder: str = "auto"
    start_pct: float = 0.0
    end_pct: float = 100.0


class CancelRequest(BaseModel):
    job_id: str


class CreateZipResponse(BaseModel):
    zip_path: str
    zip_id: str
    size_bytes: int
    timestamp: str
    tailscale_hostname: str | None


@dataclass
class JobState:
    job_id: str
    kind: str
    status: str = "queued"
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    exit_code: int | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    log_queue: "queue.Queue[str]" = field(default_factory=queue.Queue)
    logs: list[str] = field(default_factory=list)

    def append(self, text: str) -> None:
        if not text:
            return
        for line in text.splitlines():
            entry = line.rstrip()
            self.logs.append(entry)
            self.log_queue.put(entry)

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "logs": self.logs[-300:],
        }


class JobWriter(io.TextIOBase):
    def __init__(self, job: JobState):
        self.job = job
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.job.append(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self.job.append(self._buffer)
            self._buffer = ""


JOBS: dict[str, JobState] = {}
CREATED_ARCHIVES: dict[str, Path] = {}
ENGINE_LOCK = threading.Lock()
app = FastAPI(title="Deep-Live-Cam Remote API", version="1.0")


def ensure_drive_layout() -> dict[str, str]:
    for path in (SOURCE_DIR, PHOTOS_DIR, VIDEOS_DIR, OUTPUT_PHOTOS_DIR, OUTPUT_VIDEOS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "drive_root": str(DRIVE_ROOT),
        "source_dir": str(SOURCE_DIR),
        "photos_dir": str(PHOTOS_DIR),
        "videos_dir": str(VIDEOS_DIR),
        "output_photos_dir": str(OUTPUT_PHOTOS_DIR),
        "output_videos_dir": str(OUTPUT_VIDEOS_DIR),
    }


def ensure_local_layout() -> dict[str, str]:
    for path in (LOCAL_SOURCE_DIR, LOCAL_PHOTOS_DIR, LOCAL_VIDEOS_DIR, LOCAL_OUTPUT_PHOTOS_DIR, LOCAL_OUTPUT_VIDEOS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "source_dir": str(LOCAL_SOURCE_DIR),
        "photos_dir": str(LOCAL_PHOTOS_DIR),
        "videos_dir": str(LOCAL_VIDEOS_DIR),
        "output_photos_dir": str(LOCAL_OUTPUT_PHOTOS_DIR),
        "output_videos_dir": str(LOCAL_OUTPUT_VIDEOS_DIR),
    }


def safe_upload_name(filename: str | None, fallback: str) -> str:
    name = Path(filename or fallback).name
    return name or fallback


def upload_destination(kind: str, filename: str | None) -> tuple[Path, dict[str, str]]:
    paths = ensure_local_layout()
    normalized_kind = kind.lower()
    if normalized_kind == "source":
        return LOCAL_SOURCE_DIR / safe_upload_name(filename, "source.png"), paths
    if normalized_kind in {"photo", "photos"}:
        return LOCAL_PHOTOS_DIR / safe_upload_name(filename, "photo.jpg"), paths
    if normalized_kind in {"video", "videos"}:
        return LOCAL_VIDEOS_DIR / safe_upload_name(filename, "video.mp4"), paths
    raise ValueError(f"unknown upload kind: {kind}")


def output_roots(kind: str) -> tuple[list[tuple[str, Path]], set[str]]:
    normalized = kind.lower()
    if normalized == "photos":
        return [("drive", OUTPUT_PHOTOS_DIR), ("local", LOCAL_OUTPUT_PHOTOS_DIR)], OUTPUT_IMAGE_EXTENSIONS
    if normalized == "videos":
        return [("drive", OUTPUT_VIDEOS_DIR), ("local", LOCAL_OUTPUT_VIDEOS_DIR)], OUTPUT_VIDEO_EXTENSIONS
    raise HTTPException(status_code=404, detail=f"unknown output kind: {kind}")


def output_root(kind: str, source: str) -> tuple[Path, set[str]]:
    roots, extensions = output_roots(kind)
    for root_source, root in roots:
        if root_source == source:
            return root, extensions
    raise HTTPException(status_code=404, detail=f"unknown output source: {source}")


def output_file_entries(kind: str) -> list[dict[str, Any]]:
    ensure_drive_layout()
    ensure_local_layout()
    roots, extensions = output_roots(kind)
    files: list[dict[str, Any]] = []
    for source, root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            stat = path.stat()
            relative_path = path.relative_to(root).as_posix()
            files.append({
                "name": path.name,
                "relative_path": relative_path,
                "source": source,
                "path": path,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "download_path": f"/outputs/{kind}/file/{source}/{relative_path}",
            })
    files.sort(key=lambda item: (item["modified"], item["name"]), reverse=True)
    return files


def safe_output_path(kind: str, source: str, relative_path: str) -> Path:
    root, extensions = output_root(kind, source)
    candidate = (root / relative_path).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise HTTPException(status_code=400, detail="invalid output path")
    if candidate.suffix.lower() not in extensions:
        raise HTTPException(status_code=400, detail="unsupported output file type")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="output file not found")
    return candidate


def remove_file(path: str) -> None:
    Path(path).unlink(missing_ok=True)


def get_tailscale_hostname() -> str | None:
    """Returns Tailscale hostname if tailscale CLI available and connected."""
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if result.returncode == 0:
            status_data = json.loads(result.stdout)
            return status_data.get("Self", {}).get("HostName")
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return None


async def write_upload(file: UploadFile, dest: Path) -> int:
    content = await file.read()
    dest.write_bytes(content)
    return len(content)


def bool_arg(name: str, value: bool) -> list[str]:
    return [f"--{name}" if value else f"--no-{name}"]


def common_args(request: JobRequest, input_default: Path, output_default: Path) -> list[str]:
    args = [
        "--source-face", request.source_face,
        "--input-dir", request.input_dir or str(input_default),
        "--output-dir", request.output_dir or str(output_default),
        *bool_arg("recursive", request.recursive),
        *bool_arg("overwrite", request.overwrite),
        *bool_arg("skip-processed", request.skip_processed),
        *bool_arg("many-faces", request.many_faces),
        "--opacity", str(request.opacity),
        "--sharpness", str(request.sharpness),
        "--mouth-mask-size", str(request.mouth_mask_size),
        "--interpolation-weight", str(request.interpolation_weight),
        "--enhancer", request.enhancer,
    ]
    if request.poisson_blend:
        args.append("--poisson-blend")
    if request.color_correction:
        args.append("--color-correction")
    return args


def run_job(job: JobState, argv: list[str]) -> None:
    job.status = "running"
    writer = JobWriter(job)
    try:
        with ENGINE_LOCK:
            parser = colab_batch.build_parser()
            args = parser.parse_args(argv)
            args.cancel_event = job.cancel_event
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                job.exit_code = args.func(args)
        job.status = "cancelled" if job.cancel_event.is_set() else ("completed" if job.exit_code == 0 else "failed")
    except BaseException as exc:
        job.error = str(exc)
        job.status = "cancelled" if job.cancel_event.is_set() else "failed"
        job.append(f"ERROR: {exc}")
    finally:
        writer.flush()
        job.finished_at = time.time()


def start_job(kind: str, argv: list[str]) -> JobState:
    job = JobState(job_id=uuid.uuid4().hex, kind=kind)
    JOBS[job.job_id] = job
    threading.Thread(target=run_job, args=(job, argv), daemon=True).start()
    return job


def int_config(config: dict[str, Any], name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(config.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def live_processing_geometry(frame: np.ndarray, config: dict[str, Any]) -> tuple[int, int]:
    height, width = frame.shape[:2]
    configured = config.get("max_width")
    try:
        max_width = int(configured) if configured else width
    except (TypeError, ValueError):
        max_width = width
    max_width = max(2, min(width, max_width))
    process_width, process_height, _fps = colab_batch.processing_geometry(width, height, 30.0, max_width, 30.0)
    return process_width, process_height


def live_detection_size(config: dict[str, Any]) -> int:
    detector_size = int_config(config, "detector_size", 320, 160, 640)
    return max(32, detector_size // 32 * 32)


def live_jpeg_quality(config: dict[str, Any]) -> int:
    return int_config(config, "jpeg_quality", 80, 20, 95)


def live_frame_codec(config: dict[str, Any]) -> str:
    codec = str(config.get("frame_codec") or "jpeg").lower()
    if codec not in LIVE_FRAME_CODECS:
        return "jpeg"
    return codec


def live_output_codec(config: dict[str, Any]) -> str:
    codec = str(config.get("output_codec") or live_frame_codec(config)).lower()
    if codec not in LIVE_FRAME_CODECS:
        return "jpeg"
    return codec


def live_encode_frame(frame: np.ndarray, config: dict[str, Any]) -> tuple[bool, Any, str]:
    quality = live_jpeg_quality(config)
    requested = live_output_codec(config)
    if requested == "webp":
        try:
            ok, encoded = cv2.imencode(".webp", frame, [int(getattr(cv2, "IMWRITE_WEBP_QUALITY", cv2.IMWRITE_JPEG_QUALITY)), quality])
        except Exception:
            ok, encoded = False, None
        if ok:
            return True, encoded, "webp"
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return bool(ok), encoded, "jpeg"


def live_face_model_pack(config: dict[str, Any]) -> str:
    model_pack = str(config.get("face_model_pack") or "buffalo_l")
    if model_pack not in LIVE_FACE_MODEL_PACKS:
        return "buffalo_l"
    return model_pack


def live_swapper_precision(config: dict[str, Any]) -> str:
    precision = str(config.get("swapper_precision") or "fp32").lower()
    if precision not in LIVE_SWAPPER_PRECISIONS:
        return "fp32"
    return precision


def live_swapper_diagnostics(engine: colab_batch.ModernEngine) -> dict[str, str]:
    diagnostics = {}
    getter = getattr(engine.swapper, "get_face_swapper_diagnostics", None)
    if callable(getter):
        loaded = getter()
        if isinstance(loaded, dict):
            diagnostics.update({str(key): str(value) for key, value in loaded.items()})
    diagnostics.setdefault("requested_precision", live_swapper_precision({}))
    diagnostics.setdefault("loaded_precision", "")
    diagnostics.setdefault("model_path", "")
    return diagnostics


def live_detect_faces(frame: np.ndarray, many_faces: bool, detector_size: int) -> Any:
    from insightface.app.common import Face
    from modules.face_analyser import get_face_analyser

    fa = get_face_analyser()
    input_size = (detector_size, detector_size)
    max_num = 0 if many_faces else 1
    bboxes, kpss = fa.det_model.detect(frame, input_size=input_size, max_num=max_num, metric="default")
    if bboxes.shape[0] == 0:
        return [] if many_faces else None
    faces = [Face(bbox=bboxes[i, :4], kps=kpss[i], det_score=bboxes[i, 4]) for i in range(bboxes.shape[0])]
    if many_faces:
        return faces
    return min(faces, key=lambda face: face.bbox[0])


def live_process_frame(engine: colab_batch.ModernEngine, frame: np.ndarray, config: dict[str, Any], state: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    timings: dict[str, Any] = {
        "detect": 0.0,
        "landmarks": 0.0,
        "swap": 0.0,
        "post": 0.0,
        "enhance": 0.0,
        "detect_reused": False,
        "faces": 0,
    }
    if engine.mapping:
        started = time.monotonic()
        return engine.process(frame, "live"), {**timings, "swap": time.monotonic() - started}

    detector_size = live_detection_size(config)
    detect_every_n = int_config(config, "detect_every_n", 1, 1, 30)
    many_faces = bool(engine.globals.many_faces)
    needs_landmarks = bool(engine.enhancer) or bool(getattr(engine.globals, "mouth_mask", False))
    frame_index = int(state.get("frame_index", 0))
    cache_key = "many_faces" if many_faces else "single_face"
    should_detect = frame_index % detect_every_n == 0 or state.get(cache_key) is None
    state["frame_index"] = frame_index + 1

    detect_started = time.monotonic()
    if should_detect:
        detected = live_detect_faces(frame, many_faces, detector_size)
        state[cache_key] = detected
    else:
        detected = state.get(cache_key)
        timings["detect_reused"] = True
    timings["detect"] = time.monotonic() - detect_started

    landmark_started = time.monotonic()
    if needs_landmarks:
        from modules.face_analyser import ensure_landmarks
        ensure_landmarks(frame, detected)
    timings["landmarks"] = time.monotonic() - landmark_started

    if getattr(engine.globals, "opacity", 1.0) == 0:
        if hasattr(engine.swapper, "PREVIOUS_FRAME_RESULT"):
            engine.swapper.PREVIOUS_FRAME_RESULT = None
        return frame, timings

    swap_started = time.monotonic()
    if not getattr(engine, "cache_source_face", True):
        engine.refresh_default_source()
    bboxes = []
    if many_faces:
        faces = detected or []
        output = frame.copy()
        for face in faces:
            output = engine.swapper.swap_face(engine.default_source, face, output)
            if face is not None and hasattr(face, "bbox") and face.bbox is not None:
                bboxes.append(face.bbox.astype(int))
        detected_for_enhancer = faces
    else:
        face = detected
        output = frame
        if face is not None:
            output = engine.swapper.swap_face(engine.default_source, face, output)
            if hasattr(face, "bbox") and face.bbox is not None:
                bboxes.append(face.bbox.astype(int))
        detected_for_enhancer = [face] if face is not None else []
    timings["swap"] = time.monotonic() - swap_started
    timings["faces"] = len(detected_for_enhancer)

    post_started = time.monotonic()
    output = engine.swapper.apply_post_processing(output, bboxes)
    timings["post"] = time.monotonic() - post_started

    enhance_started = time.monotonic()
    if engine.enhancer:
        output = engine.enhancer.process_frame(None, output, detected_faces=detected_for_enhancer)
    timings["enhance"] = time.monotonic() - enhance_started
    return output, timings


@app.get("/health")
def health() -> dict[str, Any]:
    paths = ensure_drive_layout()
    return {
        "ok": True,
        "api_version": API_VERSION,
        "paths": paths,
        "local_paths": ensure_local_layout(),
        "active_jobs": [job.snapshot() for job in JOBS.values() if job.status in {"queued", "running"}],
    }


@app.get("/outputs/{kind}")
def list_outputs(kind: str) -> dict[str, Any]:
    files = output_file_entries(kind)
    public_files = [{key: value for key, value in item.items() if key != "path"} for item in files]
    return {"ok": True, "kind": kind, "count": len(public_files), "files": public_files}


@app.get("/outputs/{kind}/zip")
def get_output_zip(kind: str) -> FileResponse:
    files = output_file_entries(kind)
    if not files:
        raise HTTPException(status_code=404, detail=f"no {kind} outputs found")
    ZIP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ZIP_OUTPUT_DIR / f"{kind}_outputs_{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for item in files:
            archive.write(item["path"], f"{item['source']}/{item['relative_path']}")
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"{kind}_outputs.zip",
        background=BackgroundTask(remove_file, str(zip_path)),
    )


@app.post("/outputs/{kind}/create-zip")
def create_output_zip(kind: str) -> CreateZipResponse:
    """
    Creates a zip archive of outputs in /content/archive/ for Taildrop or HTTP download.
    Does NOT auto-cleanup (user manages retention).
    """
    files = output_file_entries(kind)
    if not files:
        raise HTTPException(status_code=404, detail=f"no {kind} outputs found")

    # Create archive directory
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Generate timestamped filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    zip_id = uuid.uuid4().hex
    zip_filename = f"{kind}_outputs_{timestamp}.zip"
    zip_path = ARCHIVE_DIR / zip_filename

    # Create ZIP
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for item in files:
            archive.write(item["path"], f"{item['source']}/{item['relative_path']}")

    # Store for HTTP fallback
    CREATED_ARCHIVES[zip_id] = zip_path

    # Get Tailscale info
    tailscale_hostname = get_tailscale_hostname()

    return CreateZipResponse(
        zip_path=str(zip_path),
        zip_id=zip_id,
        size_bytes=zip_path.stat().st_size,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tailscale_hostname=tailscale_hostname,
    )


@app.get("/download-archive/{archive_id}")
def download_archive(archive_id: str) -> FileResponse:
    """
    HTTP fallback download for archives created via /create-zip.
    Used when Taildrop transfer fails.
    """
    zip_path = CREATED_ARCHIVES.get(archive_id)
    if not zip_path or not zip_path.is_file():
        raise HTTPException(status_code=404, detail="archive not found or expired")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_path.name,
    )


@app.get("/outputs/{kind}/file/{source}/{relative_path:path}")
def get_output_file(kind: str, source: str, relative_path: str) -> FileResponse:
    path = safe_output_path(kind, source, relative_path)
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.post("/upload/file")
async def upload_file(kind: str = "photos", file: UploadFile = File(...)) -> dict[str, Any]:
    dest, paths = upload_destination(kind, file.filename)
    size = await write_upload(file, dest)
    response = {"ok": True, "kind": kind, "path": str(dest), "size": size}
    if kind.lower() in {"photo", "photos"}:
        response.update({"input_dir": paths["photos_dir"], "output_dir": paths["output_photos_dir"]})
    elif kind.lower() in {"video", "videos"}:
        response.update({"input_dir": paths["videos_dir"], "output_dir": paths["output_videos_dir"]})
    return response


@app.post("/upload/source")
async def upload_source(file: UploadFile = File(...)) -> dict[str, Any]:
    dest, _ = upload_destination("source", file.filename)
    size = await write_upload(file, dest)
    return {"ok": True, "path": str(dest), "size": size}


@app.post("/upload/photos")
async def upload_photos(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    paths = ensure_local_layout()
    uploaded = []
    for file in files:
        dest = LOCAL_PHOTOS_DIR / safe_upload_name(file.filename, f"photo_{len(uploaded)}.jpg")
        size = await write_upload(file, dest)
        uploaded.append({"path": str(dest), "size": size})
    return {"ok": True, "count": len(uploaded), "files": uploaded, "input_dir": paths["photos_dir"], "output_dir": paths["output_photos_dir"]}


@app.post("/upload/videos")
async def upload_videos(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    paths = ensure_local_layout()
    uploaded = []
    for file in files:
        dest = LOCAL_VIDEOS_DIR / safe_upload_name(file.filename, f"video_{len(uploaded)}.mp4")
        size = await write_upload(file, dest)
        uploaded.append({"path": str(dest), "size": size})
    return {"ok": True, "count": len(uploaded), "files": uploaded, "input_dir": paths["videos_dir"], "output_dir": paths["output_videos_dir"]}


@app.delete("/upload/clear")
def clear_uploads() -> dict[str, Any]:
    cleared = []
    for directory in (LOCAL_SOURCE_DIR, LOCAL_PHOTOS_DIR, LOCAL_VIDEOS_DIR, LOCAL_OUTPUT_PHOTOS_DIR, LOCAL_OUTPUT_VIDEOS_DIR):
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()
                    cleared.append(str(path))
    return {"ok": True, "cleared": len(cleared)}


@app.post("/jobs/photos")
def start_photos(request: JobRequest) -> dict[str, Any]:
    ensure_drive_layout()
    argv = ["photos", *common_args(request, PHOTOS_DIR, OUTPUT_PHOTOS_DIR)]
    job = start_job("photos", argv)
    return {"job_id": job.job_id, "status": job.status}


@app.post("/jobs/videos")
def start_videos(request: JobRequest) -> dict[str, Any]:
    ensure_drive_layout()
    argv = [
        "process",
        *common_args(request, VIDEOS_DIR, OUTPUT_VIDEOS_DIR),
        "--max-fps", str(request.max_fps),
        "--max-width", str(request.max_width),
        "--quality", str(request.quality),
        "--encoder", request.encoder,
        "--start-pct", str(request.start_pct),
        "--end-pct", str(request.end_pct),
    ]
    job = start_job("videos", argv)
    return {"job_id": job.job_id, "status": job.status}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        return {"error": "unknown job", "job_id": job_id}
    return job.snapshot()


@app.post("/jobs/cancel")
def cancel_job(request: CancelRequest) -> dict[str, Any]:
    job = JOBS.get(request.job_id)
    if job is None:
        return {"error": "unknown job", "job_id": request.job_id}
    job.cancel_event.set()
    job.append("cancel requested")
    return {"job_id": job.job_id, "status": job.status, "cancel_requested": True}


@app.websocket("/ws/jobs/{job_id}")
async def job_socket(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    job = JOBS.get(job_id)
    if job is None:
        await websocket.send_json({"error": "unknown job", "job_id": job_id})
        await websocket.close()
        return
    await websocket.send_json(job.snapshot())
    try:
        while True:
            try:
                line = job.log_queue.get_nowait()
                await websocket.send_json({"job_id": job_id, "log": line, "status": job.status})
            except queue.Empty:
                await websocket.send_json({"job_id": job_id, "status": job.status})
                if job.status not in {"queued", "running"}:
                    break
                await asyncio.sleep(1.0)
    finally:
        await websocket.close()


@app.websocket("/ws/live")
async def live_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        config_payload = await websocket.receive_text()
        config = json.loads(config_payload)
        process_config = colab_batch.ProcessConfig(
            input_dir=PHOTOS_DIR,
            output_dir=OUTPUT_PHOTOS_DIR,
            source_face=Path(config.get("source_face") or SOURCE_DIR / "source.png"),
            map_config=None,
            many_faces=bool(config.get("many_faces", False)),
            opacity=float(config.get("opacity", 1.0)),
            sharpness=float(config.get("sharpness", 0.0)),
            mouth_mask_size=float(config.get("mouth_mask_size", 0.0)),
            poisson_blend=bool(config.get("poisson_blend", False)),
            color_correction=bool(config.get("color_correction", False)),
            interpolation_weight=float(config.get("interpolation_weight", 0.0)),
            enhancer=config.get("enhancer", "none"),
            face_model_pack=live_face_model_pack(config),
            swapper_precision=live_swapper_precision(config),
            cache_source_face=bool_config(config, "cache_source_face", True),
        )
        with ENGINE_LOCK:
            engine = colab_batch.ModernEngine(process_config)
    except Exception as exc:
        await websocket.send_json({"error": f"live init failed: {exc}"})
        await websocket.close(code=1011)
        return

    swapper_diagnostics = live_swapper_diagnostics(engine)
    await websocket.send_json({
        "status": "ready",
        "api_version": API_VERSION,
        "live_fast_detection": True,
        "detector_size": live_detection_size(config),
        "detect_every_n": int_config(config, "detect_every_n", 1, 1, 30),
        "face_model_pack": live_face_model_pack(config),
        "swapper_precision": live_swapper_precision(config),
        "swapper_loaded_precision": swapper_diagnostics.get("loaded_precision", ""),
        "swapper_model_path": swapper_diagnostics.get("model_path", ""),
        "source_embedding_cached": engine.default_source is not None and engine.cache_source_face,
        "cache_source_face": engine.cache_source_face,
        "frame_codec": live_frame_codec(config),
        "output_codec": live_output_codec(config),
        "frame_quality": live_jpeg_quality(config),
        "jpeg_quality": live_jpeg_quality(config),
    })
    geometry_logged = False
    live_state: dict[str, Any] = {}
    perf_started = time.monotonic()
    perf_frames = 0
    perf_wait = 0.0
    perf_decode = 0.0
    perf_resize = 0.0
    perf_process = 0.0
    perf_detect = 0.0
    perf_landmarks = 0.0
    perf_swap = 0.0
    perf_post = 0.0
    perf_enhance = 0.0
    perf_detect_reused = 0
    perf_faces = 0
    perf_encode = 0.0
    perf_in_bytes = 0
    perf_out_bytes = 0
    try:
        while True:
            frame_started = time.monotonic()
            payload = await websocket.receive_bytes()
            received_at = time.monotonic()
            array = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
            decoded_at = time.monotonic()
            if frame is None:
                await websocket.send_json({"error": "invalid frame"})
                continue
            frame_height, frame_width = frame.shape[:2]
            process_width, process_height = live_processing_geometry(frame, config)
            if (process_width, process_height) == (frame_width, frame_height):
                process_frame = frame
            else:
                interpolation = cv2.INTER_AREA if process_width < frame_width or process_height < frame_height else cv2.INTER_LINEAR
                process_frame = cv2.resize(frame, (process_width, process_height), interpolation=interpolation)
            resized_at = time.monotonic()
            if not geometry_logged:
                await websocket.send_json({
                    "status": "live_geometry",
                    "api_version": API_VERSION,
                    "input": f"{frame_width}x{frame_height}",
                    "processing": f"{process_width}x{process_height}",
                    "detector_size": live_detection_size(config),
                    "detect_every_n": int_config(config, "detect_every_n", 1, 1, 30),
                    "face_model_pack": live_face_model_pack(config),
                    "swapper_precision": live_swapper_precision(config),
                    "swapper_loaded_precision": live_swapper_diagnostics(engine).get("loaded_precision", ""),
                    "cache_source_face": getattr(engine, "cache_source_face", True),
                    "frame_codec": live_frame_codec(config),
                    "output_codec": live_output_codec(config),
                    "frame_quality": live_jpeg_quality(config),
                    "jpeg_quality": live_jpeg_quality(config),
                })
                geometry_logged = True
            try:
                with ENGINE_LOCK:
                    output, frame_timings = live_process_frame(engine, process_frame.copy(), config, live_state)
                processed_at = time.monotonic()
            except Exception as exc:
                await websocket.send_json({"error": f"live frame failed: {exc}"})
                continue
            if output is None:
                output = process_frame
            if output.shape[:2] != (process_height, process_width):
                output = cv2.resize(output, (process_width, process_height), interpolation=cv2.INTER_LINEAR)
            ok, encoded, encoded_codec = live_encode_frame(output, config)
            encoded_at = time.monotonic()
            if ok:
                out_bytes = int(encoded.size)
                perf_frames += 1
                perf_wait += received_at - frame_started
                perf_decode += decoded_at - received_at
                perf_resize += resized_at - decoded_at
                perf_process += processed_at - resized_at
                perf_detect += float(frame_timings.get("detect", 0.0))
                perf_landmarks += float(frame_timings.get("landmarks", 0.0))
                perf_swap += float(frame_timings.get("swap", 0.0))
                perf_post += float(frame_timings.get("post", 0.0))
                perf_enhance += float(frame_timings.get("enhance", 0.0))
                perf_detect_reused += 1 if frame_timings.get("detect_reused") else 0
                perf_faces += int(frame_timings.get("faces", 0) or 0)
                perf_encode += encoded_at - processed_at
                perf_in_bytes += len(payload)
                perf_out_bytes += out_bytes
                elapsed = encoded_at - perf_started
                if elapsed >= 5.0 and perf_frames:
                    await websocket.send_json({
                        "status": "live_perf",
                        "api_version": API_VERSION,
                        "server_fps": round(perf_frames / elapsed, 2),
                        "wait_ms": round((perf_wait / perf_frames) * 1000.0, 1),
                        "decode_ms": round((perf_decode / perf_frames) * 1000.0, 1),
                        "resize_ms": round((perf_resize / perf_frames) * 1000.0, 1),
                        "process_ms": round((perf_process / perf_frames) * 1000.0, 1),
                        "detect_ms": round((perf_detect / perf_frames) * 1000.0, 1),
                        "landmarks_ms": round((perf_landmarks / perf_frames) * 1000.0, 1),
                        "swap_ms": round((perf_swap / perf_frames) * 1000.0, 1),
                        "post_ms": round((perf_post / perf_frames) * 1000.0, 1),
                        "enhance_ms": round((perf_enhance / perf_frames) * 1000.0, 1),
                        "detect_reuse_pct": round((perf_detect_reused / perf_frames) * 100.0, 1),
                        "faces": round(perf_faces / perf_frames, 2),
                        "detector_size": live_detection_size(config),
                        "detect_every_n": int_config(config, "detect_every_n", 1, 1, 30),
                        "face_model_pack": live_face_model_pack(config),
                        "swapper_precision": live_swapper_precision(config),
                        "swapper_loaded_precision": live_swapper_diagnostics(engine).get("loaded_precision", ""),
                        "cache_source_face": getattr(engine, "cache_source_face", True),
                        "frame_codec": live_frame_codec(config),
                        "output_codec": live_output_codec(config),
                        "encoded_codec": encoded_codec,
                        "frame_quality": live_jpeg_quality(config),
                        "jpeg_quality": live_jpeg_quality(config),
                        "encode_ms": round((perf_encode / perf_frames) * 1000.0, 1),
                        "in_kb": round((perf_in_bytes / perf_frames) / 1024.0, 1),
                        "out_kb": round((perf_out_bytes / perf_frames) / 1024.0, 1),
                    })
                    perf_started = encoded_at
                    perf_frames = 0
                    perf_wait = 0.0
                    perf_decode = 0.0
                    perf_resize = 0.0
                    perf_process = 0.0
                    perf_detect = 0.0
                    perf_landmarks = 0.0
                    perf_swap = 0.0
                    perf_post = 0.0
                    perf_enhance = 0.0
                    perf_detect_reused = 0
                    perf_faces = 0
                    perf_encode = 0.0
                    perf_in_bytes = 0
                    perf_out_bytes = 0
                await websocket.send_bytes(encoded.tobytes())
    except WebSocketDisconnect:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Deep-Live-Cam Remote Colab API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    import uvicorn
    ensure_drive_layout()
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
