from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import mimetypes
import queue
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

OUTPUT_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
OUTPUT_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


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


@app.get("/health")
def health() -> dict[str, Any]:
    paths = ensure_drive_layout()
    return {
        "ok": True,
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
    config_payload = await websocket.receive_text()
    config = json.loads(config_payload)
    process_config = colab_batch.ProcessConfig(
        input_dir=PHOTOS_DIR,
        output_dir=OUTPUT_PHOTOS_DIR,
        source_face=Path(config.get("source_face") or SOURCE_DIR / "source.png"),
        map_config=None,
        many_faces=bool(config.get("many_faces", False)),
        enhancer=config.get("enhancer", "none"),
    )
    with ENGINE_LOCK:
        engine = colab_batch.ModernEngine(process_config)
    await websocket.send_json({"status": "ready"})
    try:
        while True:
            payload = await websocket.receive_bytes()
            array = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"error": "invalid frame"})
                continue
            with ENGINE_LOCK:
                output = engine.process(frame.copy(), "live")
            ok, encoded = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), int(config.get("jpeg_quality", 80))])
            if ok:
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