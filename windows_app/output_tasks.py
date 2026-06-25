from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QImage, QImageReader, QPixmap
from PySide6.QtWidgets import QFileDialog, QListWidgetItem

from windows_app import app_base as base

DOWNLOAD_CHUNK_SIZE = 64 * 1024  # 64KB for smooth progress updates


class OutputTaskWorker(QThread):
    succeeded = Signal(str, object)
    failed = Signal(str, str)
    progress = Signal(str, int, int)  # task_id, current, total

    def __init__(self, task_id: str, task: Callable[[], object]):
        super().__init__()
        self.task_id = task_id
        self.task = task

    def run(self) -> None:
        try:
            self.succeeded.emit(self.task_id, self.task())
        except Exception as exc:
            self.failed.emit(self.task_id, str(exc))

    def report_progress(self, current: int, total: int) -> None:
        self.progress.emit(self.task_id, current, total)


def _download_file_fast(
    client: base.ApiClient,
    path: str,
    destination: Path,
    timeout: float = 900.0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with base.urllib.request.urlopen(client.url(path), timeout=timeout) as response, destination.open("wb") as handle:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total_size > 0:
                progress_callback(downloaded, total_size)
    return destination


def _download_bytes_with_progress(
    client: base.ApiClient,
    path: str,
    timeout: float = 20.0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> bytes:
    with base.urllib.request.urlopen(client.url(path), timeout=timeout) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        chunks = []
        downloaded = 0
        while True:
            chunk = response.read(DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
            downloaded += len(chunk)
            if progress_callback:
                progress_callback(downloaded, total_size)
        return b"".join(chunks)


def _ensure_output_worker_state(window: base.MainWindow) -> None:
    if not hasattr(window, "output_workers"):
        window.output_workers = {}
    if not hasattr(window, "output_refresh_task_id"):
        window.output_refresh_task_id = ""
    if not hasattr(window, "output_preview_task_id"):
        window.output_preview_task_id = ""
    if not hasattr(window, "output_download_task_id"):
        window.output_download_task_id = ""
    if not hasattr(window, "output_health_task_id"):
        window.output_health_task_id = ""
    if not hasattr(window, "output_batch_task_id"):
        window.output_batch_task_id = ""


def _start_output_task(
    window: base.MainWindow,
    status: str,
    task: Callable[[], object],
    on_success: Callable[[str, object], None],
    on_failure: Callable[[str, str], None],
    on_progress: Callable[[str, int, int], None] | None = None,
) -> str:
    _ensure_output_worker_state(window)
    task_id = uuid.uuid4().hex
    worker = OutputTaskWorker(task_id, task)
    window.output_workers[task_id] = worker
    window.output_status.setText(status)
    worker.succeeded.connect(on_success)
    worker.failed.connect(on_failure)
    if on_progress:
        worker.progress.connect(on_progress)
    worker.finished.connect(lambda task_id=task_id: window.output_workers.pop(task_id, None))
    worker.start()
    return task_id


def _start_output_task_with_progress(
    window: base.MainWindow,
    status: str,
    task_factory: Callable[[Callable[[int, int], None]], object],
    on_success: Callable[[str, object], None],
    on_failure: Callable[[str, str], None],
) -> str:
    """Start a task that can report progress via a callback."""
    _ensure_output_worker_state(window)
    task_id = uuid.uuid4().hex

    def on_progress(_tid: str, current: int, total: int) -> None:
        if not hasattr(window, "outputs_progress"):
            return
        window.outputs_progress.show()
        if total > 0:
            pct = int(current / total * 100)
            window.outputs_progress.setMaximum(100)
            window.outputs_progress.setValue(pct)
            window.output_status.setText(f"Loading... {pct}%")
        else:
            # Unknown total size - show indeterminate with bytes downloaded
            window.outputs_progress.setMaximum(0)
            window.output_status.setText(f"Loading... {base.format_size(current)}")
        window.outputs_progress.repaint()
        base.QApplication.processEvents()

    # Create a mutable container for worker reference
    worker_holder: list[OutputTaskWorker] = []

    def progress_callback(current: int, total: int) -> None:
        if worker_holder:
            worker_holder[0].report_progress(current, total)

    def wrapped_task() -> object:
        return task_factory(progress_callback)

    worker = OutputTaskWorker(task_id, wrapped_task)
    worker_holder.append(worker)
    window.output_workers[task_id] = worker
    window.output_status.setText(status)
    worker.succeeded.connect(on_success)
    worker.failed.connect(on_failure)
    # Use QueuedConnection to ensure signal is processed in main thread
    worker.progress.connect(on_progress, Qt.QueuedConnection)
    worker.finished.connect(lambda task_id=task_id: window.output_workers.pop(task_id, None))
    worker.start()
    return task_id


def _copy_settings(settings: base.AppSettings) -> base.AppSettings:
    return base.AppSettings(**base.asdict(settings))


def _source_upload_path(path: Path) -> tuple[Path, str | None]:
    if path.suffix.lower() not in base.PHOTO_EXTENSIONS:
        return path, None
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        return path, None
    output_dir = Path(tempfile.gettempdir()) / "deep_live_cam_remote_sources"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{path.stem}_{uuid.uuid4().hex[:8]}.png"
    if image.save(str(output), "PNG"):
        return output, f"normalized source image orientation for upload: {output}"
    return path, None


def _prepare_and_start_batch(settings: base.AppSettings, kind: str) -> dict[str, Any]:
    client = base.ApiClient(settings)
    logs: list[str] = ["checking Colab API before starting batch"]
    client.request_json("GET", "/health", timeout=5.0)

    source_face = settings.source_face
    if base.is_local_path(source_face):
        source_path = Path(source_face)
        if not source_path.is_file():
            raise FileNotFoundError(f"Local source face does not exist: {source_face}")
        upload_path, normalization_log = _source_upload_path(source_path)
        if normalization_log:
            logs.append(normalization_log)
        logs.append(f"uploading local source face: {source_path}")
        response = client.upload_file("/upload/file?kind=source", upload_path, timeout=30.0)
        source_face = str(response.get("path") or source_face)
        logs.append(f"source uploaded to: {source_face}")

    input_path = settings.photos_input if kind == "photos" else settings.videos_input
    output_path = settings.photos_output if kind == "photos" else settings.videos_output
    input_dir = input_path
    output_dir = output_path
    if base.is_local_path(input_path):
        extensions = base.PHOTO_EXTENSIONS if kind == "photos" else base.VIDEO_EXTENSIONS
        files = base.local_files(input_path, extensions, settings.recursive)
        if not files:
            raise FileNotFoundError(f"No supported {kind} files found in local path: {input_path}")
        logs.append(f"uploading {len(files)} local {kind} file(s)")
        response = client.upload_files(f"/upload/{kind}", files, timeout=600.0)
        input_dir = str(response.get("input_dir") or input_path)
        if base.is_local_path(output_path):
            output_dir = str(response.get("output_dir") or output_path)
            logs.append(f"local output path is not reachable from Colab; using: {output_dir}")
        logs.append(f"{kind} uploaded to: {input_dir}")

    endpoint = "/jobs/photos" if kind == "photos" else "/jobs/videos"
    payload = base.job_payload(settings, input_dir, output_dir, source_face)
    response = client.request_json("POST", endpoint, payload, timeout=10.0)
    logs.append(f"started {endpoint}: {response}")
    return {"endpoint": endpoint, "response": response, "logs": logs}


def _start_batch(self: base.MainWindow, kind: str) -> None:
    self.sync_settings()
    self.tabs.setCurrentWidget(self.log_box)
    _ensure_output_worker_state(self)
    settings = _copy_settings(self.settings)
    self.log(f"starting {kind} batch...")

    def task() -> dict[str, Any]:
        return _prepare_and_start_batch(settings, kind)

    def succeeded(task_id: str, result: object) -> None:
        if task_id != self.output_batch_task_id:
            return
        payload = result if isinstance(result, dict) else {}
        for line in payload.get("logs") or []:
            self.log(str(line))
        response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
        self.active_job_id = response.get("job_id")
        if self.active_job_id:
            if self.poller:
                self.poller.stop()
            self.poller = base.PollWorker(self.client, self.active_job_id)
            self.poller.message.connect(self.log)
            self.poller.finished_status.connect(lambda status: self.log(f"job finished: {status}"))
            self.poller.start()
        self.output_status.setText(f"{kind} batch started")

    def failed(task_id: str, error: str) -> None:
        if task_id != self.output_batch_task_id:
            return
        self.output_status.setText(f"{kind} batch failed before start: {error}")
        self.log(f"{kind} batch failed before start: {error}")

    self.output_batch_task_id = _start_output_task(self, f"Starting {kind} batch...", task, succeeded, failed)


def start_photos(self: base.MainWindow) -> None:
    _start_batch(self, "photos")


def start_videos(self: base.MainWindow) -> None:
    _start_batch(self, "videos")


def refresh_outputs(self: base.MainWindow) -> None:
    self.sync_settings()
    _ensure_output_worker_state(self)
    kind = self.outputs_kind.currentText()
    self.outputs_list.clear()
    self.outputs_list.setEnabled(False)
    self.output_files = []
    self.output_current_loaded = False
    self.stop_output_video()
    self.output_preview.setText("Loading outputs...")
    # Indeterminate progress bar during network fetch (min=max=0)
    if hasattr(self, "outputs_progress"):
        self.outputs_progress.setMinimum(0)
        self.outputs_progress.setMaximum(0)
        self.outputs_progress.show()
        self.outputs_progress.repaint()
        base.QApplication.processEvents()

    def fetch() -> dict[str, Any]:
        return self.client.request_json("GET", f"/outputs/{kind}", timeout=30.0)

    def succeeded(task_id: str, payload: object) -> None:
        if task_id != self.output_refresh_task_id:
            return
        self.outputs_list.setEnabled(True)
        self.output_files = list((payload if isinstance(payload, dict) else {}).get("files") or [])
        total = len(self.output_files)
        has_progress = hasattr(self, "outputs_progress")
        # Switch to determinate mode for list population
        if has_progress:
            self.outputs_progress.setMaximum(100)
            self.outputs_progress.setValue(0)
            self.outputs_progress.repaint()
        for idx, item in enumerate(self.output_files):
            label = f"[{item.get('source')}] {item.get('relative_path')} ({base.format_size(item.get('size'))})"
            self.outputs_list.addItem(QListWidgetItem(label))
            if has_progress:
                progress = int((idx + 1) / total * 100) if total > 0 else 0
                self.outputs_progress.setValue(progress)
            self.output_status.setText(f"Loading... {idx + 1}/{total}")
            # Update UI every 10 items or on last item
            if idx % 10 == 0 or idx == total - 1:
                if has_progress:
                    self.outputs_progress.repaint()
                base.QApplication.processEvents()
        if has_progress:
            self.outputs_progress.hide()
        self.output_status.setText(f"{len(self.output_files)} {kind} output file(s)")
        if self.output_files:
            self.outputs_list.setCurrentRow(0)
        else:
            self.output_preview.setPixmap(QPixmap())
            self.output_preview.setText("No remote outputs found")

    def failed(task_id: str, error: str) -> None:
        if task_id != self.output_refresh_task_id:
            return
        self.outputs_list.setEnabled(True)
        if hasattr(self, "outputs_progress"):
            self.outputs_progress.hide()
        self.output_status.setText(f"refresh failed: {error}")
        self.log(f"outputs refresh failed: {error}")

    self.output_refresh_task_id = _start_output_task(self, "Refreshing outputs...", fetch, succeeded, failed)


def show_output_at(self: base.MainWindow, index: int) -> None:
    if index < 0 or index >= len(self.output_files):
        return
    self.output_current_loaded = False
    _ensure_output_worker_state(self)
    item = dict(self.output_files[index])
    kind = self.outputs_kind.currentText()
    path = str(item.get("download_path") or "")
    file_size = int(item.get("size") or 0)
    if not path:
        self.output_status.setText("selected output has no download path")
        self.output_current_loaded = True
        return
    self.stop_output_video()
    if kind == "photos":
        self.output_preview.setPixmap(QPixmap())
        size_str = base.format_size(file_size) if file_size > 0 else ""
        self.output_preview.setText(f"Loading photo preview... {size_str}")
        # Show progress bar (starts at 0)
        if hasattr(self, "outputs_progress"):
            self.outputs_progress.setMinimum(0)
            self.outputs_progress.setMaximum(100)
            self.outputs_progress.setValue(0)
            self.outputs_progress.show()
            self.outputs_progress.repaint()

        def fetch_photo(progress_cb: Callable[[int, int], None]) -> bytes:
            return _download_bytes_with_progress(self.client, path, timeout=20.0, progress_callback=progress_cb)

        def photo_ready(task_id: str, data: object) -> None:
            if task_id != self.output_preview_task_id:
                return
            if hasattr(self, "outputs_progress"):
                self.outputs_progress.hide()
            if self.output_video is not None:
                self.output_video.hide()
            self.output_preview.show()
            image = QImage.fromData(data if isinstance(data, bytes) else bytes(data))
            if image.isNull():
                self.output_status.setText("preview failed: downloaded image could not be decoded")
                self.output_current_loaded = True
                return
            pixmap = QPixmap.fromImage(image).scaled(self.output_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.output_preview.setPixmap(pixmap)
            self.output_status.setText(f"Showing {item.get('relative_path')} from {item.get('source')}")
            self.output_current_loaded = True

        def photo_failed(task_id: str, error: str) -> None:
            if task_id != self.output_preview_task_id:
                return
            if hasattr(self, "outputs_progress"):
                self.outputs_progress.hide()
            self.output_status.setText(f"preview failed: {error}")
            self.log(f"output preview failed: {error}")
            self.output_current_loaded = True

        self.output_preview_task_id = _start_output_task_with_progress(self, "Loading photo preview...", fetch_photo, photo_ready, photo_failed)
        return

    self.show_video_output(item)


def show_video_output(self: base.MainWindow, item: dict[str, Any]) -> None:
    _ensure_output_worker_state(self)
    path = str(item.get("download_path") or "")
    file_size = int(item.get("size") or 0)
    relative = str(item.get("relative_path") or item.get("name") or "output.mp4")
    safe_relative = relative.replace("/", "_").replace("\\", "_")
    local_name = f"{item.get('source', 'output')}_{safe_relative}"
    local_path = self.output_temp_dir / local_name
    self.output_preview.setPixmap(QPixmap())
    size_str = base.format_size(file_size) if file_size > 0 else ""
    self.output_preview.setText(f"Loading video preview:\n{relative}\n{size_str}")
    # Show progress bar (starts at 0)
    if hasattr(self, "outputs_progress"):
        self.outputs_progress.setMinimum(0)
        self.outputs_progress.setMaximum(100)
        self.outputs_progress.setValue(0)
        self.outputs_progress.show()
        self.outputs_progress.repaint()

    def fetch_video(progress_cb: Callable[[int, int], None]) -> dict[str, str]:
        if not local_path.exists() or local_path.stat().st_size != file_size:
            _download_file_fast(self.client, path, local_path, timeout=900.0, progress_callback=progress_cb)
        return {"relative": relative, "local_path": str(local_path)}

    def video_ready(task_id: str, result: object) -> None:
        if task_id != self.output_preview_task_id:
            return
        if hasattr(self, "outputs_progress"):
            self.outputs_progress.hide()
        payload = result if isinstance(result, dict) else {}
        ready_relative = str(payload.get("relative") or relative)
        ready_path = Path(str(payload.get("local_path") or local_path))
        if self.output_player is None or self.output_video is None:
            self.output_preview.show()
            self.output_preview.setText(
                f"Video ready to download:\n{ready_relative}\n\nInstall PySide6 multimedia support for inline playback."
            )
            self.output_status.setText(f"Selected video {ready_relative}")
            self.output_current_loaded = True
            return
        self.output_preview.hide()
        self.output_video.show()
        self.output_player.setSource(QUrl.fromLocalFile(str(ready_path)))
        self.output_player.play()
        self.output_status.setText(f"Playing {ready_relative}")
        self.output_current_loaded = True

    def video_failed(task_id: str, error: str) -> None:
        if task_id != self.output_preview_task_id:
            return
        if hasattr(self, "outputs_progress"):
            self.outputs_progress.hide()
        self.output_status.setText(f"preview failed: {error}")
        self.log(f"output preview failed: {error}")
        self.output_current_loaded = True

    self.output_preview_task_id = _start_output_task_with_progress(self, f"Loading video preview: {relative}", fetch_video, video_ready, video_failed)


def download_current_output(self: base.MainWindow) -> None:
    item = self.current_output()
    if not item:
        self.output_status.setText("No output selected")
        return
    folder = QFileDialog.getExistingDirectory(self, "Download selected output to folder")
    if not folder:
        return
    item = dict(item)
    destination = Path(folder) / str(item.get("name") or Path(str(item.get("relative_path"))).name)

    def download() -> str:
        return str(_download_file_fast(self.client, str(item.get("download_path")), destination))

    def succeeded(task_id: str, result: object) -> None:
        if task_id != self.output_download_task_id:
            return
        self.output_status.setText(f"Downloaded to {result}")
        self.log(f"downloaded output: {result}")

    def failed(task_id: str, error: str) -> None:
        if task_id != self.output_download_task_id:
            return
        self.output_status.setText(f"download failed: {error}")
        self.log(f"download failed: {error}")

    self.output_download_task_id = _start_output_task(self, f"Downloading {destination.name}...", download, succeeded, failed)


def download_all_outputs(self: base.MainWindow) -> None:
    if not self.output_files:
        self.output_status.setText("No outputs to download")
        return
    folder = QFileDialog.getExistingDirectory(self, "Download all listed outputs to folder")
    if not folder:
        return
    kind = self.outputs_kind.currentText()
    destination = Path(folder) / f"{kind}_outputs.zip"

    def download_all() -> str:
        return str(_download_file_fast(self.client, f"/outputs/{kind}/zip", destination, timeout=1800.0))

    def succeeded(task_id: str, result: object) -> None:
        if task_id != self.output_download_task_id:
            return
        self.output_status.setText(f"Downloaded ZIP to {result}")
        self.log(f"downloaded {kind} outputs ZIP to {result}")

    def failed(task_id: str, error: str) -> None:
        if task_id != self.output_download_task_id:
            return
        self.output_status.setText(f"download all failed: {error}")
        self.log(f"download all failed: {error}")

    self.output_download_task_id = _start_output_task(self, f"Downloading {kind} ZIP...", download_all, succeeded, failed)


def check_connection(self: base.MainWindow) -> None:
    self.sync_settings()
    self.tabs.setCurrentWidget(self.log_box)
    _ensure_output_worker_state(self)
    self.log("checking connection...")

    def fetch_health() -> dict[str, Any]:
        return self.client.request_json("GET", "/health", timeout=5.0)

    def succeeded(task_id: str, payload: object) -> None:
        if task_id != self.output_health_task_id:
            return
        self.log("health: " + base.json.dumps(payload, indent=2))

    def failed(task_id: str, error: str) -> None:
        if task_id != self.output_health_task_id:
            return
        self.log(f"health failed: {error}")

    self.output_health_task_id = _start_output_task(self, "Checking connection...", fetch_health, succeeded, failed)


class OutputTasksMixin:
    check_connection = check_connection
    start_photos = start_photos
    start_videos = start_videos
    refresh_outputs = refresh_outputs
    show_output_at = show_output_at
    show_video_output = show_video_output
    download_current_output = download_current_output
    download_all_outputs = download_all_outputs
main = base.main