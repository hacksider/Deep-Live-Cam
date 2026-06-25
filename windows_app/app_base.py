from __future__ import annotations

import asyncio
import ctypes
import json
import mimetypes
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
except Exception:
    QAudioOutput = None
    QMediaPlayer = None
    QVideoWidget = None


DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/DeepLiveCamRemote"
APP_STATE = Path.home() / ".deep_live_cam_remote_windows_app.json"
REMOTE_PREFIXES = ("/content/", "/drive/")
PHOTO_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass
class AppSettings:
    host: str = ""
    port: int = 7860
    drive_root: str = DEFAULT_DRIVE_ROOT
    source_face: str = DEFAULT_DRIVE_ROOT + "/source/source.png"
    photos_input: str = DEFAULT_DRIVE_ROOT + "/photos"
    photos_output: str = DEFAULT_DRIVE_ROOT + "/outputs/photos"
    videos_input: str = DEFAULT_DRIVE_ROOT + "/videos"
    videos_output: str = DEFAULT_DRIVE_ROOT + "/outputs/videos"
    recursive: bool = True
    overwrite: bool = False
    skip_processed: bool = True
    many_faces: bool = False
    enhancer: str = "none"
    opacity: float = 1.0
    sharpness: float = 0.0
    mouth_mask_size: float = 0.0
    interpolation_weight: float = 0.0
    poisson_blend: bool = False
    color_correction: bool = False
    max_fps: float = 30.0
    max_width: int = 420
    quality: int = 18
    start_pct: float = 0.0
    end_pct: float = 100.0
    camera_index: int = 0
    virtual_camera: str = "OBS Virtual Camera"

    @property
    def base_url(self) -> str:
        host = self.host.replace("http://", "").replace("https://", "").strip().strip("/")
        return f"http://{host}:{self.port}"


def load_settings() -> AppSettings:
    if APP_STATE.is_file():
        try:
            data = json.loads(APP_STATE.read_text(encoding="utf-8"))
            return AppSettings(**{**asdict(AppSettings()), **data})
        except Exception:
            pass
    return AppSettings()


def save_settings(settings: AppSettings) -> None:
    APP_STATE.write_text(json.dumps(asdict(settings), indent=2) + "\n", encoding="utf-8")


def is_local_path(path: str) -> bool:
    if not path:
        return False
    normalized = path.replace("\\", "/")
    if normalized.startswith(REMOTE_PREFIXES):
        return False
    if len(path) >= 2 and path[1] == ":":
        return True
    if path.startswith("\\\\"):
        return True
    return Path(path).exists()


def local_files(path: str, extensions: set[str], recursive: bool) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root] if root.suffix.lower() in extensions else []
    if not root.is_dir():
        raise FileNotFoundError(f"Local input folder does not exist: {path}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in extensions)


def format_size(size: int | None) -> str:
    if size is None:
        return ""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size} B"


def check_tailscale_cli() -> bool:
    """Check if tailscale CLI is available on PATH."""
    try:
        subprocess.run(
            ["tailscale", "version"],
            capture_output=True,
            timeout=3.0,
            check=False,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def taildrop_receive_file(remote_host: str, remote_path: str, local_destination: Path) -> tuple[bool, str]:
    """
    Attempt to receive file via Taildrop.

    Returns: (success: bool, message: str)
    """
    try:
        local_destination.parent.mkdir(parents=True, exist_ok=True)

        # tailscale file cp <remote-host>:<remote-path> <local-destination>
        result = subprocess.run(
            ["tailscale", "file", "cp", f"{remote_host}:{remote_path}", str(local_destination)],
            capture_output=True,
            text=True,
            timeout=600.0,  # 10 minute timeout for large files
            check=False,
        )

        if result.returncode == 0:
            return (True, f"Taildrop transfer complete: {local_destination}")
        else:
            error_msg = result.stderr.strip() if result.stderr else f"exit code {result.returncode}"
            return (False, f"Taildrop failed: {error_msg}")

    except subprocess.TimeoutExpired:
        return (False, "Taildrop transfer timed out (10 min)")
    except Exception as exc:
        return (False, f"Taildrop error: {exc}")


class ApiClient:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def url(self, path: str) -> str:
        return self.settings.base_url + urllib.parse.quote(path, safe="/:?=&%")

    def request_json(self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url(path),
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def download_bytes(self, path: str, timeout: float = 120.0) -> bytes:
        with urllib.request.urlopen(self.url(path), timeout=timeout) as response:
            return response.read()

    def download_file(self, path: str, destination: Path, timeout: float = 600.0) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(self.url(path), timeout=timeout) as response, destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        return destination

    def create_zip(self, kind: str, timeout: float = 120.0) -> dict[str, Any]:
        """Call /create-zip endpoint to prepare archive for transfer."""
        return self.request_json("POST", f"/outputs/{kind}/create-zip", timeout=timeout)

    def download_archive(self, archive_id: str, destination: Path, timeout: float = 1800.0) -> Path:
        """HTTP fallback download for pre-created archives."""
        return self.download_file(f"/download-archive/{archive_id}", destination, timeout=timeout)

    def upload_file(self, endpoint: str, file_path: Path, field_name: str = "file", timeout: float = 120.0) -> dict[str, Any]:
        boundary = f"----DeepLiveCamBoundary{uuid.uuid4().hex}"
        content_type = f"multipart/form-data; boundary={boundary}"
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        file_data = file_path.read_bytes()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")
        request = urllib.request.Request(
            self.url(endpoint),
            data=body,
            method="POST",
            headers={"Content-Type": content_type},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def upload_files(self, endpoint: str, file_paths: list[Path], field_name: str = "files", timeout: float = 300.0) -> dict[str, Any]:
        boundary = f"----DeepLiveCamBoundary{uuid.uuid4().hex}"
        content_type = f"multipart/form-data; boundary={boundary}"
        body_parts = []
        for file_path in file_paths:
            mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            file_data = file_path.read_bytes()
            body_parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'
                    f"Content-Type: {mime_type}\r\n\r\n"
                ).encode("utf-8") + file_data + b"\r\n"
            )
        body = b"".join(body_parts) + f"--{boundary}--\r\n".encode("utf-8")
        request = urllib.request.Request(
            self.url(endpoint),
            data=body,
            method="POST",
            headers={"Content-Type": content_type},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def job_payload(settings: AppSettings, input_dir: str, output_dir: str, source_face: str | None = None) -> dict[str, Any]:
    return {
        "source_face": source_face or settings.source_face,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "recursive": settings.recursive,
        "overwrite": settings.overwrite,
        "skip_processed": settings.skip_processed,
        "many_faces": settings.many_faces,
        "enhancer": settings.enhancer,
        "opacity": settings.opacity,
        "sharpness": settings.sharpness,
        "mouth_mask_size": settings.mouth_mask_size,
        "interpolation_weight": settings.interpolation_weight,
        "poisson_blend": settings.poisson_blend,
        "color_correction": settings.color_correction,
        "max_fps": settings.max_fps,
        "max_width": settings.max_width,
        "quality": settings.quality,
        "start_pct": settings.start_pct,
        "end_pct": settings.end_pct,
    }


class LiveWorker(QThread):
    message = Signal(str)
    frame = Signal(bytes)

    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            asyncio.run(self._run_live())
        except Exception as exc:
            self.message.emit(f"live stopped: {exc}")

    async def _run_live(self) -> None:
        import cv2
        import websockets
        uri = self.settings.base_url.replace("http://", "ws://") + "/ws/live"
        self.message.emit(f"connecting live websocket: {uri}")
        cap = cv2.VideoCapture(self.settings.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera index {self.settings.camera_index}")
        virtual_cam = None
        try:
            async with websockets.connect(uri, max_size=8 * 1024 * 1024) as websocket:
                await websocket.send(json.dumps({"source_face": self.settings.source_face, "enhancer": self.settings.enhancer, "many_faces": self.settings.many_faces, "jpeg_quality": 80}))
                ready = await websocket.recv()
                self.message.emit(f"live backend: {ready}")
                while not self._stop:
                    ok, frame = cap.read()
                    if not ok:
                        await asyncio.sleep(0.03)
                        continue
                    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not ok:
                        continue
                    await websocket.send(encoded.tobytes())
                    reply = await websocket.recv()
                    if isinstance(reply, str):
                        self.message.emit(reply)
                        continue
                    self.frame.emit(reply)
                    if virtual_cam is None:
                        try:
                            import pyvirtualcam
                            decoded = cv2.imdecode(__import__("numpy").frombuffer(reply, dtype=__import__("numpy").uint8), cv2.IMREAD_COLOR)
                            h, w = decoded.shape[:2]
                            virtual_cam = pyvirtualcam.Camera(width=w, height=h, fps=20, device=self.settings.virtual_camera or None)
                            self.message.emit(f"virtual camera opened: {virtual_cam.device}")
                        except Exception as exc:
                            self.message.emit(f"virtual camera unavailable: {exc}")
                            virtual_cam = False
                    if virtual_cam:
                        import numpy as np
                        decoded = cv2.imdecode(np.frombuffer(reply, dtype=np.uint8), cv2.IMREAD_COLOR)
                        rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                        virtual_cam.send(rgb)
                        virtual_cam.sleep_until_next_frame()
        finally:
            cap.release()
            if virtual_cam and hasattr(virtual_cam, "close"):
                virtual_cam.close()
            self.message.emit("live worker stopped")


class PollWorker(QThread):
    message = Signal(str)
    finished_status = Signal(str)

    def __init__(self, client: ApiClient, job_id: str):
        super().__init__()
        self.client = client
        self.job_id = job_id
        self._seen = 0
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        while not self._stop:
            try:
                payload = self.client.request_json("GET", f"/jobs/{self.job_id}", timeout=5)
                logs = payload.get("logs") or []
                for line in logs[self._seen:]:
                    self.message.emit(str(line))
                self._seen = len(logs)
                status = payload.get("status", "unknown")
                if status not in {"queued", "running"}:
                    self.finished_status.emit(status)
                    return
            except Exception as exc:
                self.message.emit(f"poll error: {exc}")
            time.sleep(1.0)


def set_dark_title_bar(window: QMainWindow) -> None:
    """Enable dark title bar on Windows 10/11."""
    if sys.platform != "win32":
        return
    try:
        hwnd = int(window.winId())
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        value = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, ctypes.byref(value), ctypes.sizeof(value)
        )
    except Exception:
        pass  # Silently fail on older Windows versions


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        set_dark_title_bar(self)
        icon_path = Path(__file__).parent / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.settings = load_settings()
        self.client = ApiClient(self.settings)
        self.poller: PollWorker | None = None
        self.live_worker: LiveWorker | None = None
        self.active_job_id: str | None = None
        self.output_files: list[dict[str, Any]] = []
        self.output_current_loaded = False
        self.output_temp_dir = Path(tempfile.gettempdir()) / "deep_live_cam_remote_outputs"
        self.output_temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_timer = QTimer(self)
        self.output_timer.timeout.connect(self.next_output)
        self.setWindowTitle("Deep-Live-Cam Remote Controller")
        self.resize(980, 760)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.log_box = QTextEdit(readOnly=True)
        self._build_setup_tab()
        self._build_photos_tab()
        self._build_videos_tab()
        self._build_outputs_tab()
        self._build_live_tab()
        self.tabs.addTab(self.log_box, "Logs")

    def log(self, text: str) -> None:
        self.log_box.append(text)

    def _browse_file(self, line_edit: QLineEdit, title: str, file_filter: str = "Images (*.png *.jpg *.jpeg *.bmp *.webp)") -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
        if path:
            line_edit.setText(path)

    def _browse_folder(self, line_edit: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title)
        if path:
            line_edit.setText(path)

    def _path_row(self, line_edit: QLineEdit, browse_callback: Any) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(line_edit)
        browse = QPushButton("Browse...")
        browse.clicked.connect(browse_callback)
        row.addWidget(browse)
        return row

    def sync_settings(self) -> None:
        self.settings.host = self.host.text().strip()
        self.settings.port = int(self.port.value())
        self.settings.drive_root = self.drive_root.text().strip()
        self.settings.source_face = self.source_face.text().strip()
        self.settings.photos_input = self.photos_input.text().strip()
        self.settings.photos_output = self.photos_output.text().strip()
        self.settings.videos_input = self.videos_input.text().strip()
        self.settings.videos_output = self.videos_output.text().strip()
        self.settings.recursive = self.recursive.isChecked()
        self.settings.overwrite = self.overwrite.isChecked()
        self.settings.skip_processed = self.skip_processed.isChecked()
        self.settings.many_faces = self.many_faces.isChecked()
        self.settings.enhancer = self.enhancer.currentText()
        self.settings.opacity = float(self.opacity.value())
        self.settings.sharpness = float(self.sharpness.value())
        self.settings.mouth_mask_size = float(self.mouth_mask_size.value())
        self.settings.interpolation_weight = float(self.interpolation_weight.value())
        self.settings.poisson_blend = self.poisson_blend.isChecked()
        self.settings.color_correction = self.color_correction.isChecked()
        self.settings.max_fps = float(self.max_fps.value())
        self.settings.max_width = int(self.max_width.value())
        self.settings.quality = int(self.quality.value())
        self.settings.start_pct = float(self.start_pct.value())
        self.settings.end_pct = float(self.end_pct.value())
        self.settings.camera_index = int(self.camera_index.value())
        self.settings.virtual_camera = self.virtual_camera.text().strip()
        save_settings(self.settings)

    def _build_setup_tab(self) -> None:
        tab = QWidget(); layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.host = QLineEdit(self.settings.host)
        self.port = QSpinBox(); self.port.setRange(1, 65535); self.port.setValue(self.settings.port)
        self.drive_root = QLineEdit(self.settings.drive_root)
        form.addRow("Tailscale host/IP", self.host)
        form.addRow("API port", self.port)
        form.addRow("Drive root", self.drive_root)
        layout.addLayout(form)
        self.setup_help = QTextEdit(readOnly=True)
        self.setup_help.setObjectName("helpText")
        self.setup_help.setPlainText(
            "Colab setup checklist:\n"
            "1. Open google-colab/Deep_Live_Cam_Remote_Batch.ipynb.\n"
            "2. Run Install and initialize.\n"
            "3. Mount Drive and use /content/drive/MyDrive/DeepLiveCamRemote.\n"
            "4. Run the Remote API server cell.\n"
            "5. Start Tailscale in Colab and copy the Tailscale IP here.\n"
        )
        layout.addWidget(self.setup_help)
        row = QHBoxLayout()
        btn = QPushButton("Check connection")
        btn.setObjectName("primaryButton")
        btn.clicked.connect(self.check_connection)
        save = QPushButton("Save settings")
        save.setObjectName("successButton")
        save.clicked.connect(lambda: (self.sync_settings(), self.log("settings saved")))
        row.addWidget(btn); row.addWidget(save); row.addStretch(1)
        layout.addLayout(row)
        self.tabs.addTab(tab, "Setup")

    def _common_group(self) -> QGroupBox:
        box = QGroupBox("Common options")
        form = QFormLayout(box)
        self.source_face = QLineEdit(self.settings.source_face)
        source_row = self._path_row(self.source_face, lambda: self._browse_file(self.source_face, "Select source face image"))
        self.recursive = QCheckBox(); self.recursive.setChecked(self.settings.recursive)
        self.overwrite = QCheckBox(); self.overwrite.setChecked(self.settings.overwrite)
        self.skip_processed = QCheckBox(); self.skip_processed.setChecked(self.settings.skip_processed)
        self.many_faces = QCheckBox(); self.many_faces.setChecked(self.settings.many_faces)
        self.enhancer = QComboBox(); self.enhancer.addItems(["none", "gfpgan", "gpen256", "gpen512"]); self.enhancer.setCurrentText(self.settings.enhancer)
        self.opacity = QDoubleSpinBox(); self.opacity.setRange(0.0, 1.0); self.opacity.setSingleStep(0.1); self.opacity.setValue(self.settings.opacity)
        self.sharpness = QDoubleSpinBox(); self.sharpness.setRange(0.0, 1.0); self.sharpness.setSingleStep(0.1); self.sharpness.setValue(self.settings.sharpness)
        self.mouth_mask_size = QDoubleSpinBox(); self.mouth_mask_size.setRange(0.0, 10.0); self.mouth_mask_size.setSingleStep(0.5); self.mouth_mask_size.setValue(self.settings.mouth_mask_size)
        self.interpolation_weight = QDoubleSpinBox(); self.interpolation_weight.setRange(0.0, 1.0); self.interpolation_weight.setSingleStep(0.1); self.interpolation_weight.setValue(self.settings.interpolation_weight)
        self.poisson_blend = QCheckBox(); self.poisson_blend.setChecked(self.settings.poisson_blend)
        self.color_correction = QCheckBox(); self.color_correction.setChecked(self.settings.color_correction)
        form.addRow("Source face path", source_row)
        form.addRow("Recursive", self.recursive)
        form.addRow("Overwrite", self.overwrite)
        form.addRow("Skip processed", self.skip_processed)
        form.addRow("Many faces", self.many_faces)
        form.addRow("Enhancer", self.enhancer)
        form.addRow("Opacity (1=full)", self.opacity)
        form.addRow("Sharpness (0=off)", self.sharpness)
        form.addRow("Mouth mask (0=off)", self.mouth_mask_size)
        form.addRow("Interpolation (0=off)", self.interpolation_weight)
        form.addRow("Poisson blend", self.poisson_blend)
        form.addRow("Color correction", self.color_correction)
        return box

    def _build_photos_tab(self) -> None:
        tab = QWidget(); layout = QVBoxLayout(tab)
        layout.addWidget(self._common_group())
        form = QFormLayout()
        self.photos_input = QLineEdit(self.settings.photos_input)
        self.photos_output = QLineEdit(self.settings.photos_output)
        photos_input_row = self._path_row(self.photos_input, lambda: self._browse_folder(self.photos_input, "Select photos input folder"))
        form.addRow("Photos input path", photos_input_row)
        form.addRow("Photos output path", self.photos_output)
        layout.addLayout(form)
        btn = QPushButton("Start photo batch")
        btn.setObjectName("primaryButton")
        btn.clicked.connect(self.start_photos)
        layout.addWidget(btn); layout.addStretch(1)
        self.tabs.addTab(tab, "Photos")

    def _build_videos_tab(self) -> None:
        tab = QWidget(); layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.videos_input = QLineEdit(self.settings.videos_input)
        self.videos_output = QLineEdit(self.settings.videos_output)
        self.max_fps = QDoubleSpinBox(); self.max_fps.setRange(1, 120); self.max_fps.setValue(self.settings.max_fps)
        self.max_width = QSpinBox(); self.max_width.setRange(64, 4096); self.max_width.setValue(self.settings.max_width)
        self.quality = QSpinBox(); self.quality.setRange(0, 51); self.quality.setValue(self.settings.quality)
        videos_input_row = self._path_row(self.videos_input, lambda: self._browse_folder(self.videos_input, "Select videos input folder"))
        form.addRow("Videos input path", videos_input_row)
        form.addRow("Videos output path", self.videos_output)
        form.addRow("Max FPS", self.max_fps)
        form.addRow("Max width", self.max_width)
        form.addRow("Quality", self.quality)
        layout.addLayout(form)
        btn = QPushButton("Start video batch")
        btn.setObjectName("primaryButton")
        btn.clicked.connect(self.start_videos)
        cancel = QPushButton("Graceful cancel active job")
        cancel.setObjectName("dangerButton")
        cancel.clicked.connect(self.cancel_job)
        row = QHBoxLayout(); row.addWidget(btn); row.addWidget(cancel); row.addStretch(1)
        layout.addLayout(row); layout.addStretch(1)
        self.tabs.addTab(tab, "Videos")

    def _build_outputs_tab(self) -> None:
        tab = QWidget(); layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.outputs_kind = QComboBox(); self.outputs_kind.addItems(["photos", "videos"])
        refresh = QPushButton("Refresh")
        previous = QPushButton("Previous")
        next_button = QPushButton("Next")
        self.outputs_autoplay = QCheckBox("Auto-play")
        download_current = QPushButton("Download current")
        download_all = QPushButton("Download all")
        self.download_taildrop_btn = QPushButton("Download via Taildrop")
        self.download_taildrop_btn.setToolTip("Fast P2P transfer over Tailscale (falls back to HTTP if unavailable)")
        controls.addWidget(QLabel("Kind"))
        controls.addWidget(self.outputs_kind)
        controls.addWidget(refresh)
        controls.addWidget(previous)
        controls.addWidget(next_button)
        controls.addWidget(self.outputs_autoplay)
        controls.addWidget(download_current)
        controls.addWidget(download_all)
        controls.addWidget(self.download_taildrop_btn)
        # Disable if tailscale not available
        if not check_tailscale_cli():
            self.download_taildrop_btn.setEnabled(False)
            self.download_taildrop_btn.setToolTip("Tailscale CLI not found on PATH")
        controls.addStretch(1)
        layout.addLayout(controls)

        self.outputs_progress = QProgressBar()
        self.outputs_progress.setMaximum(100)
        self.outputs_progress.setFixedHeight(20)
        self.outputs_progress.setTextVisible(True)
        self.outputs_progress.hide()
        layout.addWidget(self.outputs_progress)

        # Connect signals AFTER all widgets are created
        refresh.clicked.connect(self.refresh_outputs)
        previous.clicked.connect(self.previous_output)
        next_button.clicked.connect(self.next_output)
        self.outputs_autoplay.stateChanged.connect(self.toggle_outputs_autoplay)
        self.outputs_kind.currentTextChanged.connect(lambda _text: self.refresh_outputs())
        download_current.clicked.connect(self.download_current_output)
        download_all.clicked.connect(self.download_all_outputs)
        self.download_taildrop_btn.clicked.connect(self.download_via_taildrop)

        self.outputs_list = QListWidget()
        self.outputs_list.currentRowChanged.connect(self.show_output_at)
        layout.addWidget(self.outputs_list)

        self.output_preview = QLabel("Refresh outputs to preview remote media")
        self.output_preview.setAlignment(Qt.AlignCenter)
        self.output_preview.setMinimumHeight(340)
        self.output_preview.setWordWrap(True)
        layout.addWidget(self.output_preview)

        self.output_video = None
        self.output_audio = None
        self.output_player = None
        if QMediaPlayer is not None and QVideoWidget is not None and QAudioOutput is not None:
            self.output_video = QVideoWidget()
            self.output_video.setMinimumHeight(340)
            self.output_audio = QAudioOutput(self)
            self.output_player = QMediaPlayer(self)
            self.output_player.setAudioOutput(self.output_audio)
            self.output_player.setVideoOutput(self.output_video)
            layout.addWidget(self.output_video)
            self.output_video.hide()

        self.output_status = QLabel("")
        self.output_status.setObjectName("statusLabel")
        self.output_status.setWordWrap(True)
        layout.addWidget(self.output_status)
        self.tabs.addTab(tab, "Outputs")

    def _build_live_tab(self) -> None:
        tab = QWidget(); layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.camera_index = QSpinBox(); self.camera_index.setRange(0, 20); self.camera_index.setValue(self.settings.camera_index)
        self.virtual_camera = QLineEdit(self.settings.virtual_camera)
        form.addRow("Camera index", self.camera_index)
        form.addRow("Virtual camera", self.virtual_camera)
        layout.addLayout(form)
        self.live_note = QLabel("Live sends webcam JPEG frames to ws://HOST:PORT/ws/live, previews returned frames, and opens the configured virtual camera when pyvirtualcam can find it.")
        self.live_note.setObjectName("statusLabel")
        self.live_note.setWordWrap(True)
        layout.addWidget(self.live_note)
        self.live_preview = QLabel("Live preview")
        self.live_preview.setAlignment(Qt.AlignCenter)
        self.live_preview.setMinimumHeight(360)
        layout.addWidget(self.live_preview)
        row = QHBoxLayout()
        start = QPushButton("Start live")
        start.setObjectName("successButton")
        stop = QPushButton("Stop live")
        stop.setObjectName("dangerButton")
        start.clicked.connect(self.start_live)
        stop.clicked.connect(self.stop_live)
        row.addWidget(start); row.addWidget(stop); row.addStretch(1)
        layout.addLayout(row); layout.addStretch(1)
        self.tabs.addTab(tab, "Live")

    def check_connection(self) -> None:
        self.sync_settings()
        self.tabs.setCurrentWidget(self.log_box)
        try:
            payload = self.client.request_json("GET", "/health")
            self.log("health: " + json.dumps(payload, indent=2))
        except Exception as exc:
            self.log(f"health failed: {exc}")

    def refresh_outputs(self) -> None:
        self.sync_settings()
        kind = self.outputs_kind.currentText()
        self.outputs_list.clear()
        self.output_files = []
        self.output_status.setText("Refreshing outputs...")
        self.stop_output_video()
        try:
            payload = self.client.request_json("GET", f"/outputs/{kind}", timeout=30.0)
            self.output_files = list(payload.get("files") or [])
            for item in self.output_files:
                label = f"[{item.get('source')}] {item.get('relative_path')} ({format_size(item.get('size'))})"
                self.outputs_list.addItem(QListWidgetItem(label))
            self.output_status.setText(f"{len(self.output_files)} {kind} output file(s)")
            if self.output_files:
                self.outputs_list.setCurrentRow(0)
            else:
                self.output_preview.setText("No remote outputs found")
        except Exception as exc:
            self.output_status.setText(f"refresh failed: {exc}")
            self.log(f"outputs refresh failed: {exc}")

    def show_output_at(self, index: int) -> None:
        if index < 0 or index >= len(self.output_files):
            return
        item = self.output_files[index]
        kind = self.outputs_kind.currentText()
        path = str(item.get("download_path") or "")
        if not path:
            self.output_status.setText("selected output has no download path")
            return
        try:
            if kind == "photos":
                self.stop_output_video()
                if self.output_video is not None:
                    self.output_video.hide()
                self.output_preview.show()
                data = self.client.download_bytes(path)
                image = QImage.fromData(data)
                if image.isNull():
                    raise ValueError("downloaded image could not be decoded")
                pixmap = QPixmap.fromImage(image).scaled(self.output_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.output_preview.setPixmap(pixmap)
                self.output_status.setText(f"Showing {item.get('relative_path')} from {item.get('source')}")
            else:
                self.show_video_output(item)
        except Exception as exc:
            self.output_status.setText(f"preview failed: {exc}")
            self.log(f"output preview failed: {exc}")

    def show_video_output(self, item: dict[str, Any]) -> None:
        path = str(item.get("download_path") or "")
        relative = str(item.get("relative_path") or item.get("name") or "output.mp4")
        safe_relative = relative.replace("/", "_").replace("\\", "_")
        local_name = f"{item.get('source', 'output')}_{safe_relative}"
        local_path = self.output_temp_dir / local_name
        self.output_status.setText(f"Loading video preview: {relative}")
        if not local_path.exists() or local_path.stat().st_size != int(item.get("size") or -1):
            self.client.download_file(path, local_path, timeout=900.0)
        if self.output_player is None or self.output_video is None:
            self.output_preview.show()
            self.output_preview.setText(f"Video ready to download:\n{relative}\n\nInstall PySide6 multimedia support for inline playback.")
            self.output_status.setText(f"Selected video {relative}")
            return
        self.output_preview.hide()
        self.output_video.show()
        self.output_player.setSource(QUrl.fromLocalFile(str(local_path)))
        self.output_player.play()
        self.output_status.setText(f"Playing {relative}")

    def stop_output_video(self) -> None:
        if self.output_player is not None:
            self.output_player.stop()
        if self.output_video is not None:
            self.output_video.hide()
        if hasattr(self, "output_preview"):
            self.output_preview.show()

    def current_output(self) -> dict[str, Any] | None:
        index = self.outputs_list.currentRow()
        if index < 0 or index >= len(self.output_files):
            return None
        return self.output_files[index]

    def previous_output(self) -> None:
        if not self.output_files:
            return
        index = self.outputs_list.currentRow()
        self.outputs_list.setCurrentRow((index - 1) % len(self.output_files))

    def next_output(self) -> None:
        if not self.output_files:
            return
        if self.outputs_autoplay.isChecked() and not self.output_current_loaded:
            return
        index = self.outputs_list.currentRow()
        self.outputs_list.setCurrentRow((index + 1) % len(self.output_files))

    def toggle_outputs_autoplay(self) -> None:
        if self.outputs_autoplay.isChecked():
            interval = 8000 if self.outputs_kind.currentText() == "videos" else 3500
            self.output_timer.start(interval)
        else:
            self.output_timer.stop()

    def download_current_output(self) -> None:
        item = self.current_output()
        if not item:
            self.output_status.setText("No output selected")
            return
        folder = QFileDialog.getExistingDirectory(self, "Download selected output to folder")
        if not folder:
            return
        try:
            destination = Path(folder) / str(item.get("name") or Path(str(item.get("relative_path"))).name)
            self.client.download_file(str(item.get("download_path")), destination)
            self.output_status.setText(f"Downloaded to {destination}")
            self.log(f"downloaded output: {destination}")
        except Exception as exc:
            self.output_status.setText(f"download failed: {exc}")
            self.log(f"download failed: {exc}")

    def download_all_outputs(self) -> None:
        if not self.output_files:
            self.output_status.setText("No outputs to download")
            return
        folder = QFileDialog.getExistingDirectory(self, "Download all listed outputs to folder")
        if not folder:
            return
        destination_root = Path(folder)
        try:
            for item in self.output_files:
                relative = Path(str(item.get("relative_path") or item.get("name")))
                destination = destination_root / str(item.get("source") or "output") / relative
                self.client.download_file(str(item.get("download_path")), destination)
            self.output_status.setText(f"Downloaded {len(self.output_files)} file(s) to {destination_root}")
            self.log(f"downloaded {len(self.output_files)} output file(s) to {destination_root}")
        except Exception as exc:
            self.output_status.setText(f"download all failed: {exc}")
            self.log(f"download all failed: {exc}")

    def download_via_taildrop(self) -> None:
        """Download outputs using Taildrop (fast P2P), fallback to HTTP if fails."""
        if not self.output_files:
            self.output_status.setText("No outputs to download")
            return

        folder = QFileDialog.getExistingDirectory(self, "Download outputs to folder")
        if not folder:
            return

        kind = self.outputs_kind.currentText()
        destination_dir = Path(folder)

        # Use output_tasks pattern for non-blocking operation
        from windows_app import output_tasks as async_base

        def transfer_task() -> str:
            # Step 1: Request Colab to create zip
            self.log(f"Creating {kind} output archive on Colab...")
            zip_info = self.client.create_zip(kind, timeout=120.0)

            zip_path = zip_info["zip_path"]
            zip_id = zip_info["zip_id"]
            size_mb = zip_info["size_bytes"] / (1024 * 1024)
            tailscale_host = zip_info.get("tailscale_hostname")

            self.log(f"Archive created: {zip_path} ({size_mb:.1f} MB)")

            local_zip = destination_dir / Path(zip_path).name

            # Step 2: Try Taildrop transfer
            if tailscale_host and check_tailscale_cli():
                self.log(f"Attempting Taildrop transfer from {tailscale_host}...")
                success, msg = taildrop_receive_file(tailscale_host, zip_path, local_zip)

                if success:
                    self.log(msg)
                    return str(local_zip)
                else:
                    self.log(f"Taildrop failed: {msg}")
                    self.log("Falling back to HTTP download...")
            else:
                reason = "Tailscale not available" if not tailscale_host else "Tailscale CLI not found"
                self.log(f"{reason}, using HTTP download...")

            # Step 3: HTTP fallback
            self.client.download_archive(zip_id, local_zip, timeout=1800.0)
            self.log(f"HTTP download complete: {local_zip}")
            return str(local_zip)

        def succeeded(task_id: str, result: object) -> None:
            if not hasattr(self, "output_taildrop_task_id") or task_id != self.output_taildrop_task_id:
                return
            self.output_status.setText(f"Downloaded to {result}")
            self.log(f"Transfer complete: {result}")

        def failed(task_id: str, error: str) -> None:
            if not hasattr(self, "output_taildrop_task_id") or task_id != self.output_taildrop_task_id:
                return
            self.output_status.setText(f"Transfer failed: {error}")
            self.log(f"Transfer failed: {error}")

        if not hasattr(self, "output_taildrop_task_id"):
            self.output_taildrop_task_id = ""
        self.output_taildrop_task_id = async_base._start_output_task(
            self, f"Transferring {kind} outputs...", transfer_task, succeeded, failed
        )

    def upload_source_if_needed(self) -> str:
        source_face = self.settings.source_face
        if not is_local_path(source_face):
            return source_face
        source_path = Path(source_face)
        if not source_path.is_file():
            raise FileNotFoundError(f"Local source face does not exist: {source_face}")
        self.log(f"uploading local source face: {source_path}")
        response = self.client.upload_file("/upload/file?kind=source", source_path)
        remote_path = str(response.get("path") or source_face)
        self.log(f"source uploaded to: {remote_path}")
        return remote_path

    def upload_input_if_needed(self, kind: str, input_path: str, output_path: str) -> tuple[str, str]:
        if not is_local_path(input_path):
            return input_path, output_path
        extensions = PHOTO_EXTENSIONS if kind == "photos" else VIDEO_EXTENSIONS
        files = local_files(input_path, extensions, self.settings.recursive)
        if not files:
            raise FileNotFoundError(f"No supported {kind} files found in local path: {input_path}")
        endpoint = f"/upload/{kind}"
        self.log(f"uploading {len(files)} local {kind} file(s)")
        response = self.client.upload_files(endpoint, files, timeout=600.0)
        remote_input = str(response.get("input_dir") or input_path)
        remote_output = output_path
        if is_local_path(output_path):
            remote_output = str(response.get("output_dir") or output_path)
            self.log(f"local output path is not reachable from Colab; using: {remote_output}")
        self.log(f"{kind} uploaded to: {remote_input}")
        return remote_input, remote_output

    def start_job(self, endpoint: str, payload: dict[str, Any]) -> None:
        try:
            response = self.client.request_json("POST", endpoint, payload)
            self.active_job_id = response.get("job_id")
            self.log(f"started {endpoint}: {response}")
            if self.active_job_id:
                if self.poller:
                    self.poller.stop()
                self.poller = PollWorker(self.client, self.active_job_id)
                self.poller.message.connect(self.log)
                self.poller.finished_status.connect(lambda status: self.log(f"job finished: {status}"))
                self.poller.start()
                self.tabs.setCurrentWidget(self.log_box)
        except Exception as exc:
            self.log(f"start failed: {exc}")

    def start_photos(self) -> None:
        self.sync_settings()
        self.tabs.setCurrentWidget(self.log_box)
        try:
            source_face = self.upload_source_if_needed()
            input_dir, output_dir = self.upload_input_if_needed("photos", self.settings.photos_input, self.settings.photos_output)
            self.start_job("/jobs/photos", job_payload(self.settings, input_dir, output_dir, source_face))
        except Exception as exc:
            self.log(f"photo batch failed before start: {exc}")

    def start_videos(self) -> None:
        self.sync_settings()
        self.tabs.setCurrentWidget(self.log_box)
        try:
            source_face = self.upload_source_if_needed()
            input_dir, output_dir = self.upload_input_if_needed("videos", self.settings.videos_input, self.settings.videos_output)
            self.start_job("/jobs/videos", job_payload(self.settings, input_dir, output_dir, source_face))
        except Exception as exc:
            self.log(f"video batch failed before start: {exc}")

    def cancel_job(self) -> None:
        self.sync_settings()
        if not self.active_job_id:
            self.log("no active job")
            return
        try:
            payload = self.client.request_json("POST", "/jobs/cancel", {"job_id": self.active_job_id})
            self.log("cancel: " + json.dumps(payload))
        except Exception as exc:
            self.log(f"cancel failed: {exc}")

    def start_live(self) -> None:
        self.sync_settings()
        if self.live_worker and self.live_worker.isRunning():
            self.log("live already running")
            return
        self.live_worker = LiveWorker(self.settings)
        self.live_worker.message.connect(self.log)
        self.live_worker.frame.connect(self.update_live_preview)
        self.live_worker.start()

    def stop_live(self) -> None:
        if self.live_worker:
            self.live_worker.stop()
            self.log("live stop requested")

    def update_live_preview(self, jpeg_bytes: bytes) -> None:
        image = QImage.fromData(jpeg_bytes, "JPG")
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(image).scaled(self.live_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.live_preview.setPixmap(pixmap)


def main() -> int:
    app = QApplication([])

    qss_path = Path(__file__).parent / "dark_theme.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())