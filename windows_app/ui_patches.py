from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QListWidget, QSplitter

from windows_app import app as base
from windows_app import async_outputs as async_base


def _link_source_fields(window: base.MainWindow) -> None:
    if getattr(window, "_source_fields_linked", False):
        return
    if not hasattr(window, "source_face") or not hasattr(window, "video_source_face"):
        return

    def mirror(target: base.QLineEdit, text: str) -> None:
        if target.text() == text:
            return
        target.blockSignals(True)
        target.setText(text)
        target.blockSignals(False)

    window.source_face.textChanged.connect(lambda text: mirror(window.video_source_face, text))
    window.video_source_face.textChanged.connect(lambda text: mirror(window.source_face, text))
    window._source_fields_linked = True


def _status_label(text: str = "") -> base.QLabel:
    label = base.QLabel(text)
    label.setObjectName("statusLabel")
    label.setWordWrap(True)
    return label


def _set_process_status(window: base.MainWindow, kind: str, text: str) -> None:
    names = {
        "setup": "setup_status",
        "photos": "photos_status",
        "videos": "videos_status",
        "outputs": "output_status",
        "live": "live_status",
    }
    label = getattr(window, names.get(kind, ""), None)
    if label is not None:
        label.setText(text)
    if kind != "outputs" and hasattr(window, "output_status"):
        # Keep the Outputs footer useful for background tasks started elsewhere.
        window.output_status.setText(text)


def _set_batch_button_running(window: base.MainWindow, kind: str) -> None:
    """Set the batch button to 'Stop' state (danger style)."""
    btn = getattr(window, f"{kind}_start_btn", None)
    if btn is None:
        return
    btn.setText(f"Stop {kind} batch")
    btn.setObjectName("dangerButton")
    btn.setStyle(btn.style())  # Force style refresh


def _set_batch_button_idle(window: base.MainWindow, kind: str) -> None:
    """Set the batch button to 'Start' state (primary style)."""
    btn = getattr(window, f"{kind}_start_btn", None)
    if btn is None:
        return
    btn.setText(f"Start {kind[:-1] if kind.endswith('s') else kind} batch")
    btn.setObjectName("primaryButton")
    btn.setStyle(btn.style())  # Force style refresh


def _is_batch_running(window: base.MainWindow) -> bool:
    """Check if a batch job is currently active."""
    return bool(window.active_job_id)


def _toggle_photos_batch(window: base.MainWindow) -> None:
    """Toggle between starting and stopping photo batch."""
    if _is_batch_running(window):
        cancel_job(window)
    else:
        start_photos(window)


def _toggle_videos_batch(window: base.MainWindow) -> None:
    """Toggle between starting and stopping video batch."""
    if _is_batch_running(window):
        cancel_job(window)
    else:
        start_videos(window)


def _save_settings_from_setup(window: base.MainWindow) -> None:
    window.sync_settings()
    window.log("settings saved")
    _set_process_status(window, "setup", "Settings saved")


def _build_setup_tab(self: base.MainWindow) -> None:
    tab = base.QWidget()
    layout = base.QVBoxLayout(tab)
    form = base.QFormLayout()
    self.host = base.QLineEdit(self.settings.host)
    self.port = base.QSpinBox()
    self.port.setRange(1, 65535)
    self.port.setValue(self.settings.port)
    self.drive_root = base.QLineEdit(self.settings.drive_root)
    form.addRow("Tailscale host/IP", self.host)
    form.addRow("API port", self.port)
    form.addRow("Drive root", self.drive_root)
    layout.addLayout(form)

    self.setup_help = base.QTextEdit(readOnly=True)
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

    row = base.QHBoxLayout()
    btn = base.QPushButton("Check connection")
    btn.setObjectName("primaryButton")
    btn.clicked.connect(self.check_connection)
    save = base.QPushButton("Save settings")
    save.setObjectName("successButton")
    save.clicked.connect(lambda: _save_settings_from_setup(self))
    row.addWidget(btn)
    row.addWidget(save)
    row.addStretch(1)
    layout.addLayout(row)

    self.setup_status = _status_label("Idle")
    layout.addWidget(self.setup_status)
    self.tabs.addTab(tab, "Setup")


def _build_photos_tab(self: base.MainWindow) -> None:
    tab = base.QWidget()
    layout = base.QVBoxLayout(tab)
    layout.addWidget(self._common_group())

    form = base.QFormLayout()
    self.photos_input = base.QLineEdit(self.settings.photos_input)
    self.photos_output = base.QLineEdit(self.settings.photos_output)
    photos_input_row = self._path_row(
        self.photos_input,
        lambda: self._browse_folder(self.photos_input, "Select photos input folder"),
    )
    form.addRow("Photos input path", photos_input_row)
    form.addRow("Photos output path", self.photos_output)
    layout.addLayout(form)

    self.photos_start_btn = base.QPushButton("Start photo batch")
    self.photos_start_btn.setObjectName("primaryButton")
    self.photos_start_btn.clicked.connect(lambda: _toggle_photos_batch(self))
    layout.addWidget(self.photos_start_btn)
    self.photos_status = _status_label("Idle")
    layout.addWidget(self.photos_status)
    layout.addStretch(1)
    self.tabs.addTab(tab, "Photos")


def _build_videos_tab(self: base.MainWindow) -> None:
    tab = base.QWidget()
    layout = base.QVBoxLayout(tab)
    form = base.QFormLayout()

    self.video_source_face = base.QLineEdit(self.settings.source_face)
    source_row = self._path_row(
        self.video_source_face,
        lambda: self._browse_file(self.video_source_face, "Select source face image"),
    )
    _link_source_fields(self)

    self.videos_input = base.QLineEdit(self.settings.videos_input)
    self.videos_output = base.QLineEdit(self.settings.videos_output)
    self.max_fps = base.QDoubleSpinBox()
    self.max_fps.setRange(1, 120)
    self.max_fps.setValue(self.settings.max_fps)
    self.max_width = base.QSpinBox()
    self.max_width.setRange(64, 4096)
    self.max_width.setValue(self.settings.max_width)
    self.quality = base.QSpinBox()
    self.quality.setRange(0, 51)
    self.quality.setValue(self.settings.quality)

    # Video segment range (percentage)
    self.start_pct = base.QDoubleSpinBox()
    self.start_pct.setRange(0, 99)
    self.start_pct.setValue(self.settings.start_pct)
    self.start_pct.setSuffix("%")
    self.start_pct.setDecimals(1)
    self.end_pct = base.QDoubleSpinBox()
    self.end_pct.setRange(1, 100)
    self.end_pct.setValue(self.settings.end_pct)
    self.end_pct.setSuffix("%")
    self.end_pct.setDecimals(1)
    range_row = base.QHBoxLayout()
    range_row.addWidget(self.start_pct)
    range_row.addWidget(base.QLabel("to"))
    range_row.addWidget(self.end_pct)
    range_row.addStretch(1)

    videos_input_row = self._path_row(
        self.videos_input,
        lambda: self._browse_folder(self.videos_input, "Select videos input folder"),
    )

    form.addRow("Source face path", source_row)
    form.addRow("Videos input path", videos_input_row)
    form.addRow("Videos output path", self.videos_output)
    form.addRow("Process range", range_row)
    form.addRow("Max FPS", self.max_fps)
    form.addRow("Max width", self.max_width)
    form.addRow("Quality", self.quality)
    layout.addLayout(form)

    # Common processing options (shared with Photos via linked widgets)
    options_box = base.QGroupBox("Processing options")
    options_form = base.QFormLayout(options_box)

    # Link to Photos tab widgets - changes sync automatically via sync_settings
    self.v_recursive = base.QCheckBox()
    self.v_recursive.setChecked(self.settings.recursive)
    self.v_overwrite = base.QCheckBox()
    self.v_overwrite.setChecked(self.settings.overwrite)
    self.v_skip_processed = base.QCheckBox()
    self.v_skip_processed.setChecked(self.settings.skip_processed)
    self.v_many_faces = base.QCheckBox()
    self.v_many_faces.setChecked(self.settings.many_faces)
    self.v_enhancer = base.QComboBox()
    self.v_enhancer.addItems(["none", "gfpgan", "gpen256", "gpen512"])
    self.v_enhancer.setCurrentText(self.settings.enhancer)
    self.v_opacity = base.QDoubleSpinBox()
    self.v_opacity.setRange(0.0, 1.0)
    self.v_opacity.setSingleStep(0.1)
    self.v_opacity.setValue(self.settings.opacity)
    self.v_sharpness = base.QDoubleSpinBox()
    self.v_sharpness.setRange(0.0, 1.0)
    self.v_sharpness.setSingleStep(0.1)
    self.v_sharpness.setValue(self.settings.sharpness)
    self.v_mouth_mask_size = base.QDoubleSpinBox()
    self.v_mouth_mask_size.setRange(0.0, 10.0)
    self.v_mouth_mask_size.setSingleStep(0.5)
    self.v_mouth_mask_size.setValue(self.settings.mouth_mask_size)
    self.v_interpolation_weight = base.QDoubleSpinBox()
    self.v_interpolation_weight.setRange(0.0, 1.0)
    self.v_interpolation_weight.setSingleStep(0.1)
    self.v_interpolation_weight.setValue(self.settings.interpolation_weight)
    self.v_poisson_blend = base.QCheckBox()
    self.v_poisson_blend.setChecked(self.settings.poisson_blend)
    self.v_color_correction = base.QCheckBox()
    self.v_color_correction.setChecked(self.settings.color_correction)

    options_form.addRow("Recursive", self.v_recursive)
    options_form.addRow("Overwrite", self.v_overwrite)
    options_form.addRow("Skip processed", self.v_skip_processed)
    options_form.addRow("Many faces", self.v_many_faces)
    options_form.addRow("Enhancer", self.v_enhancer)
    options_form.addRow("Opacity (1=full)", self.v_opacity)
    options_form.addRow("Sharpness (0=off)", self.v_sharpness)
    options_form.addRow("Mouth mask (0=off)", self.v_mouth_mask_size)
    options_form.addRow("Interpolation (0=off)", self.v_interpolation_weight)
    options_form.addRow("Poisson blend", self.v_poisson_blend)
    options_form.addRow("Color correction", self.v_color_correction)
    layout.addWidget(options_box)

    self.videos_start_btn = base.QPushButton("Start video batch")
    self.videos_start_btn.setObjectName("primaryButton")
    self.videos_start_btn.clicked.connect(lambda: _toggle_videos_batch(self))
    layout.addWidget(self.videos_start_btn)
    self.videos_status = _status_label("Idle")
    layout.addWidget(self.videos_status)
    layout.addStretch(1)
    self.tabs.addTab(tab, "Videos")


def _build_outputs_tab(self: base.MainWindow) -> None:
    tab = base.QWidget()
    layout = base.QVBoxLayout(tab)

    controls = base.QHBoxLayout()
    self.outputs_kind = base.QComboBox()
    self.outputs_kind.addItems(["photos", "videos"])
    refresh = base.QPushButton("Refresh")
    previous = base.QPushButton("Previous")
    next_button = base.QPushButton("Next")
    self.outputs_autoplay = base.QCheckBox("Auto-play")
    download_current = base.QPushButton("Download current")
    download_all = base.QPushButton("Download all")
    refresh.clicked.connect(self.refresh_outputs)
    previous.clicked.connect(self.previous_output)
    next_button.clicked.connect(self.next_output)
    self.outputs_autoplay.stateChanged.connect(self.toggle_outputs_autoplay)
    self.outputs_kind.currentTextChanged.connect(lambda _text: self.refresh_outputs())
    download_current.clicked.connect(self.download_current_output)
    download_all.clicked.connect(self.download_all_outputs)
    controls.addWidget(base.QLabel("Kind"))
    controls.addWidget(self.outputs_kind)
    controls.addWidget(refresh)
    controls.addWidget(previous)
    controls.addWidget(next_button)
    controls.addWidget(self.outputs_autoplay)
    controls.addWidget(download_current)
    controls.addWidget(download_all)
    controls.addStretch(1)
    layout.addLayout(controls)

    self.outputs_progress = base.QProgressBar()
    self.outputs_progress.setMaximum(100)
    self.outputs_progress.setFixedHeight(20)
    self.outputs_progress.setTextVisible(True)
    self.outputs_progress.hide()
    layout.addWidget(self.outputs_progress)

    splitter = QSplitter(Qt.Horizontal)
    self.outputs_list = QListWidget()
    self.outputs_list.setMinimumWidth(180)
    self.outputs_list.currentRowChanged.connect(self.show_output_at)
    splitter.addWidget(self.outputs_list)

    preview_panel = base.QWidget()
    preview_layout = base.QVBoxLayout(preview_panel)
    self.output_preview = base.QLabel("Refresh outputs to preview remote media")
    self.output_preview.setAlignment(Qt.AlignCenter)
    self.output_preview.setMinimumHeight(340)
    self.output_preview.setWordWrap(True)
    preview_layout.addWidget(self.output_preview, 1)

    self.output_video = None
    self.output_audio = None
    self.output_player = None
    if base.QMediaPlayer is not None and base.QVideoWidget is not None and base.QAudioOutput is not None:
        self.output_video = base.QVideoWidget()
        self.output_video.setMinimumHeight(340)
        self.output_audio = base.QAudioOutput(self)
        self.output_player = base.QMediaPlayer(self)
        self.output_player.setAudioOutput(self.output_audio)
        self.output_player.setVideoOutput(self.output_video)
        preview_layout.addWidget(self.output_video, 1)
        self.output_video.hide()

    self.output_status = base.QLabel("")
    self.output_status.setObjectName("statusLabel")
    self.output_status.setWordWrap(True)
    preview_layout.addWidget(self.output_status)

    splitter.addWidget(preview_panel)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([300, 820])
    layout.addWidget(splitter, 1)
    self.tabs.addTab(tab, "Outputs")


def _build_live_tab(self: base.MainWindow) -> None:
    tab = base.QWidget()
    layout = base.QVBoxLayout(tab)
    form = base.QFormLayout()
    self.camera_index = base.QSpinBox()
    self.camera_index.setRange(0, 20)
    self.camera_index.setValue(self.settings.camera_index)
    self.virtual_camera = base.QLineEdit(self.settings.virtual_camera)
    form.addRow("Camera index", self.camera_index)
    form.addRow("Virtual camera", self.virtual_camera)
    layout.addLayout(form)

    self.live_note = _status_label(
        "Live sends webcam JPEG frames to ws://HOST:PORT/ws/live, previews returned frames, "
        "and opens the configured virtual camera when pyvirtualcam can find it."
    )
    layout.addWidget(self.live_note)
    self.live_preview = base.QLabel("Live preview")
    self.live_preview.setAlignment(Qt.AlignCenter)
    self.live_preview.setMinimumHeight(360)
    layout.addWidget(self.live_preview)

    row = base.QHBoxLayout()
    start = base.QPushButton("Start live")
    start.setObjectName("successButton")
    stop = base.QPushButton("Stop live")
    stop.setObjectName("dangerButton")
    start.clicked.connect(self.start_live)
    stop.clicked.connect(self.stop_live)
    row.addWidget(start)
    row.addWidget(stop)
    row.addStretch(1)
    layout.addLayout(row)

    self.live_status = _status_label("Idle")
    layout.addWidget(self.live_status)
    layout.addStretch(1)
    self.tabs.addTab(tab, "Live")


def _ensure_prefetch_state(window: base.MainWindow) -> None:
    async_base._ensure_output_worker_state(window)
    if not hasattr(window, "output_prefetch_cache"):
        window.output_prefetch_cache = {}
    if not hasattr(window, "output_prefetching"):
        window.output_prefetching = set()


def _cache_key(item: dict[str, Any]) -> str:
    return str(item.get("download_path") or "")


def _video_cache_path(window: base.MainWindow, item: dict[str, Any]) -> Path:
    relative = str(item.get("relative_path") or item.get("name") or "output.mp4")
    safe_relative = relative.replace("/", "_").replace("\\", "_")
    return window.output_temp_dir / f"{item.get('source', 'output')}_{safe_relative}"


def _display_photo(window: base.MainWindow, item: dict[str, Any], data: bytes) -> None:
    if window.output_video is not None:
        window.output_video.hide()
    window.output_preview.show()
    image = QImage.fromData(data)
    if image.isNull():
        window.output_status.setText("preview failed: downloaded image could not be decoded")
        return
    pixmap = QPixmap.fromImage(image).scaled(window.output_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    window.output_preview.setPixmap(pixmap)
    window.output_status.setText(f"Showing {item.get('relative_path')} from {item.get('source')}")


def _display_video(window: base.MainWindow, item: dict[str, Any], local_path: Path) -> None:
    relative = str(item.get("relative_path") or item.get("name") or "output.mp4")
    if window.output_player is None or window.output_video is None:
        window.output_preview.show()
        window.output_preview.setText(
            f"Video ready to download:\n{relative}\n\nInstall PySide6 multimedia support for inline playback."
        )
        window.output_status.setText(f"Selected video {relative}")
        return
    window.output_preview.hide()
    window.output_video.show()
    window.output_player.setSource(QUrl.fromLocalFile(str(local_path)))
    window.output_player.play()
    window.output_status.setText(f"Playing {relative}")


def _prefetch_output(window: base.MainWindow, index: int) -> None:
    _ensure_prefetch_state(window)
    if index < 0 or index >= len(window.output_files):
        return
    item = dict(window.output_files[index])
    key = _cache_key(item)
    if not key or key in window.output_prefetch_cache or key in window.output_prefetching:
        return
    kind = window.outputs_kind.currentText()
    window.output_prefetching.add(key)
    task_id = uuid.uuid4().hex

    def task() -> object:
        if kind == "photos":
            return window.client.download_bytes(key, timeout=30.0)
        local_path = _video_cache_path(window, item)
        if not local_path.exists() or local_path.stat().st_size != int(item.get("size") or -1):
            async_base._download_file_fast(window.client, key, local_path, timeout=900.0)
        return str(local_path)

    def succeeded(done_task_id: str, result: object) -> None:
        if done_task_id != task_id:
            return
        window.output_prefetching.discard(key)
        window.output_prefetch_cache[key] = Path(result) if kind == "videos" else result

    def failed(done_task_id: str, _error: str) -> None:
        if done_task_id == task_id:
            window.output_prefetching.discard(key)

    worker = async_base.OutputTaskWorker(task_id, task)
    window.output_workers[task_id] = worker
    worker.succeeded.connect(succeeded)
    worker.failed.connect(failed)
    worker.finished.connect(lambda task_id=task_id: window.output_workers.pop(task_id, None))
    worker.start()


def _prefetch_neighbors(window: base.MainWindow, index: int) -> None:
    if not window.output_files:
        return
    _prefetch_output(window, (index + 1) % len(window.output_files))
    _prefetch_output(window, (index + 2) % len(window.output_files))


def _poll_message(window: base.MainWindow, kind: str, text: str) -> None:
    window.log(text)
    _set_process_status(window, kind, text)


def _start_batch_with_status(self: base.MainWindow, kind: str) -> None:
    self.sync_settings()
    async_base._ensure_output_worker_state(self)
    settings = async_base._copy_settings(self.settings)
    self.log(f"starting {kind} batch...")
    _set_process_status(self, kind, f"Starting {kind} batch...")
    _set_batch_button_running(self, kind)

    def task() -> dict[str, Any]:
        return async_base._prepare_and_start_batch(settings, kind)

    def on_job_finished(status: str, batch_kind: str) -> None:
        self.log(f"job finished: {status}")
        _set_process_status(self, batch_kind, f"Job finished: {status}")
        self.active_job_id = None
        _set_batch_button_idle(self, batch_kind)

    def succeeded(task_id: str, result: object) -> None:
        if task_id != self.output_batch_task_id:
            return
        payload = result if isinstance(result, dict) else {}
        for line in payload.get("logs") or []:
            line_text = str(line)
            self.log(line_text)
            _set_process_status(self, kind, line_text)
        response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
        self.active_job_id = response.get("job_id")
        if self.active_job_id:
            if self.poller:
                self.poller.stop()
            self.poller = base.PollWorker(self.client, self.active_job_id)
            self.poller.message.connect(lambda text, batch_kind=kind: _poll_message(self, batch_kind, text))
            self.poller.finished_status.connect(lambda status, batch_kind=kind: on_job_finished(status, batch_kind))
            self.poller.start()
            _set_process_status(self, kind, f"{kind} batch running...")
        else:
            # No job ID means it didn't start properly
            _set_batch_button_idle(self, kind)
            _set_process_status(self, kind, f"{kind} batch failed to start")

    def failed(task_id: str, error: str) -> None:
        if task_id != self.output_batch_task_id:
            return
        text = f"{kind} batch failed before start: {error}"
        _set_process_status(self, kind, text)
        self.log(text)
        _set_batch_button_idle(self, kind)

    self.output_batch_task_id = async_base._start_output_task(
        self,
        f"Starting {kind} batch...",
        task,
        succeeded,
        failed,
    )


def start_photos(self: base.MainWindow) -> None:
    _start_batch_with_status(self, "photos")


def start_videos(self: base.MainWindow) -> None:
    _start_batch_with_status(self, "videos")


def check_connection(self: base.MainWindow) -> None:
    self.sync_settings()
    async_base._ensure_output_worker_state(self)
    self.log("checking connection...")
    _set_process_status(self, "setup", "Checking connection...")

    def fetch_health() -> dict[str, Any]:
        return self.client.request_json("GET", "/health", timeout=5.0)

    def succeeded(task_id: str, payload: object) -> None:
        if task_id != self.output_health_task_id:
            return
        self.log("health: " + base.json.dumps(payload, indent=2))
        _set_process_status(self, "setup", "Connected to Colab API")

    def failed(task_id: str, error: str) -> None:
        if task_id != self.output_health_task_id:
            return
        text = f"health failed: {error}"
        self.log(text)
        _set_process_status(self, "setup", text)

    self.output_health_task_id = async_base._start_output_task(
        self,
        "Checking connection...",
        fetch_health,
        succeeded,
        failed,
    )


def cancel_job(self: base.MainWindow) -> None:
    self.sync_settings()
    if not self.active_job_id:
        self.log("no active job")
        _set_process_status(self, "photos", "No active job")
        _set_process_status(self, "videos", "No active job")
        return
    try:
        payload = self.client.request_json("POST", "/jobs/cancel", {"job_id": self.active_job_id})
        text = "cancel: " + base.json.dumps(payload)
        self.log(text)
        _set_process_status(self, "photos", "Cancel requested")
        _set_process_status(self, "videos", "Cancel requested")
    except Exception as exc:
        text = f"cancel failed: {exc}"
        self.log(text)
        _set_process_status(self, "photos", text)
        _set_process_status(self, "videos", text)


def start_live(self: base.MainWindow) -> None:
    self.sync_settings()
    if self.live_worker and self.live_worker.isRunning():
        self.log("live already running")
        _set_process_status(self, "live", "Live already running")
        return
    self.live_worker = base.LiveWorker(self.settings)
    self.live_worker.message.connect(lambda text: _poll_message(self, "live", text))
    self.live_worker.frame.connect(self.update_live_preview)
    self.live_worker.start()
    _set_process_status(self, "live", "Starting live...")


def stop_live(self: base.MainWindow) -> None:
    if self.live_worker:
        self.live_worker.stop()
        self.log("live stop requested")
        _set_process_status(self, "live", "Live stop requested")


def update_live_preview(self: base.MainWindow, jpeg_bytes: bytes) -> None:
    image = QImage.fromData(jpeg_bytes, "JPG")
    if image.isNull():
        return
    pixmap = QPixmap.fromImage(image).scaled(self.live_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    self.live_preview.setPixmap(pixmap)
    _set_process_status(self, "live", "Live receiving frames")


def show_output_at(self: base.MainWindow, index: int) -> None:
    if index < 0 or index >= len(self.output_files):
        return
    self.output_current_loaded = False
    _ensure_prefetch_state(self)
    item = dict(self.output_files[index])
    key = _cache_key(item)
    kind = self.outputs_kind.currentText()
    file_size = int(item.get("size") or 0)
    if not key:
        self.output_status.setText("selected output has no download path")
        self.output_current_loaded = True
        return
    self.stop_output_video()

    cached = self.output_prefetch_cache.get(key)
    if kind == "photos":
        self.output_preview.setPixmap(QPixmap())
        if isinstance(cached, (bytes, bytearray)):
            _display_photo(self, item, bytes(cached))
            _prefetch_neighbors(self, index)
            self.output_current_loaded = True
            return
        size_str = base.format_size(file_size) if file_size > 0 else ""
        self.output_preview.setText(f"Loading photo preview... {size_str}")
        # Show progress bar
        if hasattr(self, "outputs_progress"):
            self.outputs_progress.setMinimum(0)
            self.outputs_progress.setMaximum(100)
            self.outputs_progress.setValue(0)
            self.outputs_progress.show()

        from typing import Callable
        def fetch_photo(progress_cb: Callable[[int, int], None]) -> bytes:
            return async_base._download_bytes_with_progress(self.client, key, timeout=20.0, progress_callback=progress_cb)

        def photo_ready(task_id: str, data: object) -> None:
            if task_id != self.output_preview_task_id:
                return
            if hasattr(self, "outputs_progress"):
                self.outputs_progress.hide()
            if isinstance(data, (bytes, bytearray)):
                self.output_prefetch_cache[key] = bytes(data)
                _display_photo(self, item, bytes(data))
                _prefetch_neighbors(self, index)
            self.output_current_loaded = True

        def photo_failed(task_id: str, error: str) -> None:
            if task_id != self.output_preview_task_id:
                return
            if hasattr(self, "outputs_progress"):
                self.outputs_progress.hide()
            self.output_status.setText(f"preview failed: {error}")
            self.log(f"output preview failed: {error}")
            self.output_current_loaded = True

        self.output_preview_task_id = async_base._start_output_task_with_progress(
            self, "Loading photo preview...", fetch_photo, photo_ready, photo_failed
        )
        return

    if isinstance(cached, Path) and cached.exists():
        _display_video(self, item, cached)
        _prefetch_neighbors(self, index)
        self.output_current_loaded = True
        return
    self.show_video_output(item)


def show_video_output(self: base.MainWindow, item: dict[str, Any]) -> None:
    _ensure_prefetch_state(self)
    key = _cache_key(item)
    file_size = int(item.get("size") or 0)
    relative = str(item.get("relative_path") or item.get("name") or "output.mp4")
    local_path = _video_cache_path(self, item)
    self.output_preview.setPixmap(QPixmap())
    size_str = base.format_size(file_size) if file_size > 0 else ""
    self.output_preview.setText(f"Loading video preview:\n{relative}\n{size_str}")
    # Show progress bar
    if hasattr(self, "outputs_progress"):
        self.outputs_progress.setMinimum(0)
        self.outputs_progress.setMaximum(100)
        self.outputs_progress.setValue(0)
        self.outputs_progress.show()

    from typing import Callable
    def fetch_video(progress_cb: Callable[[int, int], None]) -> dict[str, str]:
        if not local_path.exists() or local_path.stat().st_size != file_size:
            async_base._download_file_fast(self.client, key, local_path, timeout=900.0, progress_callback=progress_cb)
        return {"relative": relative, "local_path": str(local_path), "key": key}

    def video_ready(task_id: str, result: object) -> None:
        if task_id != self.output_preview_task_id:
            return
        if hasattr(self, "outputs_progress"):
            self.outputs_progress.hide()
        payload = result if isinstance(result, dict) else {}
        ready_path = Path(str(payload.get("local_path") or local_path))
        self.output_prefetch_cache[str(payload.get("key") or key)] = ready_path
        _display_video(self, item, ready_path)
        _prefetch_neighbors(self, self.outputs_list.currentRow())
        self.output_current_loaded = True

    def video_failed(task_id: str, error: str) -> None:
        if task_id != self.output_preview_task_id:
            return
        if hasattr(self, "outputs_progress"):
            self.outputs_progress.hide()
        self.output_status.setText(f"preview failed: {error}")
        self.log(f"output preview failed: {error}")
        self.output_current_loaded = True

    self.output_preview_task_id = async_base._start_output_task_with_progress(
        self, f"Loading video preview: {relative}", fetch_video, video_ready, video_failed
    )


def sync_settings(self: base.MainWindow) -> None:
    """Extended sync_settings that reads from both Photos and Videos tab widgets."""
    self.settings.host = self.host.text().strip()
    self.settings.port = int(self.port.value())
    self.settings.drive_root = self.drive_root.text().strip()
    self.settings.source_face = self.source_face.text().strip()
    self.settings.photos_input = self.photos_input.text().strip()
    self.settings.photos_output = self.photos_output.text().strip()
    self.settings.videos_input = self.videos_input.text().strip()
    self.settings.videos_output = self.videos_output.text().strip()
    # Read from videos tab widgets if available, else photos tab
    self.settings.recursive = getattr(self, "v_recursive", self.recursive).isChecked()
    self.settings.overwrite = getattr(self, "v_overwrite", self.overwrite).isChecked()
    self.settings.skip_processed = getattr(self, "v_skip_processed", self.skip_processed).isChecked()
    self.settings.many_faces = getattr(self, "v_many_faces", self.many_faces).isChecked()
    self.settings.enhancer = getattr(self, "v_enhancer", self.enhancer).currentText()
    self.settings.opacity = float(getattr(self, "v_opacity", self.opacity).value())
    self.settings.sharpness = float(getattr(self, "v_sharpness", self.sharpness).value())
    self.settings.mouth_mask_size = float(getattr(self, "v_mouth_mask_size", self.mouth_mask_size).value())
    self.settings.interpolation_weight = float(getattr(self, "v_interpolation_weight", self.interpolation_weight).value())
    self.settings.poisson_blend = getattr(self, "v_poisson_blend", self.poisson_blend).isChecked()
    self.settings.color_correction = getattr(self, "v_color_correction", self.color_correction).isChecked()
    self.settings.max_fps = float(self.max_fps.value())
    self.settings.max_width = int(self.max_width.value())
    self.settings.quality = int(self.quality.value())
    self.settings.start_pct = float(self.start_pct.value())
    self.settings.end_pct = float(self.end_pct.value())
    self.settings.camera_index = int(self.camera_index.value())
    self.settings.virtual_camera = self.virtual_camera.text().strip()
    base.save_settings(self.settings)
    # Sync both tabs' widgets to stay consistent
    _sync_common_widgets(self)


def _sync_common_widgets(window: base.MainWindow) -> None:
    """Keep Photos and Videos tab common widgets in sync."""
    s = window.settings
    # Update photos tab widgets
    if hasattr(window, "recursive"):
        window.recursive.setChecked(s.recursive)
        window.overwrite.setChecked(s.overwrite)
        window.skip_processed.setChecked(s.skip_processed)
        window.many_faces.setChecked(s.many_faces)
        window.enhancer.setCurrentText(s.enhancer)
        window.opacity.setValue(s.opacity)
        window.sharpness.setValue(s.sharpness)
        window.mouth_mask_size.setValue(s.mouth_mask_size)
        window.interpolation_weight.setValue(s.interpolation_weight)
        window.poisson_blend.setChecked(s.poisson_blend)
        window.color_correction.setChecked(s.color_correction)
    # Update videos tab widgets
    if hasattr(window, "v_recursive"):
        window.v_recursive.setChecked(s.recursive)
        window.v_overwrite.setChecked(s.overwrite)
        window.v_skip_processed.setChecked(s.skip_processed)
        window.v_many_faces.setChecked(s.many_faces)
        window.v_enhancer.setCurrentText(s.enhancer)
        window.v_opacity.setValue(s.opacity)
        window.v_sharpness.setValue(s.sharpness)
        window.v_mouth_mask_size.setValue(s.mouth_mask_size)
        window.v_interpolation_weight.setValue(s.interpolation_weight)
        window.v_poisson_blend.setChecked(s.poisson_blend)
        window.v_color_correction.setChecked(s.color_correction)


def install() -> None:
    base.MainWindow._build_setup_tab = _build_setup_tab
    base.MainWindow._build_photos_tab = _build_photos_tab
    base.MainWindow._build_videos_tab = _build_videos_tab
    base.MainWindow._build_outputs_tab = _build_outputs_tab
    base.MainWindow._build_live_tab = _build_live_tab
    base.MainWindow.check_connection = check_connection
    base.MainWindow.sync_settings = sync_settings
    base.MainWindow.start_photos = start_photos
    base.MainWindow.start_videos = start_videos
    base.MainWindow.cancel_job = cancel_job
    base.MainWindow.start_live = start_live
    base.MainWindow.stop_live = stop_live
    base.MainWindow.update_live_preview = update_live_preview
    base.MainWindow.show_output_at = show_output_at
    base.MainWindow.show_video_output = show_video_output


install()
main = base.main
