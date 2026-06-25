from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QScrollArea

from windows_app import processing_options_patches as _processing_options_patches
from windows_app import app as base
from windows_app import async_outputs as async_base
from windows_app import ui_patches as ui_base

DEFAULT_LIVE_WIDTH = 1280
DEFAULT_LIVE_HEIGHT = 720
DEFAULT_LIVE_FPS = 30
DEFAULT_LIVE_PIPELINE_FRAMES = 16
DEFAULT_LIVE_JPEG_QUALITY = 80
DEFAULT_LIVE_FRAME_CODEC = "jpeg"
DEFAULT_LIVE_OUTPUT_CODEC = "jpeg"
LIVE_FRAME_CODECS = ("jpeg", "webp")
DEFAULT_LIVE_DETECTOR_SIZE = 320
DEFAULT_LIVE_DETECT_EVERY_N = 1
DEFAULT_LIVE_FACE_MODEL_PACK = "buffalo_l"
LIVE_FACE_MODEL_PACKS = ("buffalo_l", "buffalo_m", "buffalo_s")
DEFAULT_LIVE_SWAPPER_PRECISION = "fp32"
LIVE_SWAPPER_PRECISIONS = ("fp32", "fp16")
DEFAULT_LIVE_PREVIEW_BUFFER_SECONDS = 1.0
DEFAULT_LIVE_PREVIEW_SCALE = "fit"
LIVE_PREVIEW_SCALES = ("fit", "1x", "1.5x", "2x")
LIVE_OPTION_KEYS = (
    "many_faces",
    "enhancer",
    "opacity",
    "sharpness",
    "mouth_mask_size",
    "interpolation_weight",
    "poisson_blend",
    "color_correction",
    "max_width",
    "frame_codec",
    "output_codec",
    "jpeg_quality",
    "detector_size",
    "detect_every_n",
    "face_model_pack",
    "swapper_precision",
    "cache_source_face",
    "preview_buffer_seconds",
    "preview_scale",
)

_previous_load_settings = base.load_settings
_previous_save_settings = base.save_settings
_original_sync_settings = base.MainWindow.sync_settings
_original_close_event = base.MainWindow.closeEvent
_original_stop_live = base.MainWindow.stop_live


def _live_setting(settings: base.AppSettings, name: str, default: int) -> int:
    try:
        value = int(getattr(settings, name, default))
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def _json_payload(text: object) -> dict[str, Any]:
    if not isinstance(text, str):
        return {}
    try:
        payload = base.json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _status_label(text: str = "") -> base.QLabel:
    label = base.QLabel(text)
    label.setObjectName("statusLabel")
    label.setWordWrap(True)
    return label


def _default_live_options() -> dict[str, Any]:
    defaults = base.AppSettings()
    return {
        "many_faces": False,
        "enhancer": "none",
        "opacity": 1.0,
        "sharpness": 0.0,
        "mouth_mask_size": 0.0,
        "interpolation_weight": 0.0,
        "poisson_blend": False,
        "color_correction": False,
        "max_width": defaults.max_width,
        "frame_codec": DEFAULT_LIVE_FRAME_CODEC,
        "output_codec": DEFAULT_LIVE_OUTPUT_CODEC,
        "jpeg_quality": DEFAULT_LIVE_JPEG_QUALITY,
        "detector_size": DEFAULT_LIVE_DETECTOR_SIZE,
        "detect_every_n": DEFAULT_LIVE_DETECT_EVERY_N,
        "face_model_pack": DEFAULT_LIVE_FACE_MODEL_PACK,
        "swapper_precision": DEFAULT_LIVE_SWAPPER_PRECISION,
        "cache_source_face": True,
        "preview_buffer_seconds": DEFAULT_LIVE_PREVIEW_BUFFER_SECONDS,
        "preview_scale": DEFAULT_LIVE_PREVIEW_SCALE,
    }


def _coerce_live_options(value: object) -> dict[str, Any]:
    options = _default_live_options()
    if isinstance(value, dict):
        for key in LIVE_OPTION_KEYS:
            if key in value:
                options[key] = value[key]
    options["many_faces"] = bool(options["many_faces"])
    options["enhancer"] = str(options["enhancer"])
    options["opacity"] = float(options["opacity"])
    options["sharpness"] = float(options["sharpness"])
    options["mouth_mask_size"] = float(options["mouth_mask_size"])
    options["interpolation_weight"] = float(options["interpolation_weight"])
    options["poisson_blend"] = bool(options["poisson_blend"])
    options["color_correction"] = bool(options["color_correction"])
    options["max_width"] = max(64, int(options["max_width"]))
    options["frame_codec"] = str(options["frame_codec"]).lower()
    if options["frame_codec"] not in LIVE_FRAME_CODECS:
        options["frame_codec"] = DEFAULT_LIVE_FRAME_CODEC
    options["output_codec"] = str(options["output_codec"]).lower()
    if options["output_codec"] not in LIVE_FRAME_CODECS:
        options["output_codec"] = DEFAULT_LIVE_OUTPUT_CODEC
    options["jpeg_quality"] = max(20, min(95, int(options["jpeg_quality"])))
    options["detector_size"] = max(160, min(640, int(options["detector_size"])))
    options["detector_size"] = max(32, int(options["detector_size"]) // 32 * 32)
    options["detect_every_n"] = max(1, min(30, int(options["detect_every_n"])))
    options["face_model_pack"] = str(options["face_model_pack"])
    if options["face_model_pack"] not in LIVE_FACE_MODEL_PACKS:
        options["face_model_pack"] = DEFAULT_LIVE_FACE_MODEL_PACK
    options["cache_source_face"] = bool(options["cache_source_face"])
    options["preview_buffer_seconds"] = max(0.0, min(5.0, float(options["preview_buffer_seconds"])))
    options["preview_scale"] = str(options["preview_scale"]).lower()
    if options["preview_scale"] not in LIVE_PREVIEW_SCALES:
        options["preview_scale"] = DEFAULT_LIVE_PREVIEW_SCALE
    options["swapper_precision"] = str(options["swapper_precision"]).lower()
    if options["swapper_precision"] not in LIVE_SWAPPER_PRECISIONS:
        options["swapper_precision"] = DEFAULT_LIVE_SWAPPER_PRECISION
    return options


def _live_options(settings: base.AppSettings) -> dict[str, Any]:
    return _coerce_live_options(getattr(settings, "live_options", None))


def _apply_live_options_to_settings(settings: base.AppSettings) -> None:
    options = _live_options(settings)
    settings.live_options = options
    for key in LIVE_OPTION_KEYS:
        if key != "jpeg_quality":
            setattr(settings, key, options[key])
    settings.live_jpeg_quality = options["jpeg_quality"]


def _source_fields(window: base.MainWindow) -> list[Any]:
    fields = []
    for name in ("source_face", "video_source_face", "live_source_face"):
        field = getattr(window, name, None)
        if field is not None:
            fields.append(field)
    return fields


def _link_live_source_fields(window: base.MainWindow) -> None:
    if getattr(window, "_live_source_fields_linked", False):
        return
    fields = _source_fields(window)
    if len(fields) < 2:
        return

    def mirror(origin: Any, text: str) -> None:
        window.settings.source_face = text.strip()
        for target in _source_fields(window):
            if target is origin or target.text() == text:
                continue
            target.blockSignals(True)
            target.setText(text)
            target.blockSignals(False)

    for field in fields:
        field.textChanged.connect(lambda text, origin=field: mirror(origin, text))
    window._live_source_fields_linked = True


def _read_live_options(window: base.MainWindow) -> dict[str, Any]:
    if not hasattr(window, "live_max_width"):
        return _live_options(window.settings)
    return _coerce_live_options(
        {
            "many_faces": window.live_many_faces.isChecked(),
            "enhancer": window.live_enhancer.currentText(),
            "opacity": float(window.live_opacity.value()),
            "sharpness": float(window.live_sharpness.value()),
            "mouth_mask_size": float(window.live_mouth_mask_size.value()),
            "interpolation_weight": float(window.live_interpolation_weight.value()),
            "poisson_blend": window.live_poisson_blend.isChecked(),
            "color_correction": window.live_color_correction.isChecked(),
            "max_width": int(window.live_max_width.value()),
            "frame_codec": window.live_frame_codec.currentText(),
            "output_codec": window.live_output_codec.currentText(),
            "jpeg_quality": int(window.live_jpeg_quality.value()),
            "detector_size": int(window.live_detector_size.value()),
            "detect_every_n": int(window.live_detect_every_n.value()),
            "face_model_pack": window.live_face_model_pack.currentText(),
            "swapper_precision": window.live_swapper_precision.currentText(),
            "cache_source_face": window.live_cache_source_face.isChecked(),
            "preview_buffer_seconds": float(window.live_preview_buffer_seconds.value()),
            "preview_scale": window.live_preview_scale.currentText() if hasattr(window, "live_preview_scale") else DEFAULT_LIVE_PREVIEW_SCALE,
        }
    )


def _apply_live_options_to_widgets(window: base.MainWindow) -> None:
    if not hasattr(window, "live_max_width"):
        return
    options = _live_options(window.settings)
    window.live_many_faces.setChecked(bool(options["many_faces"]))
    window.live_enhancer.setCurrentText(str(options["enhancer"]))
    window.live_opacity.setValue(float(options["opacity"]))
    window.live_sharpness.setValue(float(options["sharpness"]))
    window.live_mouth_mask_size.setValue(float(options["mouth_mask_size"]))
    window.live_interpolation_weight.setValue(float(options["interpolation_weight"]))
    window.live_poisson_blend.setChecked(bool(options["poisson_blend"]))
    window.live_color_correction.setChecked(bool(options["color_correction"]))
    window.live_max_width.setValue(int(options["max_width"]))
    window.live_frame_codec.setCurrentText(str(options["frame_codec"]))
    window.live_output_codec.setCurrentText(str(options["output_codec"]))
    window.live_jpeg_quality.setValue(int(options["jpeg_quality"]))
    window.live_detector_size.setValue(int(options["detector_size"]))
    window.live_detect_every_n.setValue(int(options["detect_every_n"]))
    window.live_face_model_pack.setCurrentText(str(options["face_model_pack"]))
    window.live_swapper_precision.setCurrentText(str(options["swapper_precision"]))
    window.live_cache_source_face.setChecked(bool(options["cache_source_face"]))
    window.live_preview_buffer_seconds.setValue(float(options["preview_buffer_seconds"]))
    if hasattr(window, "live_preview_scale"):
        window.live_preview_scale.setCurrentText(str(options["preview_scale"]))


def load_settings() -> base.AppSettings:
    settings = _previous_load_settings()
    data: dict[str, Any] = {}
    if base.APP_STATE.is_file():
        try:
            loaded = base.json.loads(base.APP_STATE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}
    settings.live_width = int(data.get("live_width") or DEFAULT_LIVE_WIDTH)
    settings.live_height = int(data.get("live_height") or DEFAULT_LIVE_HEIGHT)
    settings.live_fps = int(data.get("live_fps") or DEFAULT_LIVE_FPS)
    settings.live_pipeline_frames = int(data.get("live_pipeline_frames") or DEFAULT_LIVE_PIPELINE_FRAMES)
    settings.live_options = _coerce_live_options(data.get("live_options"))
    return settings


def save_settings(settings: base.AppSettings) -> None:
    _previous_save_settings(settings)
    try:
        data = base.json.loads(base.APP_STATE.read_text(encoding="utf-8")) if base.APP_STATE.is_file() else {}
        if not isinstance(data, dict):
            data = {}
        data["live_width"] = _live_setting(settings, "live_width", DEFAULT_LIVE_WIDTH)
        data["live_height"] = _live_setting(settings, "live_height", DEFAULT_LIVE_HEIGHT)
        data["live_fps"] = _live_setting(settings, "live_fps", DEFAULT_LIVE_FPS)
        data["live_pipeline_frames"] = _live_setting(settings, "live_pipeline_frames", DEFAULT_LIVE_PIPELINE_FRAMES)
        data["live_options"] = _live_options(settings)
        base.APP_STATE.write_text(base.json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def _build_live_tab(self: base.MainWindow) -> None:
    tab = base.QWidget()
    layout = base.QVBoxLayout(tab)
    splitter = ui_base.QSplitter(ui_base.Qt.Horizontal)

    controls_panel = base.QWidget()
    controls_layout = base.QVBoxLayout(controls_panel)
    form = base.QFormLayout()

    self.camera_index = base.QSpinBox()
    self.camera_index.setRange(0, 20)
    self.camera_index.setValue(self.settings.camera_index)
    self.virtual_camera = base.QLineEdit(self.settings.virtual_camera)
    self.live_source_face = base.QLineEdit(self.settings.source_face)
    live_source_row = self._path_row(
        self.live_source_face,
        lambda: self._browse_file(self.live_source_face, "Select source face image"),
    )
    self.live_width = base.QSpinBox()
    self.live_width.setRange(160, 4096)
    self.live_width.setValue(_live_setting(self.settings, "live_width", DEFAULT_LIVE_WIDTH))
    self.live_height = base.QSpinBox()
    self.live_height.setRange(120, 2160)
    self.live_height.setValue(_live_setting(self.settings, "live_height", DEFAULT_LIVE_HEIGHT))
    self.live_fps = base.QSpinBox()
    self.live_fps.setRange(1, 120)
    self.live_fps.setValue(_live_setting(self.settings, "live_fps", DEFAULT_LIVE_FPS))
    self.live_pipeline_frames = base.QSpinBox()
    self.live_pipeline_frames.setRange(8, 512)
    self.live_pipeline_frames.setValue(_live_setting(self.settings, "live_pipeline_frames", DEFAULT_LIVE_PIPELINE_FRAMES))

    form.addRow("Camera index", self.camera_index)
    form.addRow("Virtual camera", self.virtual_camera)
    form.addRow("Source face path", live_source_row)
    form.addRow("Capture width", self.live_width)
    form.addRow("Capture height", self.live_height)
    form.addRow("Capture FPS", self.live_fps)
    form.addRow("Pipeline frames", self.live_pipeline_frames)
    controls_layout.addLayout(form)
    _link_live_source_fields(self)

    options_box = base.QGroupBox("Live processing options")
    options_form = base.QFormLayout(options_box)
    self.live_many_faces = base.QCheckBox()
    self.live_enhancer = base.QComboBox()
    self.live_enhancer.addItems(["none", "gfpgan", "gpen256", "gpen512"])
    self.live_opacity = base.QDoubleSpinBox()
    self.live_opacity.setRange(0.0, 1.0)
    self.live_opacity.setSingleStep(0.1)
    self.live_sharpness = base.QDoubleSpinBox()
    self.live_sharpness.setRange(0.0, 1.0)
    self.live_sharpness.setSingleStep(0.1)
    self.live_mouth_mask_size = base.QDoubleSpinBox()
    self.live_mouth_mask_size.setRange(0.0, 10.0)
    self.live_mouth_mask_size.setSingleStep(0.5)
    self.live_interpolation_weight = base.QDoubleSpinBox()
    self.live_interpolation_weight.setRange(0.0, 1.0)
    self.live_interpolation_weight.setSingleStep(0.1)
    self.live_poisson_blend = base.QCheckBox()
    self.live_color_correction = base.QCheckBox()
    self.live_max_width = base.QSpinBox()
    self.live_max_width.setRange(64, 4096)
    self.live_frame_codec = base.QComboBox()
    self.live_frame_codec.addItems(list(LIVE_FRAME_CODECS))
    self.live_frame_codec.setToolTip("Codec used for webcam frames sent to the Colab live websocket. WebP can reduce in_kb when OpenCV supports it.")
    self.live_output_codec = base.QComboBox()
    self.live_output_codec.addItems(list(LIVE_FRAME_CODECS))
    self.live_output_codec.setToolTip("Codec used by the Colab server for frames returned to preview/virtual camera. JPEG is safest; WebP may reduce out_kb.")
    self.live_jpeg_quality = base.QSpinBox()
    self.live_jpeg_quality.setRange(20, 95)
    self.live_detector_size = base.QSpinBox()
    self.live_detector_size.setRange(160, 640)
    self.live_detector_size.setSingleStep(32)
    self.live_detect_every_n = base.QSpinBox()
    self.live_detect_every_n.setRange(1, 30)
    self.live_face_model_pack = base.QComboBox()
    self.live_face_model_pack.addItems(list(LIVE_FACE_MODEL_PACKS))
    self.live_face_model_pack.setToolTip(
        "buffalo_l is safest for inswapper_128; buffalo_m/s are experimental speed options. "
        "Use Swapper precision to compare fp32 vs fp16 swap_ms."
    )
    self.live_swapper_precision = base.QComboBox()
    self.live_swapper_precision.addItems(list(LIVE_SWAPPER_PRECISIONS))
    self.live_swapper_precision.setToolTip("Use fp32 as baseline; choose fp16 to test T4/RTX swap_ms.")
    self.live_cache_source_face = base.QCheckBox()
    self.live_cache_source_face.setToolTip("Keep on for speed. Turn off to re-read/re-analyze the source face each frame if a source swap looks stale.")
    self.live_preview_buffer_seconds = base.QDoubleSpinBox()
    self.live_preview_buffer_seconds.setRange(0.0, 5.0)
    self.live_preview_buffer_seconds.setSingleStep(0.25)
    self.live_preview_buffer_seconds.setDecimals(2)
    self.live_preview_buffer_seconds.setToolTip("Delay preview by this many seconds so frames can render at an even cadence.")

    options_form.addRow("Many faces", self.live_many_faces)
    options_form.addRow("Enhancer", self.live_enhancer)
    options_form.addRow("Opacity (1=full)", self.live_opacity)
    options_form.addRow("Sharpness (0=off)", self.live_sharpness)
    options_form.addRow("Mouth mask (0=off)", self.live_mouth_mask_size)
    options_form.addRow("Interpolation (0=off)", self.live_interpolation_weight)
    options_form.addRow("Poisson blend", self.live_poisson_blend)
    options_form.addRow("Color correction", self.live_color_correction)
    options_form.addRow("Process max width", self.live_max_width)
    options_form.addRow("Send codec", self.live_frame_codec)
    options_form.addRow("Return codec", self.live_output_codec)
    options_form.addRow("Frame quality", self.live_jpeg_quality)
    options_form.addRow("Detector size", self.live_detector_size)
    options_form.addRow("Detect every N frames", self.live_detect_every_n)
    options_form.addRow("InsightFace pack", self.live_face_model_pack)
    options_form.addRow("Swapper precision", self.live_swapper_precision)
    options_form.addRow("Cache source face", self.live_cache_source_face)
    options_form.addRow("Preview buffer seconds", self.live_preview_buffer_seconds)
    _apply_live_options_to_widgets(self)
    controls_layout.addWidget(options_box)

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
    controls_layout.addLayout(row)

    self.live_status = _status_label("Idle")
    controls_layout.addWidget(self.live_status)
    self.live_note = _status_label(
        "Live sends webcam JPEG/WebP frames to ws://HOST:PORT/ws/live and previews returned frames. "
        "buffalo_l is safest for inswapper_128; buffalo_m/s are experimental speed options. "
        "Use Swapper precision to compare fp32 vs fp16 swap_ms."
    )
    controls_layout.addWidget(self.live_note)
    controls_layout.addStretch(1)

    controls_scroll = QScrollArea()
    controls_scroll.setWidgetResizable(True)
    controls_scroll.setMinimumWidth(260)
    controls_scroll.setWidget(controls_panel)
    splitter.addWidget(controls_scroll)

    preview_panel = base.QWidget()
    preview_layout = base.QVBoxLayout(preview_panel)
    preview_controls = base.QHBoxLayout()
    preview_controls.addWidget(base.QLabel("Preview size"))
    self.live_preview_scale = base.QComboBox()
    self.live_preview_scale.addItems(list(LIVE_PREVIEW_SCALES))
    self.live_preview_scale.setCurrentText(str(_live_options(self.settings)["preview_scale"]))
    self.live_preview_scale.setToolTip("Fit fills the panel. 1x/1.5x/2x use that pixel scale only when it fits; otherwise they fall back to fit.")
    self.live_preview_scale.currentTextChanged.connect(lambda _text: update_live_preview_from_last_frame(self))
    preview_controls.addWidget(self.live_preview_scale)
    preview_controls.addStretch(1)
    preview_layout.addLayout(preview_controls)
    self.live_preview = base.QLabel("Live preview")
    self.live_preview.setAlignment(base.Qt.AlignCenter)
    self.live_preview.setMinimumSize(320, 240)
    self.live_preview.setWordWrap(True)
    preview_layout.addWidget(self.live_preview, 1)
    self._live_latest_jpeg = None
    self._live_preview_buffer = deque()
    self._live_preview_buffer_seconds = DEFAULT_LIVE_PREVIEW_BUFFER_SECONDS
    self._live_preview_frames = 0
    self._live_preview_last_frame = None
    self._live_preview_timer = base.QTimer(self)
    self._live_preview_timer.timeout.connect(lambda: render_live_preview_frame(self))
    splitter.addWidget(preview_panel)

    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([360, 900])
    layout.addWidget(splitter, 1)
    self.tabs.addTab(tab, "Live")


def sync_settings(self: base.MainWindow) -> None:
    _original_sync_settings(self)
    if hasattr(self, "live_source_face"):
        self.settings.source_face = self.live_source_face.text().strip()
    if hasattr(self, "live_width"):
        self.settings.live_width = int(self.live_width.value())
    else:
        self.settings.live_width = _live_setting(self.settings, "live_width", DEFAULT_LIVE_WIDTH)
    if hasattr(self, "live_height"):
        self.settings.live_height = int(self.live_height.value())
    else:
        self.settings.live_height = _live_setting(self.settings, "live_height", DEFAULT_LIVE_HEIGHT)
    if hasattr(self, "live_fps"):
        self.settings.live_fps = int(self.live_fps.value())
    else:
        self.settings.live_fps = _live_setting(self.settings, "live_fps", DEFAULT_LIVE_FPS)
    if hasattr(self, "live_pipeline_frames"):
        self.settings.live_pipeline_frames = int(self.live_pipeline_frames.value())
    else:
        self.settings.live_pipeline_frames = _live_setting(self.settings, "live_pipeline_frames", DEFAULT_LIVE_PIPELINE_FRAMES)
    self.settings.live_options = _read_live_options(self)
    base.save_settings(self.settings)


def closeEvent(self: base.MainWindow, event: Any) -> None:
    try:
        self.sync_settings()
    except Exception as exc:
        self.log(f"settings save on close failed: {exc}")
    stop_live_preview_timer(self)
    _original_close_event(self, event)


def _prepare_live_settings(settings: base.AppSettings) -> dict[str, Any]:
    client = base.ApiClient(settings)
    logs: list[str] = ["checking Colab API before starting live"]
    client.request_json("GET", "/health", timeout=5.0)

    live_settings = async_base._copy_settings(settings)
    live_settings.live_width = _live_setting(settings, "live_width", DEFAULT_LIVE_WIDTH)
    live_settings.live_height = _live_setting(settings, "live_height", DEFAULT_LIVE_HEIGHT)
    live_settings.live_fps = _live_setting(settings, "live_fps", DEFAULT_LIVE_FPS)
    live_settings.live_pipeline_frames = _live_setting(settings, "live_pipeline_frames", DEFAULT_LIVE_PIPELINE_FRAMES)
    live_settings.live_options = _live_options(settings)
    _apply_live_options_to_settings(live_settings)
    source_face = live_settings.source_face
    logs.append(f"live source face path: {source_face or '(empty)'}")
    if base.is_local_path(source_face):
        source_path = Path(source_face)
        if not source_path.is_file():
            raise FileNotFoundError(f"Local source face does not exist: {source_face}")
        upload_path, normalization_log = async_base._source_upload_path(source_path)
        if normalization_log:
            logs.append(normalization_log)
        logs.append(f"uploading local source face for live: {source_path}")
        response = client.upload_file("/upload/file?kind=source", upload_path, timeout=30.0)
        live_settings.source_face = str(response.get("path") or source_face)
        logs.append(f"live source uploaded to: {live_settings.source_face}")
    else:
        logs.append(f"using remote source face for live: {source_face}")

    return {"settings": live_settings, "logs": logs}


class LiveWorker(base.LiveWorker):
    async def _run_live(self) -> None:
        import cv2
        import websockets

        uri = self.settings.base_url.replace("http://", "ws://") + "/ws/live"
        self.message.emit(f"connecting live websocket: {uri}")
        cap = cv2.VideoCapture(self.settings.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera index {self.settings.camera_index}")
        requested_width = _live_setting(self.settings, "live_width", DEFAULT_LIVE_WIDTH)
        requested_height = _live_setting(self.settings, "live_height", DEFAULT_LIVE_HEIGHT)
        requested_fps = _live_setting(self.settings, "live_fps", DEFAULT_LIVE_FPS)
        pipeline_frames = _live_setting(self.settings, "live_pipeline_frames", DEFAULT_LIVE_PIPELINE_FRAMES)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
        cap.set(cv2.CAP_PROP_FPS, requested_fps)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.message.emit(
            f"webcam capture: requested {requested_width}x{requested_height}@{requested_fps}, "
            f"actual {actual_width}x{actual_height}@{actual_fps:.1f}, pipeline {pipeline_frames}"
        )
        virtual_cam = None
        frame_codec = str(getattr(self.settings, "frame_codec", DEFAULT_LIVE_FRAME_CODEC)).lower()
        if frame_codec not in LIVE_FRAME_CODECS:
            frame_codec = DEFAULT_LIVE_FRAME_CODEC
        output_codec = str(getattr(self.settings, "output_codec", DEFAULT_LIVE_OUTPUT_CODEC)).lower()
        if output_codec not in LIVE_FRAME_CODECS:
            output_codec = DEFAULT_LIVE_OUTPUT_CODEC
        frame_quality = int(getattr(self.settings, "live_jpeg_quality", DEFAULT_LIVE_JPEG_QUALITY))
        self.message.emit(f"live frame codec: send={frame_codec}, return={output_codec}, quality={frame_quality}")
        clock = asyncio.get_running_loop().time
        stats_started = clock()
        stats_frames = 0
        in_flight = 0
        condition = asyncio.Condition()

        async def sender(websocket: Any) -> None:
            nonlocal in_flight
            while not self._stop:
                async with condition:
                    while in_flight >= pipeline_frames and not self._stop:
                        await condition.wait()
                    if self._stop:
                        break
                    in_flight += 1
                try:
                    ok, frame = cap.read()
                    if not ok:
                        async with condition:
                            in_flight = max(0, in_flight - 1)
                            condition.notify_all()
                        await asyncio.sleep(0.03)
                        continue
                    encode_ext = ".webp" if frame_codec == "webp" else ".jpg"
                    encode_flag = int(getattr(cv2, "IMWRITE_WEBP_QUALITY", cv2.IMWRITE_JPEG_QUALITY)) if frame_codec == "webp" else int(cv2.IMWRITE_JPEG_QUALITY)
                    try:
                        ok, encoded = cv2.imencode(encode_ext, frame, [encode_flag, frame_quality])
                    except Exception:
                        if frame_codec != "webp":
                            raise
                        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), frame_quality])
                    if not ok:
                        async with condition:
                            in_flight = max(0, in_flight - 1)
                            condition.notify_all()
                        continue
                    await websocket.send(encoded.tobytes())
                except Exception:
                    async with condition:
                        in_flight = max(0, in_flight - 1)
                        condition.notify_all()
                    raise

        async def receiver(websocket: Any) -> None:
            nonlocal in_flight, stats_started, stats_frames, virtual_cam
            while not self._stop:
                reply = await websocket.recv()
                if isinstance(reply, str):
                    self.message.emit(reply)
                    payload = _json_payload(reply)
                    if "error" in payload:
                        raise RuntimeError(str(payload["error"]))
                    continue
                async with condition:
                    in_flight = max(0, in_flight - 1)
                    condition.notify_all()
                self.frame.emit(reply)
                stats_frames += 1
                now = clock()
                if now - stats_started >= 5.0:
                    self.message.emit(f"live throughput: {stats_frames / (now - stats_started):.1f} fps")
                    stats_started = now
                    stats_frames = 0
                if virtual_cam is None:
                    try:
                        import numpy as np
                        import pyvirtualcam

                        decoded = cv2.imdecode(np.frombuffer(reply, dtype=np.uint8), cv2.IMREAD_COLOR)
                        h, w = decoded.shape[:2]
                        virtual_cam = pyvirtualcam.Camera(width=w, height=h, fps=requested_fps, device=self.settings.virtual_camera or None)
                        self.message.emit(f"virtual camera opened: {virtual_cam.device} at {requested_fps} fps")
                    except Exception as exc:
                        self.message.emit(f"virtual camera unavailable: {exc}")
                        virtual_cam = False
                if virtual_cam:
                    import numpy as np

                    decoded = cv2.imdecode(np.frombuffer(reply, dtype=np.uint8), cv2.IMREAD_COLOR)
                    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                    virtual_cam.send(rgb)
                    virtual_cam.sleep_until_next_frame()

        try:
            async with websockets.connect(uri, max_size=8 * 1024 * 1024) as websocket:
                await websocket.send(
                    base.json.dumps(
                        {
                            "source_face": self.settings.source_face,
                            "many_faces": self.settings.many_faces,
                            "enhancer": self.settings.enhancer,
                            "opacity": self.settings.opacity,
                            "sharpness": self.settings.sharpness,
                            "mouth_mask_size": self.settings.mouth_mask_size,
                            "interpolation_weight": self.settings.interpolation_weight,
                            "poisson_blend": self.settings.poisson_blend,
                            "color_correction": self.settings.color_correction,
                            "max_width": self.settings.max_width,
                            "frame_codec": frame_codec,
                            "output_codec": output_codec,
                            "jpeg_quality": frame_quality,
                            "frame_quality": frame_quality,
                            "detector_size": getattr(self.settings, "detector_size", DEFAULT_LIVE_DETECTOR_SIZE),
                            "detect_every_n": getattr(self.settings, "detect_every_n", DEFAULT_LIVE_DETECT_EVERY_N),
                            "face_model_pack": getattr(self.settings, "face_model_pack", DEFAULT_LIVE_FACE_MODEL_PACK),
                            "swapper_precision": getattr(self.settings, "swapper_precision", DEFAULT_LIVE_SWAPPER_PRECISION),
                            "cache_source_face": getattr(self.settings, "cache_source_face", True),
                        }
                    )
                )
                ready = await websocket.recv()
                self.message.emit(f"live backend: {ready}")
                ready_payload = _json_payload(ready)
                if "error" in ready_payload:
                    raise RuntimeError(str(ready_payload["error"]))

                sender_task = asyncio.create_task(sender(websocket))
                receiver_task = asyncio.create_task(receiver(websocket))
                done, pending = await asyncio.wait(
                    {sender_task, receiver_task},
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                for task in done:
                    error = task.exception()
                    if error is not None:
                        raise error
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        finally:
            cap.release()
            if virtual_cam and hasattr(virtual_cam, "close"):
                virtual_cam.close()
            self.message.emit("live worker stopped")


def start_live(self: base.MainWindow) -> None:
    self.sync_settings()
    if self.live_worker and self.live_worker.isRunning():
        self.log("live already running")
        ui_base._set_process_status(self, "live", "Live already running")
        return

    async_base._ensure_output_worker_state(self)
    self.settings.live_options = _read_live_options(self)
    base.save_settings(self.settings)
    settings = async_base._copy_settings(self.settings)
    settings.live_width = _live_setting(self.settings, "live_width", DEFAULT_LIVE_WIDTH)
    settings.live_height = _live_setting(self.settings, "live_height", DEFAULT_LIVE_HEIGHT)
    settings.live_fps = _live_setting(self.settings, "live_fps", DEFAULT_LIVE_FPS)
    settings.live_pipeline_frames = _live_setting(self.settings, "live_pipeline_frames", DEFAULT_LIVE_PIPELINE_FRAMES)
    settings.live_options = _live_options(self.settings)
    _apply_live_options_to_settings(settings)
    self.log("starting live...")
    ui_base._set_process_status(self, "live", "Preparing live...")

    def task() -> dict[str, Any]:
        return _prepare_live_settings(settings)

    def succeeded(task_id: str, result: object) -> None:
        if task_id != getattr(self, "output_live_task_id", ""):
            return
        payload = result if isinstance(result, dict) else {}
        for line in payload.get("logs") or []:
            line_text = str(line)
            self.log(line_text)
            ui_base._set_process_status(self, "live", line_text)
        live_settings = payload.get("settings")
        if not isinstance(live_settings, base.AppSettings):
            text = "live failed before start: invalid prepared settings"
            self.log(text)
            ui_base._set_process_status(self, "live", text)
            return
        self.live_worker = LiveWorker(live_settings)
        self.live_worker.message.connect(lambda text: ui_base._poll_message(self, "live", text))
        self.live_worker.frame.connect(self.enqueue_live_preview_frame)
        start_live_preview_timer(self, live_settings)
        self.live_worker.start()
        ui_base._set_process_status(self, "live", f"Starting live on camera index {live_settings.camera_index}...")

    def failed(task_id: str, error: str) -> None:
        if task_id != getattr(self, "output_live_task_id", ""):
            return
        text = f"live failed before start: {error}"
        self.log(text)
        ui_base._set_process_status(self, "live", text)

    self.output_live_task_id = async_base._start_output_task(
        self,
        "Preparing live...",
        task,
        succeeded,
        failed,
    )


def start_live_preview_timer(self: base.MainWindow, settings: base.AppSettings) -> None:
    timer = getattr(self, "_live_preview_timer", None)
    if timer is None:
        return
    self._live_latest_jpeg = None
    self._live_preview_buffer = deque()
    self._live_preview_buffer_seconds = float(_live_options(settings)["preview_buffer_seconds"])
    self._live_preview_started = self._live_preview_buffer_seconds <= 0
    self._live_preview_frames = 0
    self._live_preview_last_frame = None
    fps = _live_setting(settings, "live_fps", DEFAULT_LIVE_FPS)
    interval_ms = max(1, int(round(1000.0 / max(1, fps))))
    timer.setInterval(interval_ms)
    timer.start()


def stop_live_preview_timer(self: base.MainWindow) -> None:
    timer = getattr(self, "_live_preview_timer", None)
    if timer is not None:
        timer.stop()
    self._live_latest_jpeg = None
    self._live_preview_last_frame = None
    buffer = getattr(self, "_live_preview_buffer", None)
    if buffer is not None:
        buffer.clear()


def stop_live(self: base.MainWindow) -> None:
    stop_live_preview_timer(self)
    _original_stop_live(self)


def enqueue_live_preview_frame(self: base.MainWindow, frame_bytes: bytes) -> None:
    # Buffer by arrival time so the QTimer can render frames at an even cadence
    # after a small delay. Do not coalesce during normal playback; render one
    # queued frame per timer tick. Drop only if the backlog exceeds a safety cap.
    buffer = getattr(self, "_live_preview_buffer", None)
    if buffer is None:
        buffer = deque()
        self._live_preview_buffer = buffer
    now = time.monotonic()
    buffer.append((now, bytes(frame_bytes)))
    buffer_seconds = float(getattr(self, "_live_preview_buffer_seconds", DEFAULT_LIVE_PREVIEW_BUFFER_SECONDS))
    fps = _live_setting(self.settings, "live_fps", DEFAULT_LIVE_FPS)
    max_frames = max(3, int(math.ceil((buffer_seconds + 2.0) * fps)))
    while len(buffer) > max_frames:
        buffer.popleft()


def render_live_preview_frame(self: base.MainWindow) -> None:
    buffer = getattr(self, "_live_preview_buffer", None)
    if not buffer:
        return
    buffer_seconds = float(getattr(self, "_live_preview_buffer_seconds", DEFAULT_LIVE_PREVIEW_BUFFER_SECONDS))
    if not getattr(self, "_live_preview_started", False):
        if time.monotonic() - buffer[0][0] < buffer_seconds:
            return
        self._live_preview_started = True
    _timestamp, frame_bytes = buffer.popleft()
    if buffer_seconds > 0:
        fps = _live_setting(self.settings, "live_fps", DEFAULT_LIVE_FPS)
        # If the producer outruns the preview for a while, keep the stream near
        # the target delay by dropping only the oldest excess frames.
        target_frames = max(1, int(round(buffer_seconds * fps)))
        max_frames = max(target_frames + fps, target_frames * 2)
        while len(buffer) > max_frames:
            buffer.popleft()
    if not frame_bytes:
        return
    update_live_preview(self, frame_bytes)


def _preview_scale_factor(scale: str) -> float | None:
    normalized = str(scale or DEFAULT_LIVE_PREVIEW_SCALE).lower()
    if normalized == "fit":
        return None
    try:
        return float(normalized.rstrip("x"))
    except ValueError:
        return None


def _preview_target_size(self: base.MainWindow, image: base.QImage) -> tuple[int, int]:
    panel_size = self.live_preview.size()
    panel_width = max(1, int(panel_size.width()))
    panel_height = max(1, int(panel_size.height()))
    factor = _preview_scale_factor(getattr(getattr(self, "live_preview_scale", None), "currentText", lambda: DEFAULT_LIVE_PREVIEW_SCALE)())
    if factor is not None:
        target_width = max(1, int(round(image.width() * factor)))
        target_height = max(1, int(round(image.height() * factor)))
        if target_width <= panel_width and target_height <= panel_height:
            return target_width, target_height
    image_ratio = image.width() / max(1, image.height())
    panel_ratio = panel_width / max(1, panel_height)
    if image_ratio >= panel_ratio:
        return panel_width, max(1, int(round(panel_width / image_ratio)))
    return max(1, int(round(panel_height * image_ratio))), panel_height


def update_live_preview_from_last_frame(self: base.MainWindow) -> None:
    frame = getattr(self, "_live_preview_last_frame", None)
    if frame:
        update_live_preview(self, frame, remember=False)


def update_live_preview(self: base.MainWindow, frame_bytes: bytes, remember: bool = True) -> None:
    image = base.QImage.fromData(frame_bytes)
    if image.isNull():
        try:
            import cv2
            import numpy as np

            decoded = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                return
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            image = base.QImage(rgb.data, width, height, channels * width, base.QImage.Format_RGB888).copy()
        except Exception:
            return
    if remember:
        self._live_preview_last_frame = bytes(frame_bytes)
    target_width, target_height = _preview_target_size(self, image)
    pixmap = base.QPixmap.fromImage(image).scaled(
        target_width,
        target_height,
        base.Qt.KeepAspectRatio,
        base.Qt.FastTransformation,
    )
    self.live_preview.setPixmap(pixmap)
    self._live_preview_frames = int(getattr(self, "_live_preview_frames", 0)) + 1
    if self._live_preview_frames == 1 or self._live_preview_frames % max(1, _live_setting(self.settings, "live_fps", DEFAULT_LIVE_FPS)) == 0:
        ui_base._set_process_status(
            self,
            "live",
            (
                f"Live buffered preview ({image.width()}x{image.height()} -> {pixmap.width()}x{pixmap.height()}, "
                f"size {getattr(getattr(self, 'live_preview_scale', None), 'currentText', lambda: DEFAULT_LIVE_PREVIEW_SCALE)()}, "
                f"buffer {float(getattr(self, '_live_preview_buffer_seconds', DEFAULT_LIVE_PREVIEW_BUFFER_SECONDS)):.2f}s, "
                f"rendered {self._live_preview_frames})"
            ),
        )


def install() -> None:
    base.load_settings = load_settings
    base.save_settings = save_settings
    base.MainWindow._build_live_tab = _build_live_tab
    base.MainWindow.sync_settings = sync_settings
    base.MainWindow.closeEvent = closeEvent
    base.MainWindow.start_live = start_live
    base.MainWindow.stop_live = stop_live
    base.MainWindow.enqueue_live_preview_frame = enqueue_live_preview_frame
    base.MainWindow.update_live_preview = update_live_preview
    base.MainWindow.update_live_preview_from_last_frame = update_live_preview_from_last_frame


install()
main = base.main
