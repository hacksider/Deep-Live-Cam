from __future__ import annotations

from typing import Any

from windows_app import async_outputs as async_base
from windows_app import ui_patches as ui_base
from windows_app import app as base


PROCESSING_OPTION_KEYS = (
    "recursive",
    "overwrite",
    "skip_processed",
    "many_faces",
    "enhancer",
    "opacity",
    "sharpness",
    "mouth_mask_size",
    "interpolation_weight",
    "poisson_blend",
    "color_correction",
)


def _default_processing_options() -> dict[str, Any]:
    defaults = base.AppSettings()
    return {key: getattr(defaults, key) for key in PROCESSING_OPTION_KEYS}


def _legacy_processing_options(data: dict[str, Any] | None = None) -> dict[str, Any]:
    options = _default_processing_options()
    if data:
        for key in PROCESSING_OPTION_KEYS:
            if key in data:
                options[key] = data[key]
    return options


def _coerce_processing_options(value: object, fallback: dict[str, Any]) -> dict[str, Any]:
    options = dict(fallback)
    if isinstance(value, dict):
        for key in PROCESSING_OPTION_KEYS:
            if key in value:
                options[key] = value[key]
    return options


def load_settings() -> base.AppSettings:
    data: dict[str, Any] = {}
    if base.APP_STATE.is_file():
        try:
            loaded = base.json.loads(base.APP_STATE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}

    defaults = base.asdict(base.AppSettings())
    valid_fields = set(base.AppSettings.__dataclass_fields__)
    kwargs = {key: data.get(key, defaults[key]) for key in valid_fields if key in defaults}
    settings = base.AppSettings(**kwargs)

    legacy = _legacy_processing_options(data)
    settings.photos_options = _coerce_processing_options(data.get("photos_options"), legacy)
    settings.videos_options = _coerce_processing_options(data.get("videos_options"), legacy)

    # Keep legacy flat fields aligned with Photos for the initial Photos tab build.
    _apply_processing_options_to_settings(settings, "photos")
    return settings


def save_settings(settings: base.AppSettings) -> None:
    data = base.asdict(settings)
    legacy = _legacy_processing_options(data)
    data["photos_options"] = _coerce_processing_options(getattr(settings, "photos_options", None), legacy)
    data["videos_options"] = _coerce_processing_options(getattr(settings, "videos_options", None), legacy)
    base.APP_STATE.write_text(base.json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _settings_options(settings: base.AppSettings, kind: str) -> dict[str, Any]:
    return _coerce_processing_options(
        getattr(settings, f"{kind}_options", None),
        _legacy_processing_options(base.asdict(settings)),
    )


def _apply_processing_options_to_settings(settings: base.AppSettings, kind: str) -> None:
    for key, value in _settings_options(settings, kind).items():
        setattr(settings, key, value)


def _read_processing_options(window: base.MainWindow, kind: str) -> dict[str, Any]:
    if kind == "videos" and hasattr(window, "v_enhancer"):
        return {
            "recursive": window.v_recursive.isChecked(),
            "overwrite": window.v_overwrite.isChecked(),
            "skip_processed": window.v_skip_processed.isChecked(),
            "many_faces": window.v_many_faces.isChecked(),
            "enhancer": window.v_enhancer.currentText(),
            "opacity": float(window.v_opacity.value()),
            "sharpness": float(window.v_sharpness.value()),
            "mouth_mask_size": float(window.v_mouth_mask_size.value()),
            "interpolation_weight": float(window.v_interpolation_weight.value()),
            "poisson_blend": window.v_poisson_blend.isChecked(),
            "color_correction": window.v_color_correction.isChecked(),
        }
    if kind == "photos" and hasattr(window, "enhancer"):
        return {
            "recursive": window.recursive.isChecked(),
            "overwrite": window.overwrite.isChecked(),
            "skip_processed": window.skip_processed.isChecked(),
            "many_faces": window.many_faces.isChecked(),
            "enhancer": window.enhancer.currentText(),
            "opacity": float(window.opacity.value()),
            "sharpness": float(window.sharpness.value()),
            "mouth_mask_size": float(window.mouth_mask_size.value()),
            "interpolation_weight": float(window.interpolation_weight.value()),
            "poisson_blend": window.poisson_blend.isChecked(),
            "color_correction": window.color_correction.isChecked(),
        }
    return _settings_options(window.settings, kind)


def _apply_processing_options_to_widgets(window: base.MainWindow, kind: str) -> None:
    options = _settings_options(window.settings, kind)
    if kind == "photos" and hasattr(window, "enhancer"):
        window.recursive.setChecked(bool(options["recursive"]))
        window.overwrite.setChecked(bool(options["overwrite"]))
        window.skip_processed.setChecked(bool(options["skip_processed"]))
        window.many_faces.setChecked(bool(options["many_faces"]))
        window.enhancer.setCurrentText(str(options["enhancer"]))
        window.opacity.setValue(float(options["opacity"]))
        window.sharpness.setValue(float(options["sharpness"]))
        window.mouth_mask_size.setValue(float(options["mouth_mask_size"]))
        window.interpolation_weight.setValue(float(options["interpolation_weight"]))
        window.poisson_blend.setChecked(bool(options["poisson_blend"]))
        window.color_correction.setChecked(bool(options["color_correction"]))
    if kind == "videos" and hasattr(window, "v_enhancer"):
        window.v_recursive.setChecked(bool(options["recursive"]))
        window.v_overwrite.setChecked(bool(options["overwrite"]))
        window.v_skip_processed.setChecked(bool(options["skip_processed"]))
        window.v_many_faces.setChecked(bool(options["many_faces"]))
        window.v_enhancer.setCurrentText(str(options["enhancer"]))
        window.v_opacity.setValue(float(options["opacity"]))
        window.v_sharpness.setValue(float(options["sharpness"]))
        window.v_mouth_mask_size.setValue(float(options["mouth_mask_size"]))
        window.v_interpolation_weight.setValue(float(options["interpolation_weight"]))
        window.v_poisson_blend.setChecked(bool(options["poisson_blend"]))
        window.v_color_correction.setChecked(bool(options["color_correction"]))


def sync_settings(self: base.MainWindow) -> None:
    self.settings.host = self.host.text().strip()
    self.settings.port = int(self.port.value())
    self.settings.drive_root = self.drive_root.text().strip()
    self.settings.source_face = self.source_face.text().strip()
    self.settings.photos_input = self.photos_input.text().strip()
    self.settings.photos_output = self.photos_output.text().strip()
    self.settings.videos_input = self.videos_input.text().strip()
    self.settings.videos_output = self.videos_output.text().strip()

    self.settings.photos_options = _read_processing_options(self, "photos")
    self.settings.videos_options = _read_processing_options(self, "videos")

    active_kind = "videos" if self.tabs.tabText(self.tabs.currentIndex()) == "Videos" else "photos"
    _apply_processing_options_to_settings(self.settings, active_kind)

    self.settings.max_fps = float(self.max_fps.value())
    self.settings.max_width = int(self.max_width.value())
    self.settings.quality = int(self.quality.value())
    self.settings.start_pct = float(self.start_pct.value())
    self.settings.end_pct = float(self.end_pct.value())
    self.settings.camera_index = int(self.camera_index.value())
    self.settings.virtual_camera = self.virtual_camera.text().strip()
    base.save_settings(self.settings)


def _start_batch_with_status(self: base.MainWindow, kind: str) -> None:
    self.sync_settings()
    _apply_processing_options_to_settings(self.settings, kind)
    base.save_settings(self.settings)
    async_base._ensure_output_worker_state(self)
    settings = async_base._copy_settings(self.settings)
    self.log(f"starting {kind} batch...")
    ui_base._set_process_status(self, kind, f"Starting {kind} batch...")
    ui_base._set_batch_button_running(self, kind)

    def task() -> dict[str, Any]:
        return async_base._prepare_and_start_batch(settings, kind)

    def on_job_finished(status: str, batch_kind: str) -> None:
        self.log(f"job finished: {status}")
        ui_base._set_process_status(self, batch_kind, f"Job finished: {status}")
        self.active_job_id = None
        ui_base._set_batch_button_idle(self, batch_kind)

    def succeeded(task_id: str, result: object) -> None:
        if task_id != self.output_batch_task_id:
            return
        payload = result if isinstance(result, dict) else {}
        for line in payload.get("logs") or []:
            line_text = str(line)
            self.log(line_text)
            ui_base._set_process_status(self, kind, line_text)
        response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
        self.active_job_id = response.get("job_id")
        if self.active_job_id:
            if self.poller:
                self.poller.stop()
            self.poller = base.PollWorker(self.client, self.active_job_id)
            self.poller.message.connect(lambda text, batch_kind=kind: ui_base._poll_message(self, batch_kind, text))
            self.poller.finished_status.connect(lambda status, batch_kind=kind: on_job_finished(status, batch_kind))
            self.poller.start()
            ui_base._set_process_status(self, kind, f"{kind} batch running...")
        else:
            ui_base._set_batch_button_idle(self, kind)
            ui_base._set_process_status(self, kind, f"{kind} batch failed to start")

    def failed(task_id: str, error: str) -> None:
        if task_id != self.output_batch_task_id:
            return
        text = f"{kind} batch failed before start: {error}"
        ui_base._set_process_status(self, kind, text)
        self.log(text)
        ui_base._set_batch_button_idle(self, kind)

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


_original_build_photos_tab = base.MainWindow._build_photos_tab
_original_build_videos_tab = base.MainWindow._build_videos_tab


def _build_photos_tab(self: base.MainWindow) -> None:
    _original_build_photos_tab(self)
    _apply_processing_options_to_widgets(self, "photos")


def _build_videos_tab(self: base.MainWindow) -> None:
    _original_build_videos_tab(self)
    _apply_processing_options_to_widgets(self, "videos")


def install() -> None:
    base.load_settings = load_settings
    base.save_settings = save_settings
    base.MainWindow._build_photos_tab = _build_photos_tab
    base.MainWindow._build_videos_tab = _build_videos_tab
    base.MainWindow.sync_settings = sync_settings
    base.MainWindow.start_photos = start_photos
    base.MainWindow.start_videos = start_videos


install()
main = base.main
