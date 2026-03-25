import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from modules.gpu_processing import gpu_cvt_color, gpu_resize, gpu_flip
from PIL import Image, ImageOps
import time
import json
import queue
import threading
import subprocess
import re
import sys
import numpy as np
import modules.globals
import modules.metadata
from modules.face_analyser import (
    get_one_face,
    get_many_faces,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    add_blank_map,
    has_valid_map,
    simplify_maps,
    reset_face_analyser_state,
)
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    is_image,
    is_video,
    resolve_relative_path,
    has_image_extension,
)
from modules.video_capture import VideoCapturer
from modules.gettext import LanguageManager
from modules import globals
import platform

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

# --- Tk 9.0 compatibility patch ---
# In Tk 9.0, Menu.index("end") returns "" instead of raising TclError
# when the menu is empty. CustomTkinter's CTkOptionMenu doesn't handle
# this consistently in some environments. Keep a safe fallback rebuild.
try:
    from customtkinter.windows.widgets.core_widget_classes import (
        DropdownMenu as _DropdownMenu,
    )

    _original_add_menu_commands = _DropdownMenu._add_menu_commands

    def _patched_add_menu_commands(self, *args, **kwargs):
        try:
            return _original_add_menu_commands(self, *args, **kwargs)
        except Exception:
            # Fallback for Tk variants where menu internals behave differently.
            try:
                self.delete(0, "end")
            except Exception:
                pass
            values = list(getattr(self, "_values", []))
            min_width = int(getattr(self, "_min_character_width", 0))
            for value in values:
                label = value.ljust(min_width)
                if sys.platform.startswith("linux"):
                    label = f"  {label}  "
                self.add_command(
                    label=label,
                    command=lambda v=value: self._button_callback(v),
                    compound="left",
                )
            return None

    _DropdownMenu._add_menu_commands = _patched_add_menu_commands
except (ImportError, AttributeError):
    pass  # CustomTkinter version doesn't have this class path
# --- End Tk 9.0 patch ---

ROOT = None
POPUP = None
POPUP_LIVE = None
ROOT_HEIGHT = 900
ROOT_WIDTH = 760

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
# 480p resolution for better FPS and more natural look
PREVIEW_DEFAULT_WIDTH = 854  # 480p width (16:9)
PREVIEW_DEFAULT_HEIGHT = 480  # 480p height
LIVE_DETECTION_INTERVAL = 0.033
DISPLAY_LOOP_INTERVAL_MS = 16
LIVE_INTERNAL_PROCESS_WIDTH = 320
LIVE_INTERNAL_PROCESS_HEIGHT = 180
LIVE_PROCESS_EVERY_N_FRAMES = 1

if platform.system() == "Darwin":
    PREVIEW_DEFAULT_WIDTH = 854
    PREVIEW_DEFAULT_HEIGHT = 480
    LIVE_DETECTION_INTERVAL = 0.02
    DISPLAY_LOOP_INTERVAL_MS = 12
    LIVE_PROCESS_EVERY_N_FRAMES = 1

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = (740,)
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = (890,)
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_MAX_HEIGHT = 100
MAPPER_PREVIEW_MAX_WIDTH = 100

DEFAULT_BUTTON_WIDTH = 200
DEFAULT_BUTTON_HEIGHT = 40

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

_ = None
preview_label = None
preview_tk_image = None
preview_slider = None
source_label = None
target_label = None
status_label = None
popup_status_label = None
popup_status_label_live = None
source_label_dict = {}
source_label_dict_live = {}
target_label_dict_live = {}
camera_optionmenu = None
camera_variable = None
camera_indices = []
camera_names = []
live_button = None
transparency_value_label = None
sharpness_value_label = None
enhancer_variables = {}
face_engine_selector = None
face_engine_variable = None
mlx_detector_selector = None
mlx_detector_variable = None

NO_CAMERAS_LABEL = "No cameras found"
ENHANCER_KEYS = ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512")
FACE_ENGINE_OPTIONS = {
    "InsightFace (ONNX)": "insightface",
    "MLX UniFace (Apple Silicon)": "mlx_uniface",
}
FACE_ENGINE_OPTIONS_REVERSE = {value: key for key, value in FACE_ENGINE_OPTIONS.items()}
MLX_DETECTOR_OPTIONS = {
    "RetinaFace (MLX)": "retinaface",
}
MLX_DETECTOR_OPTIONS_REVERSE = {
    value: key for key, value in MLX_DETECTOR_OPTIONS.items()
}

_enhancer_toggle_guard = False
_processor_state_lock = threading.Lock()
_processor_state_version = 0
_status_lock = threading.Lock()
_deferred_status_text: str | None = None
_deferred_popup_status_text: str | None = None
_deferred_popup_live_status_text: str | None = None
_status_poller_started = False

img_ft, vid_ft = modules.globals.file_types


def _bump_processor_state_version() -> None:
    global _processor_state_version
    with _processor_state_lock:
        _processor_state_version += 1


def _get_processor_state_version() -> int:
    with _processor_state_lock:
        return _processor_state_version


def _flush_deferred_status_updates() -> None:
    global \
        _deferred_status_text, \
        _deferred_popup_status_text, \
        _deferred_popup_live_status_text

    if ROOT is None:
        return
    try:
        if not ROOT.winfo_exists():
            return
    except Exception:
        return

    with _status_lock:
        status_text = _deferred_status_text
        popup_text = _deferred_popup_status_text
        popup_live_text = _deferred_popup_live_status_text
        _deferred_status_text = None
        _deferred_popup_status_text = None
        _deferred_popup_live_status_text = None

    if status_text is not None and status_label is not None:
        try:
            if status_label.winfo_exists():
                status_label.configure(text=status_text)
                ROOT.update_idletasks()
        except Exception:
            pass

    if popup_text is not None and popup_status_label is not None:
        try:
            if popup_status_label.winfo_exists():
                popup_status_label.configure(text=popup_text)
        except Exception:
            pass

    if popup_live_text is not None and popup_status_label_live is not None:
        try:
            if popup_status_label_live.winfo_exists():
                popup_status_label_live.configure(text=popup_live_text)
        except Exception:
            pass

    try:
        ROOT.after(33, _flush_deferred_status_updates)
    except Exception:
        pass


def _ensure_status_poller_started() -> None:
    global _status_poller_started
    if _status_poller_started or ROOT is None:
        return
    try:
        if not ROOT.winfo_exists():
            return
        _status_poller_started = True
        ROOT.after(33, _flush_deferred_status_updates)
    except Exception:
        pass


def _normalize_enhancer_states() -> None:
    enabled_enhancers = [
        enhancer_key
        for enhancer_key in ENHANCER_KEYS
        if modules.globals.fp_ui.get(enhancer_key, False)
    ]
    if len(enabled_enhancers) <= 1:
        return

    preferred_order = (
        "face_enhancer_gpen512",
        "face_enhancer_gpen256",
        "face_enhancer",
    )
    active_enhancer = next(
        (
            enhancer_key
            for enhancer_key in preferred_order
            if enhancer_key in enabled_enhancers
        ),
        enabled_enhancers[0],
    )
    for enhancer_key in ENHANCER_KEYS:
        modules.globals.fp_ui[enhancer_key] = enhancer_key == active_enhancer


def _get_face_engine_label() -> str:
    engine = getattr(modules.globals, "face_analyser_engine", "insightface")
    return FACE_ENGINE_OPTIONS_REVERSE.get(engine, "InsightFace (ONNX)")


def _get_mlx_detector_label() -> str:
    detector = getattr(modules.globals, "mlx_face_detector", "retinaface")
    return MLX_DETECTOR_OPTIONS_REVERSE.get(detector, "RetinaFace (MLX)")


def _set_mlx_detector_selector_state() -> None:
    if mlx_detector_selector is None:
        return
    using_mlx = (
        getattr(modules.globals, "face_analyser_engine", "insightface") == "mlx_uniface"
    )
    mlx_detector_selector.configure(state=("readonly" if using_mlx else "disabled"))


def _apply_face_engine_selection() -> None:
    if face_engine_variable is None:
        return
    selected_label = face_engine_variable.get()
    selected_engine = FACE_ENGINE_OPTIONS.get(selected_label, "insightface")
    modules.globals.face_analyser_engine = selected_engine
    reset_face_analyser_state()
    _set_mlx_detector_selector_state()
    save_switch_states()
    if selected_engine == "mlx_uniface":
        update_status(
            "MLX face analyser selected. Requires Apple Silicon + mlx-uniface."
        )
    else:
        update_status("InsightFace face analyser selected.")


def _apply_mlx_detector_selection() -> None:
    if mlx_detector_variable is None:
        return
    selected_label = mlx_detector_variable.get()
    selected_detector = MLX_DETECTOR_OPTIONS.get(selected_label, "retinaface")
    modules.globals.mlx_face_detector = selected_detector
    reset_face_analyser_state()
    save_switch_states()
    update_status(f"MLX detector set to {selected_detector}.")


def init(start: Callable[[], None], destroy: Callable[[], None], lang: str) -> ctk.CTk:
    global ROOT, PREVIEW, _

    lang_manager = LanguageManager(lang)
    _ = lang_manager._
    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def save_switch_states():
    switch_states = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "opacity": modules.globals.opacity,
        "sharpness": modules.globals.sharpness,
        "selected_camera_name": camera_variable.get()
        if camera_variable
        else modules.globals.camera_input_combobox,
        "face_analyser_engine": modules.globals.face_analyser_engine,
        "mlx_face_detector": modules.globals.mlx_face_detector,
    }
    with open("switch_states.json", "w") as f:
        json.dump(switch_states, f)


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            switch_states = json.load(f)
        modules.globals.keep_fps = switch_states.get("keep_fps", True)
        modules.globals.keep_audio = switch_states.get("keep_audio", True)
        modules.globals.keep_frames = switch_states.get("keep_frames", False)
        modules.globals.many_faces = switch_states.get("many_faces", False)
        modules.globals.map_faces = switch_states.get("map_faces", False)
        modules.globals.poisson_blend = switch_states.get("poisson_blend", False)
        modules.globals.color_correction = switch_states.get("color_correction", False)
        modules.globals.nsfw_filter = switch_states.get("nsfw_filter", False)
        modules.globals.live_mirror = switch_states.get("live_mirror", False)
        modules.globals.live_resizable = switch_states.get("live_resizable", False)
        loaded_fp_ui = switch_states.get("fp_ui", {})
        modules.globals.fp_ui = {**modules.globals.fp_ui, **loaded_fp_ui}
        _normalize_enhancer_states()
        modules.globals.show_fps = switch_states.get("show_fps", False)
        modules.globals.mouth_mask = switch_states.get("mouth_mask", False)
        modules.globals.show_mouth_mask_box = switch_states.get(
            "show_mouth_mask_box", False
        )
        modules.globals.opacity = float(
            switch_states.get("opacity", modules.globals.opacity)
        )
        modules.globals.sharpness = float(
            switch_states.get("sharpness", modules.globals.sharpness)
        )
        modules.globals.camera_input_combobox = switch_states.get(
            "selected_camera_name"
        )
        modules.globals.face_analyser_engine = switch_states.get(
            "face_analyser_engine", modules.globals.face_analyser_engine
        )
        modules.globals.mlx_face_detector = switch_states.get(
            "mlx_face_detector", modules.globals.mlx_face_detector
        )
        if modules.globals.mlx_face_detector not in MLX_DETECTOR_OPTIONS_REVERSE:
            modules.globals.mlx_face_detector = "retinaface"
    except FileNotFoundError:
        # If the file doesn't exist, use default values
        pass


def _get_selected_camera_index() -> int | None:
    if not camera_variable or not camera_names or camera_names[0] == NO_CAMERAS_LABEL:
        return None
    selected = camera_variable.get()
    if selected not in camera_names:
        return None
    return camera_indices[camera_names.index(selected)]


def _set_camera_widgets_state() -> None:
    has_camera = bool(camera_names and camera_names[0] != NO_CAMERAS_LABEL)
    if camera_optionmenu is not None:
        camera_optionmenu.configure(state=("readonly" if has_camera else "disabled"))
    if live_button is not None:
        live_button.configure(state=("normal" if has_camera else "disabled"))


def refresh_camera_list(show_status: bool = True) -> None:
    global camera_indices, camera_names

    preferred_camera = (
        camera_variable.get()
        if camera_variable
        else modules.globals.camera_input_combobox
    )
    camera_indices, camera_names = get_available_cameras()

    values = camera_names if camera_names else [NO_CAMERAS_LABEL]
    if camera_optionmenu is not None:
        camera_optionmenu.configure(values=values)

    if camera_variable is not None:
        selected_name = None
        if preferred_camera in camera_names:
            selected_name = preferred_camera
        elif camera_names:
            selected_name = camera_names[0]
        else:
            selected_name = NO_CAMERAS_LABEL
        camera_variable.set(selected_name)
        modules.globals.camera_input_combobox = selected_name

    _set_camera_widgets_state()
    save_switch_states()

    if show_status:
        if camera_names and camera_names[0] != NO_CAMERAS_LABEL:
            update_status(f"Detected {len(camera_names)} camera(s).")
        else:
            update_status("No cameras found.")


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label, show_fps_switch
    global camera_optionmenu, camera_variable, live_button
    global transparency_value_label, sharpness_value_label, enhancer_variables
    global \
        face_engine_selector, \
        face_engine_variable, \
        mlx_detector_selector, \
        mlx_detector_variable

    load_switch_states()

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = ctk.CTk()
    root.geometry(f"{ROOT_WIDTH}x{ROOT_HEIGHT}")
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(
        f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
    )
    root.configure()
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())

    # --- Header ---
    header_card = ctk.CTkFrame(root, corner_radius=16, border_width=1)
    header_card.place(relx=0.03, rely=0.02, relwidth=0.94, relheight=0.08)

    title_label = ctk.CTkLabel(
        header_card,
        text="Deep Live Cam",
        font=ctk.CTkFont(size=30, weight="bold"),
        anchor="w",
    )
    title_label.place(relx=0.03, rely=0.12, relwidth=0.60, relheight=0.50)

    subtitle_label = ctk.CTkLabel(
        header_card,
        text=_("Real-time Face Swap"),
        text_color=("gray30", "gray75"),
        anchor="w",
    )
    subtitle_label.place(relx=0.03, rely=0.60, relwidth=0.60, relheight=0.30)

    meta_label = ctk.CTkLabel(
        header_card,
        text=f"v{modules.metadata.version}",
        corner_radius=10,
        fg_color=("gray84", "gray22"),
        text_color=("gray18", "gray90"),
    )
    meta_label.place(relx=0.83, rely=0.26, relwidth=0.14, relheight=0.48)

    # --- Media Selection Card ---
    media_card = ctk.CTkFrame(root, corner_radius=16, border_width=1)
    media_card.place(relx=0.03, rely=0.11, relwidth=0.94, relheight=0.29)

    source_box = ctk.CTkFrame(media_card, corner_radius=14, border_width=1)
    source_box.place(relx=0.03, rely=0.08, relwidth=0.43, relheight=0.62)
    source_box_title = ctk.CTkLabel(
        source_box, text=_("Source"), font=ctk.CTkFont(size=15, weight="bold")
    )
    source_box_title.place(relx=0.03, rely=0.03, relwidth=0.40, relheight=0.12)

    source_label = ctk.CTkLabel(source_box, text=None)
    source_label.place(relx=0.03, rely=0.17, relwidth=0.94, relheight=0.79)

    target_box = ctk.CTkFrame(media_card, corner_radius=14, border_width=1)
    target_box.place(relx=0.54, rely=0.08, relwidth=0.43, relheight=0.62)
    target_box_title = ctk.CTkLabel(
        target_box, text=_("Target"), font=ctk.CTkFont(size=15, weight="bold")
    )
    target_box_title.place(relx=0.03, rely=0.03, relwidth=0.40, relheight=0.12)

    target_label = ctk.CTkLabel(target_box, text=None)
    target_label.place(relx=0.03, rely=0.17, relwidth=0.94, relheight=0.79)

    select_face_button = ctk.CTkButton(
        media_card,
        text=_("Select a face"),
        cursor="hand2",
        command=lambda: select_source_path(),
    )
    select_face_button.place(relx=0.03, rely=0.77, relwidth=0.33, relheight=0.17)

    swap_faces_button = ctk.CTkButton(
        media_card, text="Swap", cursor="hand2", command=lambda: swap_faces_paths()
    )
    swap_faces_button.place(relx=0.38, rely=0.77, relwidth=0.24, relheight=0.17)

    select_target_button = ctk.CTkButton(
        media_card,
        text=_("Select a target"),
        cursor="hand2",
        command=lambda: select_target_path(),
    )
    select_target_button.place(relx=0.64, rely=0.77, relwidth=0.33, relheight=0.17)

    # --- Processing Cards ---
    processing_card = ctk.CTkFrame(root, corner_radius=16, border_width=1)
    processing_card.place(relx=0.03, rely=0.42, relwidth=0.46, relheight=0.38)
    processing_label = ctk.CTkLabel(
        processing_card,
        text=_("Processing"),
        anchor="w",
        font=ctk.CTkFont(size=18, weight="bold"),
    )
    processing_label.place(relx=0.05, rely=0.03, relwidth=0.90, relheight=0.09)

    live_card = ctk.CTkFrame(root, corner_radius=16, border_width=1)
    live_card.place(relx=0.51, rely=0.42, relwidth=0.46, relheight=0.38)
    live_label = ctk.CTkLabel(
        live_card,
        text=_("Live & Enhancers"),
        anchor="w",
        font=ctk.CTkFont(size=18, weight="bold"),
    )
    live_label.place(relx=0.05, rely=0.03, relwidth=0.90, relheight=0.09)

    def create_switch(
        parent: ctk.CTkFrame,
        text: str,
        variable: ctk.BooleanVar,
        command: Callable[[], None],
        rely: float,
        relheight: float = 0.08,
    ) -> ctk.CTkSwitch:
        switch = ctk.CTkSwitch(
            parent, text=text, variable=variable, cursor="hand2", command=command
        )
        switch.place(relx=0.06, rely=rely, relwidth=0.88, relheight=relheight)
        return switch

    face_engine_label = ctk.CTkLabel(live_card, text=_("Face Engine"), anchor="w")
    face_engine_label.place(relx=0.06, rely=0.13, relwidth=0.34, relheight=0.06)

    face_engine_variable = ctk.StringVar(value=_get_face_engine_label())
    face_engine_selector = ctk.CTkComboBox(
        live_card,
        variable=face_engine_variable,
        values=list(FACE_ENGINE_OPTIONS.keys()),
        state="readonly",
        command=lambda _value: _apply_face_engine_selection(),
    )
    face_engine_selector.place(relx=0.40, rely=0.13, relwidth=0.54, relheight=0.07)

    mlx_detector_label = ctk.CTkLabel(live_card, text=_("MLX Detector"), anchor="w")
    mlx_detector_label.place(relx=0.06, rely=0.22, relwidth=0.34, relheight=0.06)

    mlx_detector_variable = ctk.StringVar(value=_get_mlx_detector_label())
    mlx_detector_selector = ctk.CTkComboBox(
        live_card,
        variable=mlx_detector_variable,
        values=list(MLX_DETECTOR_OPTIONS.keys()),
        state="disabled",
        command=lambda _value: _apply_mlx_detector_selection(),
    )
    mlx_detector_selector.place(relx=0.40, rely=0.22, relwidth=0.54, relheight=0.07)
    _set_mlx_detector_selector_state()

    keep_fps_value = ctk.BooleanVar(value=modules.globals.keep_fps)
    keep_fps_checkbox = create_switch(
        processing_card,
        text=_("Keep fps"),
        variable=keep_fps_value,
        command=lambda: (
            setattr(modules.globals, "keep_fps", keep_fps_value.get()),
            save_switch_states(),
        ),
        rely=0.15,
    )

    keep_frames_value = ctk.BooleanVar(value=modules.globals.keep_frames)
    keep_frames_switch = create_switch(
        processing_card,
        text=_("Keep frames"),
        variable=keep_frames_value,
        command=lambda: (
            setattr(modules.globals, "keep_frames", keep_frames_value.get()),
            save_switch_states(),
        ),
        rely=0.26,
    )

    keep_audio_value = ctk.BooleanVar(value=modules.globals.keep_audio)
    keep_audio_switch = create_switch(
        live_card,
        text=_("Keep audio"),
        variable=keep_audio_value,
        command=lambda: (
            setattr(modules.globals, "keep_audio", keep_audio_value.get()),
            save_switch_states(),
        ),
        rely=0.33,
    )

    many_faces_value = ctk.BooleanVar(value=modules.globals.many_faces)
    many_faces_switch = create_switch(
        live_card,
        text=_("Many faces"),
        variable=many_faces_value,
        command=lambda: (
            setattr(modules.globals, "many_faces", many_faces_value.get()),
            save_switch_states(),
        ),
        rely=0.41,
    )

    color_correction_value = ctk.BooleanVar(value=modules.globals.color_correction)
    color_correction_switch = create_switch(
        live_card,
        text=_("Fix Blueish Cam"),
        variable=color_correction_value,
        command=lambda: (
            setattr(modules.globals, "color_correction", color_correction_value.get()),
            save_switch_states(),
        ),
        rely=0.49,
    )

    map_faces = ctk.BooleanVar(value=modules.globals.map_faces)
    map_faces_switch = create_switch(
        processing_card,
        text=_("Map faces"),
        variable=map_faces,
        command=lambda: (
            setattr(modules.globals, "map_faces", map_faces.get()),
            save_switch_states(),
            close_mapper_window() if not map_faces.get() else None,
        ),
        rely=0.37,
    )

    poisson_blend_value = ctk.BooleanVar(value=modules.globals.poisson_blend)
    poisson_blend_switch = create_switch(
        processing_card,
        text=_("Poisson Blend"),
        variable=poisson_blend_value,
        command=lambda: (
            setattr(modules.globals, "poisson_blend", poisson_blend_value.get()),
            save_switch_states(),
        ),
        rely=0.48,
    )

    show_fps_value = ctk.BooleanVar(value=modules.globals.show_fps)
    show_fps_switch = create_switch(
        live_card,
        text=_("Show FPS"),
        variable=show_fps_value,
        command=lambda: (
            setattr(modules.globals, "show_fps", show_fps_value.get()),
            save_switch_states(),
        ),
        rely=0.57,
    )

    mouth_mask_var = ctk.BooleanVar(value=modules.globals.mouth_mask)
    mouth_mask_switch = create_switch(
        processing_card,
        text=_("Mouth Mask"),
        variable=mouth_mask_var,
        command=lambda: (
            setattr(modules.globals, "mouth_mask", mouth_mask_var.get()),
            save_switch_states(),
        ),
        rely=0.59,
    )

    show_mouth_mask_box_var = ctk.BooleanVar(value=modules.globals.show_mouth_mask_box)
    show_mouth_mask_box_switch = create_switch(
        processing_card,
        text=_("Show Mouth Mask Box"),
        variable=show_mouth_mask_box_var,
        command=lambda: (
            setattr(
                modules.globals, "show_mouth_mask_box", show_mouth_mask_box_var.get()
            ),
            save_switch_states(),
        ),
        rely=0.70,
    )

    enhancer_hint = ctk.CTkLabel(
        live_card,
        text=_("Enhancers (exclusive mode)"),
        text_color=("gray36", "gray70"),
        anchor="w",
    )
    enhancer_hint.place(relx=0.06, rely=0.65, relwidth=0.88, relheight=0.06)

    enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui["face_enhancer"])
    create_switch(
        live_card,
        text=_("Face Enhancer"),
        variable=enhancer_value,
        command=lambda: update_tumbler("face_enhancer", enhancer_value.get()),
        rely=0.72,
        relheight=0.07,
    )

    gpen256_value = ctk.BooleanVar(
        value=modules.globals.fp_ui.get("face_enhancer_gpen256", False)
    )
    create_switch(
        live_card,
        text=_("GPEN Enhancer 256"),
        variable=gpen256_value,
        command=lambda: update_tumbler("face_enhancer_gpen256", gpen256_value.get()),
        rely=0.80,
        relheight=0.07,
    )

    gpen512_value = ctk.BooleanVar(
        value=modules.globals.fp_ui.get("face_enhancer_gpen512", False)
    )
    create_switch(
        live_card,
        text=_("GPEN Enhancer 512"),
        variable=gpen512_value,
        command=lambda: update_tumbler("face_enhancer_gpen512", gpen512_value.get()),
        rely=0.88,
        relheight=0.07,
    )

    enhancer_variables = {
        "face_enhancer": enhancer_value,
        "face_enhancer_gpen256": gpen256_value,
        "face_enhancer_gpen512": gpen512_value,
    }

    # --- Fine Tuning ---
    controls_label = ctk.CTkLabel(
        root, text=_("Fine Tuning"), font=ctk.CTkFont(size=18, weight="bold")
    )
    controls_label.place(relx=0.03, rely=0.81, relwidth=0.30, relheight=0.03)

    tuning_card = ctk.CTkFrame(root, corner_radius=16, border_width=1)
    tuning_card.place(relx=0.03, rely=0.84, relwidth=0.94, relheight=0.10)

    transparency_var = ctk.DoubleVar(value=float(modules.globals.opacity))

    def on_transparency_change(value: float):
        val = float(value)
        modules.globals.opacity = val
        percentage = int(val * 100)
        if transparency_value_label is not None:
            transparency_value_label.configure(text=f"{percentage}%")
        save_switch_states()

        if percentage == 0:
            modules.globals.face_swapper_enabled = False
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif percentage == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {percentage}%")

    transparency_label = ctk.CTkLabel(tuning_card, text="Transparency:")
    transparency_label.place(relx=0.03, rely=0.12, relwidth=0.16, relheight=0.32)

    transparency_value_label = ctk.CTkLabel(
        tuning_card,
        text=f"{int(float(modules.globals.opacity) * 100)}%",
        text_color=("gray35", "gray70"),
    )
    transparency_value_label.place(relx=0.20, rely=0.12, relwidth=0.08, relheight=0.32)

    transparency_slider = ctk.CTkSlider(
        tuning_card,
        from_=0.0,
        to=1.0,
        variable=transparency_var,
        command=on_transparency_change,
        height=5,
        border_width=1,
        corner_radius=3,
    )
    transparency_slider.place(relx=0.30, rely=0.22, relwidth=0.67, relheight=0.12)

    sharpness_var = ctk.DoubleVar(value=float(modules.globals.sharpness))

    def on_sharpness_change(value: float):
        modules.globals.sharpness = float(value)
        if sharpness_value_label is not None:
            sharpness_value_label.configure(text=f"{float(value):.1f}")
        save_switch_states()
        update_status(f"Sharpness set to {value:.1f}")

    sharpness_label = ctk.CTkLabel(tuning_card, text="Sharpness:")
    sharpness_label.place(relx=0.03, rely=0.54, relwidth=0.16, relheight=0.32)

    sharpness_value_label = ctk.CTkLabel(
        tuning_card,
        text=f"{float(modules.globals.sharpness):.1f}",
        text_color=("gray35", "gray70"),
    )
    sharpness_value_label.place(relx=0.20, rely=0.54, relwidth=0.08, relheight=0.32)

    sharpness_slider = ctk.CTkSlider(
        tuning_card,
        from_=0.0,
        to=5.0,
        variable=sharpness_var,
        command=on_sharpness_change,
        height=5,
        border_width=1,
        corner_radius=3,
    )
    sharpness_slider.place(relx=0.30, rely=0.64, relwidth=0.67, relheight=0.12)

    # --- Actions / Camera ---
    action_row = ctk.CTkFrame(root, corner_radius=16, border_width=1)
    action_row.place(relx=0.03, rely=0.93, relwidth=0.94, relheight=0.05)

    start_button = ctk.CTkButton(
        action_row,
        text=_("Start"),
        cursor="hand2",
        command=lambda: analyze_target(start, root),
    )
    start_button.place(relx=0.01, rely=0.16, relwidth=0.16, relheight=0.68)

    stop_button = ctk.CTkButton(
        action_row, text=_("Stop"), cursor="hand2", command=lambda: destroy()
    )
    stop_button.place(relx=0.18, rely=0.16, relwidth=0.16, relheight=0.68)

    preview_button = ctk.CTkButton(
        action_row, text=_("Preview"), cursor="hand2", command=lambda: toggle_preview()
    )
    preview_button.place(relx=0.35, rely=0.16, relwidth=0.16, relheight=0.68)

    # --- Camera Selection ---
    camera_label = ctk.CTkLabel(action_row, text=_("Camera:"), anchor="e")
    camera_label.place(relx=0.53, rely=0.10, relwidth=0.10, relheight=0.80)

    camera_variable = ctk.StringVar(value=NO_CAMERAS_LABEL)
    camera_optionmenu = ctk.CTkComboBox(
        action_row,
        variable=camera_variable,
        values=[NO_CAMERAS_LABEL],
        state="disabled",
    )
    camera_optionmenu.place(relx=0.64, rely=0.16, relwidth=0.20, relheight=0.68)
    camera_variable.trace_add(
        "write",
        lambda *_: (
            setattr(modules.globals, "camera_input_combobox", camera_variable.get()),
            save_switch_states(),
        ),
    )

    refresh_camera_button = ctk.CTkButton(
        action_row,
        text=_("Refresh"),
        cursor="hand2",
        command=lambda: refresh_camera_list(show_status=True),
    )
    refresh_camera_button.place(relx=0.85, rely=0.16, relwidth=0.07, relheight=0.68)

    live_button = ctk.CTkButton(
        action_row,
        text=_("Live"),
        cursor="hand2",
        command=lambda: webcam_preview(root, _get_selected_camera_index()),
        state="disabled",
    )
    live_button.place(relx=0.93, rely=0.16, relwidth=0.06, relheight=0.68)

    refresh_camera_list(show_status=False)
    # --- End Camera Selection ---

    # Status and link
    global status_label
    status_label = ctk.CTkLabel(root, text=None, justify="left", anchor="w")
    status_label.place(relx=0.03, rely=0.981, relwidth=0.58, relheight=0.018)

    donate_label = ctk.CTkLabel(
        root, text="Deep Live Cam", justify="right", anchor="e", cursor="hand2"
    )
    donate_label.place(relx=0.62, rely=0.981, relwidth=0.35, relheight=0.018)
    donate_label.configure(
        text_color=ctk.ThemeManager.theme.get("URL").get("text_color")
    )
    donate_label.bind(
        "<Button>", lambda event: webbrowser.open("https://deeplivecam.net")
    )

    _ensure_status_poller_started()

    return root


def close_mapper_window():
    global POPUP, POPUP_LIVE
    if POPUP and POPUP.winfo_exists():
        POPUP.destroy()
        POPUP = None
    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        POPUP_LIVE.destroy()
        POPUP_LIVE = None


def analyze_target(start: Callable[[], None], root: ctk.CTk):
    if POPUP != None and POPUP.winfo_exists():
        update_status("Please complete pop-up or close it.")
        return

    if modules.globals.map_faces:
        modules.globals.source_target_map = []

        if is_image(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_image()
        elif is_video(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_video()

        if len(modules.globals.source_target_map) > 0:
            create_source_target_popup(start, root, modules.globals.source_target_map)
        else:
            update_status("No faces found in target")
    else:
        select_output_path(start)


def create_source_target_popup(
    start: Callable[[], None], root: ctk.CTk, map: list
) -> None:
    global POPUP, popup_status_label

    POPUP = ctk.CTkToplevel(root)
    POPUP.title(_("Source x Target Mapper"))
    POPUP.geometry(f"{POPUP_WIDTH}x{POPUP_HEIGHT}")
    POPUP.focus()

    def on_submit_click(start):
        if has_valid_map():
            POPUP.destroy()
            select_output_path(start)
        else:
            update_pop_status("Atleast 1 source with target is required!")

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP, width=POPUP_SCROLL_WIDTH, height=POPUP_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

    def on_button_click(map, button_num):
        map = update_popup_source(scrollable_frame, map, button_num)

    for item in map:
        id = item["id"]

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select source image"),
            command=lambda id=id: on_button_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=0, padx=50, pady=10)

        x_label = ctk.CTkLabel(
            scrollable_frame,
            text=f"X",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=id, column=2, padx=10, pady=10)

        image = Image.fromarray(gpu_cvt_color(item["target"]["cv2"], cv2.COLOR_BGR2RGB))
        image = image.resize(
            (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        tk_image = ctk.CTkImage(image, size=image.size)

        target_image = ctk.CTkLabel(
            scrollable_frame,
            text=f"T-{id}",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        target_image.grid(row=id, column=3, padx=10, pady=10)
        target_image.configure(image=tk_image)

    popup_status_label = ctk.CTkLabel(POPUP, text=None, justify="center")
    popup_status_label.grid(row=1, column=0, pady=15)

    close_button = ctk.CTkButton(
        POPUP, text=_("Submit"), command=lambda: on_submit_click(start)
    )
    close_button.grid(row=2, column=0, pady=10)


def update_popup_source(
    scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global source_label_dict

    source_path = ctk.filedialog.askopenfilename(
        title=_("select an source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "source" in map[button_num]:
        map[button_num].pop("source")
        source_label_dict[button_num].destroy()
        del source_label_dict[button_num]

    if source_path == "":
        return map
    else:
        cv2_img = cv2.imread(source_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["source"] = {
                "cv2": cv2_img[int(y_min) : int(y_max), int(x_min) : int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                gpu_cvt_color(map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=button_num, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)
            source_label_dict[button_num] = source_image
        else:
            update_pop_status("Face could not be detected in last upload!")
        return map


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider, preview_tk_image

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title(_("Preview"))
    preview.configure()
    preview.protocol("WM_DELETE_WINDOW", lambda: toggle_preview())
    preview.resizable(width=True, height=True)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)
    preview_tk_image = None

    preview_slider = ctk.CTkSlider(
        preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value)
    )

    return preview


def update_status(text: str) -> None:
    translated = _(text) if _ else text

    if threading.current_thread() is not threading.main_thread():
        with _status_lock:
            global _deferred_status_text
            _deferred_status_text = translated
        return

    def _apply() -> None:
        if status_label is not None and status_label.winfo_exists():
            status_label.configure(text=translated)
        if ROOT is not None and ROOT.winfo_exists():
            ROOT.update_idletasks()

    _apply()


def update_pop_status(text: str) -> None:
    translated = _(text) if _ else text

    if threading.current_thread() is not threading.main_thread():
        with _status_lock:
            global _deferred_popup_status_text
            _deferred_popup_status_text = translated
        return

    def _apply() -> None:
        if popup_status_label is not None and popup_status_label.winfo_exists():
            popup_status_label.configure(text=translated)

    _apply()


def update_pop_live_status(text: str) -> None:
    translated = _(text) if _ else text

    if threading.current_thread() is not threading.main_thread():
        with _status_lock:
            global _deferred_popup_live_status_text
            _deferred_popup_live_status_text = translated
        return

    def _apply() -> None:
        if (
            popup_status_label_live is not None
            and popup_status_label_live.winfo_exists()
        ):
            popup_status_label_live.configure(text=translated)

    _apply()


def update_tumbler(var: str, value: bool) -> None:
    global _enhancer_toggle_guard

    if var in ENHANCER_KEYS and _enhancer_toggle_guard:
        return

    if var in ENHANCER_KEYS and value:
        _enhancer_toggle_guard = True
        try:
            for enhancer_key in ENHANCER_KEYS:
                is_enabled = enhancer_key == var
                modules.globals.fp_ui[enhancer_key] = is_enabled
                enhancer_var = enhancer_variables.get(enhancer_key)
                if enhancer_var is not None and enhancer_var.get() != is_enabled:
                    enhancer_var.set(is_enabled)
        finally:
            _enhancer_toggle_guard = False
    else:
        modules.globals.fp_ui[var] = value

    _bump_processor_state_version()
    save_switch_states()


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(
        title=_("select an source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )
    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
        update_status(f"Source selected: {os.path.basename(source_path)}")
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)
        update_status("Source cleared.")


def swap_faces_paths() -> None:
    global RECENT_DIRECTORY_SOURCE, RECENT_DIRECTORY_TARGET

    source_path = modules.globals.source_path
    target_path = modules.globals.target_path

    if not is_image(source_path) or not is_image(target_path):
        return

    modules.globals.source_path = target_path
    modules.globals.target_path = source_path

    RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
    RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)

    PREVIEW.withdraw()

    source_image = render_image_preview(modules.globals.source_path, (200, 200))
    source_label.configure(image=source_image)

    target_image = render_image_preview(modules.globals.target_path, (200, 200))
    target_label.configure(image=target_image)


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(
        title=_("select an target image or video"),
        initialdir=RECENT_DIRECTORY_TARGET,
        filetypes=[img_ft, vid_ft],
    )
    if is_image(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        image = render_image_preview(modules.globals.target_path, (200, 200))
        target_label.configure(image=image)
        update_status(f"Target image selected: {os.path.basename(target_path)}")
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
        update_status(f"Target video selected: {os.path.basename(target_path)}")
    else:
        modules.globals.target_path = None
        target_label.configure(image=None)
        update_status("Target cleared.")


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title=_("save image output file"),
            filetypes=[img_ft],
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title=_("save video output file"),
            filetypes=[vid_ft],
            defaultextension=".mp4",
            initialfile="output.mp4",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    else:
        output_path = None
    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
        start()


def check_and_ignore_nsfw(target, destroy: Callable = None) -> bool:
    """Check if the target is NSFW.
    TODO: Consider to make blur the target.
    """
    from numpy import ndarray

    try:
        from modules.predicter import predict_image, predict_video, predict_frame
    except ModuleNotFoundError:
        update_status(
            "NSFW filter dependencies are missing on this platform. Continuing."
        )
        return False

    check_nsfw = None
    if isinstance(target, str):  # image/video file path
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):  # frame object
        check_nsfw = predict_frame
    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(
                to_quit=False
            )  # Do not need to destroy the window frame if the target is NSFW
        update_status("Processing ignored!")
        return True
    else:
        return False


def _ensure_bgr_uint8_frame(frame):
    if frame is None:
        return None
    if not isinstance(frame, np.ndarray):
        return None
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        try:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None
    if frame.ndim == 3 and frame.shape[2] == 4:
        try:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception:
            return None
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame
    return None


def fit_image_to_size(image, width: int, height: int):
    if image is None:
        return image
    if width is None or height is None or width <= 1 or height <= 1:
        return image

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image

    scale = min(width / w, height / h)
    if scale <= 0:
        return image

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return image

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return gpu_resize(image, dsize=(new_w, new_h), interpolation=interpolation)


def fit_image_to_canvas(image, width: int, height: int):
    image = _ensure_bgr_uint8_frame(image)
    if image is None:
        return None
    if width is None or height is None or width <= 1 or height <= 1:
        return image

    fitted = fit_image_to_size(image, width, height)
    fitted = _ensure_bgr_uint8_frame(fitted)
    if fitted is None:
        return None

    out_h, out_w = fitted.shape[:2]
    if out_w == width and out_h == height:
        return fitted

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = max(0, (width - out_w) // 2)
    y0 = max(0, (height - out_h) // 2)
    x1 = min(width, x0 + out_w)
    y1 = min(height, y0 + out_h)
    canvas[y0:y1, x0:x1] = fitted[: y1 - y0, : x1 - x0]
    return canvas


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == "normal":
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill="x")
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    global preview_tk_image
    if modules.globals.source_path and modules.globals.target_path:
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        for frame_processor in get_frame_processors_modules(
            modules.globals.frame_processors
        ):
            temp_frame = frame_processor.process_frame(
                get_one_face(cv2.imread(modules.globals.source_path)), temp_frame
            )
        image = Image.fromarray(gpu_cvt_color(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        if preview_tk_image is None or preview_tk_image.cget("size") != image.size:
            preview_tk_image = ctk.CTkImage(image, size=image.size)
        else:
            preview_tk_image.configure(
                light_image=image, dark_image=image, size=image.size
            )
        preview_label.configure(image=preview_tk_image)
        preview_label.image = preview_tk_image
        update_status("Processing succeed!")
        PREVIEW.deiconify()


def webcam_preview(root: ctk.CTk, camera_index: int):
    global POPUP_LIVE

    if camera_index is None:
        update_status("No camera selected.")
        return

    if not modules.globals.map_faces:
        if modules.globals.source_path is None:
            update_status("Please select a source image first")
            return
        source_frame = cv2.imread(modules.globals.source_path)
        if source_frame is None:
            update_status("Unable to read source image.")
            return
        if get_one_face(source_frame) is None:
            update_status("No face detected in the selected source image.")
            return
    else:
        # Live must always open camera directly, even with map mode enabled.
        # Mapping can be configured separately without blocking startup.
        if POPUP_LIVE and POPUP_LIVE.winfo_exists():
            POPUP_LIVE.destroy()
            POPUP_LIVE = None
        if not modules.globals.simple_map and not has_valid_map():
            update_status("Live opened. Configure face maps to apply swaps.")

    create_webcam_preview(camera_index)


def _get_macos_cameras_via_ffmpeg():
    """Return AVFoundation video devices as (indices, names)."""
    try:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-f",
                "avfoundation",
                "-list_devices",
                "true",
                "-i",
                "",
            ],
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        )
    except Exception:
        return [], []

    output = f"{proc.stdout}\n{proc.stderr}"
    camera_indices = []
    camera_names = []
    in_video_section = False

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if "AVFoundation video devices" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices" in line:
            in_video_section = False
            continue
        if not in_video_section:
            continue
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match:
            idx = int(match.group(1))
            name = match.group(2).strip()
            lower_name = name.lower()
            # Skip ffmpeg "screen capture" pseudo-devices from camera list.
            if "capture screen" in lower_name:
                continue
            camera_indices.append(idx)
            camera_names.append(name)

    return camera_indices, camera_names


def _probe_macos_cameras_with_opencv(max_index: int = 8) -> Tuple[list[int], list[str]]:
    """Probe AVFoundation indices when ffmpeg list_devices is unavailable."""
    camera_indices: list[int] = []
    camera_names: list[str] = []
    for idx in range(max_index):
        cap = None
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                continue
            has_frame, _ = cap.read()
            if has_frame:
                camera_indices.append(idx)
                camera_names.append(f"Camera {idx}")
        except Exception:
            continue
        finally:
            if cap is not None:
                cap.release()
    return camera_indices, camera_names


def get_available_cameras():
    """Returns a list of available camera names and indices."""
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()

            # Create list of indices and names
            camera_indices = list(range(len(devices)))
            camera_names = devices

            # If no cameras found through DirectShow, try OpenCV fallback
            if not camera_names:
                # Try to open camera with index -1 and 0
                test_indices = [-1, 0]
                working_cameras = []

                for idx in test_indices:
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        working_cameras.append(f"Camera {idx}")
                        cap.release()

                if working_cameras:
                    return test_indices[: len(working_cameras)], working_cameras

            # If still no cameras found, return empty lists
            if not camera_names:
                return [], [NO_CAMERAS_LABEL]

            return camera_indices, camera_names

        except Exception as e:
            print(f"Error detecting cameras: {str(e)}")
            return [], [NO_CAMERAS_LABEL]
    else:
        # Unix-like systems (Linux/Mac) camera detection
        camera_indices = []
        camera_names = []

        if platform.system() == "Darwin":
            # Avoid probing random indices with OpenCV on macOS.
            # Parse AVFoundation device list from ffmpeg output instead.
            camera_indices, camera_names = _get_macos_cameras_via_ffmpeg()
            if not camera_names:
                camera_indices, camera_names = _probe_macos_cameras_with_opencv()
        else:
            # Linux camera detection - test first 10 indices
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_indices.append(i)
                    camera_names.append(f"Camera {i}")
                    cap.release()

        if not camera_names:
            return [], [NO_CAMERAS_LABEL]

        return camera_indices, camera_names


def _capture_thread_func(cap, capture_queue, stop_event):
    """Capture thread: reads frames from camera and puts them into the queue.
    Drops frames when the queue is full to avoid backpressure on the camera."""
    flat_frame_streak = 0
    flat_stream_warned = False
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break

        try:
            sample = frame
            if sample is not None and sample.ndim == 3 and sample.shape[2] >= 3:
                sample = sample[::8, ::8, :3]
            elif sample is not None and sample.ndim == 2:
                sample = sample[::8, ::8]
            stream_std = float(np.std(sample)) if sample is not None else 0.0
            if stream_std < 2.0:
                flat_frame_streak += 1
            else:
                flat_frame_streak = 0
            if flat_frame_streak >= 20 and not flat_stream_warned:
                update_status(
                    "Camera feed appears flat/gray. Check selected camera device and avoid screen-capture virtual inputs."
                )
                flat_stream_warned = True
        except Exception:
            pass

        try:
            capture_queue.put_nowait(frame)
        except queue.Full:
            # Drop the oldest frame and enqueue the new one
            try:
                capture_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                capture_queue.put_nowait(frame)
            except queue.Full:
                pass


def _is_informative_camera_frame(frame: np.ndarray) -> bool:
    frame = _ensure_bgr_uint8_frame(frame)
    if frame is None:
        return False
    sample = frame[::8, ::8, :3]
    std_val = float(np.std(sample))
    dynamic_range = float(np.max(sample) - np.min(sample))
    return std_val >= 3.0 and dynamic_range >= 10.0


def _start_validated_camera_capture(
    preferred_index: int,
    width: int,
    height: int,
    fps: int,
):
    candidate_indices: list[int] = []
    for idx in [preferred_index, *camera_indices]:
        if idx is None or idx in candidate_indices:
            continue
        candidate_indices.append(idx)

    for idx in candidate_indices:
        cap = VideoCapturer(idx)
        if not cap.start(width, height, fps):
            continue

        # Reduced validation frames for faster startup on macOS
        informative_hits = 0
        for _ in range(10):  # Was 20
            ret, frame = cap.read()
            if ret and _is_informative_camera_frame(frame):
                informative_hits += 1
                if informative_hits >= 2:
                    return cap, idx
            time.sleep(0.01)  # Reduced from 0.01

        cap.release()

    return None, None


def _detection_thread_func(
    latest_frame_holder, detection_result, detection_lock, stop_event
):
    """Detection thread: continuously runs face detection on the latest
    captured frame and stores results in detection_result under detection_lock.

    This decouples face detection (~15-30ms) from face swapping (~5-10ms)
    so the swap loop never blocks on detection, significantly improving
    live mode FPS."""
    last_detection_time = 0.0
    last_valid_face = None
    last_valid_many_faces = None
    last_valid_ts = 0.0
    # Reduced stale face TTL for more responsive updates on macOS
    stale_face_ttl = 0.25  # Was 0.35
    mlx_no_face_streak = 0
    mlx_adaptive_warned = False
    mlx_fallback_probe_interval = 45
    # Optimized detection interval for macOS
    detection_interval = LIVE_DETECTION_INTERVAL
    while not stop_event.is_set():
        with detection_lock:
            frame = latest_frame_holder[0]

        if frame is None:
            time.sleep(0.005)
            continue
        now = time.perf_counter()
        if now - last_detection_time < detection_interval:
            time.sleep(0.001)
            continue

        try:
            using_mlx_engine = (
                getattr(modules.globals, "face_analyser_engine", "insightface")
                == "mlx_uniface"
            )
            # Adaptive fallback strategy optimized for realtime:
            # - MLX path stays primary.
            # - Fallback is sparse to avoid tanking FPS.
            if not using_mlx_engine:
                allow_fallback = True
            else:
                allow_fallback = (
                    mlx_no_face_streak >= mlx_fallback_probe_interval
                    and (mlx_no_face_streak % mlx_fallback_probe_interval) == 0
                )

            if modules.globals.many_faces:
                many = get_many_faces(
                    frame, require_embedding=False, allow_fallback=allow_fallback
                )
                if many:
                    mlx_no_face_streak = 0
                    last_valid_many_faces = many
                    last_valid_ts = now
                elif (
                    last_valid_many_faces is not None
                    and (now - last_valid_ts) <= stale_face_ttl
                ):
                    many = last_valid_many_faces
                else:
                    mlx_no_face_streak += 1
                with detection_lock:
                    detection_result["target_face"] = None
                    detection_result["many_faces"] = many
            else:
                face = get_one_face(
                    frame, require_embedding=False, allow_fallback=allow_fallback
                )
                if face is not None:
                    mlx_no_face_streak = 0
                    last_valid_face = face
                    last_valid_ts = now
                elif (
                    last_valid_face is not None
                    and (now - last_valid_ts) <= stale_face_ttl
                ):
                    face = last_valid_face
                else:
                    mlx_no_face_streak += 1
                with detection_lock:
                    detection_result["target_face"] = face
                    detection_result["many_faces"] = None

            if (
                using_mlx_engine
                and mlx_no_face_streak >= 10
                and not mlx_adaptive_warned
            ):
                update_status(
                    "MLX detector is missing faces in live mode. Adaptive fallback enabled to keep swaps active."
                )
                mlx_adaptive_warned = True
            elif mlx_no_face_streak == 0:
                mlx_adaptive_warned = False
            last_detection_time = now
        except Exception as e:
            print(f"Detection thread error: {e}")
            time.sleep(0.01)


def _processing_thread_func(
    capture_queue,
    processed_queue,
    stop_event,
    latest_frame_holder,
    detection_result,
    detection_lock,
    initial_source_face,
):
    """Processing thread: takes raw frames from capture_queue, reads the
    latest detection result from the shared detection_result dict, applies
    face swap/enhancement, and puts results into processed_queue.

    Face detection runs concurrently in _detection_thread_func — this thread
    only reads cached results so it never blocks on detection."""
    frame_processors = []
    last_processor_state_version = -1
    source_image = initial_source_face
    last_source_path = None
    prev_time = time.time()
    fps_update_interval = 0.5
    frame_count = 0
    fps = 0
    map_fallback_warned = False
    live_frame_index = 0
    gpen512_cadence_warned = False
    last_processed_frame = None
    is_live_mode = bool(getattr(modules.globals, "live_mode", False))

    while not stop_event.is_set():
        current_processor_state_version = _get_processor_state_version()
        if current_processor_state_version != last_processor_state_version:
            frame_processors = get_frame_processors_modules(
                modules.globals.frame_processors
            )
            last_processor_state_version = current_processor_state_version

        try:
            frame = capture_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        raw_frame = _ensure_bgr_uint8_frame(frame)
        if raw_frame is None:
            continue
        temp_frame = raw_frame
        live_frame_index += 1

        if modules.globals.live_mirror:
            temp_frame = gpu_flip(temp_frame, 1)
            mirrored = _ensure_bgr_uint8_frame(temp_frame)
            temp_frame = mirrored if mirrored is not None else raw_frame

        # Process live frames at lower internal resolution to reduce swap/detection cost.
        if is_live_mode and (
            temp_frame.shape[1] > LIVE_INTERNAL_PROCESS_WIDTH
            or temp_frame.shape[0] > LIVE_INTERNAL_PROCESS_HEIGHT
        ):
            try:
                temp_frame = gpu_resize(
                    temp_frame,
                    (LIVE_INTERNAL_PROCESS_WIDTH, LIVE_INTERNAL_PROCESS_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
                resized = _ensure_bgr_uint8_frame(temp_frame)
                if resized is not None:
                    temp_frame = resized
                else:
                    temp_frame = raw_frame
            except Exception:
                temp_frame = raw_frame

        # Publish the mirrored frame for the detection thread to pick up
        with detection_lock:
            latest_frame_holder[0] = temp_frame.copy()

        source_path = modules.globals.source_path
        if source_path and source_path != last_source_path:
            last_source_path = source_path
            source_frame = cv2.imread(source_path)
            source_image = (
                get_one_face(source_frame, require_embedding=True, allow_fallback=True)
                if source_frame is not None
                else None
            )

        # Read latest detection results (brief lock to avoid blocking detection thread)
        with detection_lock:
            cached_target_face = detection_result.get("target_face")
            cached_many_faces = detection_result.get("many_faces")

        skip_expensive_live_processing = (
            is_live_mode
            and last_processed_frame is not None
            and (live_frame_index % LIVE_PROCESS_EVERY_N_FRAMES) != 0
        )
        if skip_expensive_live_processing:
            temp_frame = last_processed_frame.copy()
        else:
            has_live_map = bool(modules.globals.simple_map) or has_valid_map()
            use_simple_live_mode = (not modules.globals.map_faces) or (not has_live_map)
            if modules.globals.map_faces and not has_live_map:
                if not map_fallback_warned:
                    update_status(
                        "Map faces is ON but no map is loaded. Using simple live swap."
                    )
                    map_fallback_warned = True
            else:
                map_fallback_warned = False

            if use_simple_live_mode:
                for frame_processor in frame_processors:
                    try:
                        if frame_processor.NAME == "DLC.FACE-ENHANCER":
                            if modules.globals.fp_ui["face_enhancer"]:
                                if modules.globals.many_faces and cached_many_faces:
                                    temp_frame = frame_processor.enhance_faces(
                                        temp_frame, cached_many_faces
                                    )
                                elif cached_target_face is not None:
                                    temp_frame = frame_processor.enhance_faces(
                                        temp_frame, [cached_target_face]
                                    )
                        elif frame_processor.NAME == "DLC.FACE-ENHANCER-GPEN256":
                            if modules.globals.fp_ui.get(
                                "face_enhancer_gpen256", False
                            ):
                                if modules.globals.many_faces and cached_many_faces:
                                    for detected_face in cached_many_faces:
                                        temp_frame = frame_processor.enhance_face(
                                            temp_frame, detected_face
                                        )
                                elif cached_target_face is not None:
                                    temp_frame = frame_processor.enhance_face(
                                        temp_frame, cached_target_face
                                    )
                        elif frame_processor.NAME == "DLC.FACE-ENHANCER-GPEN512":
                            if modules.globals.fp_ui.get(
                                "face_enhancer_gpen512", False
                            ):
                                enhance_interval = 1
                                if platform.system() == "Darwin" and getattr(
                                    modules.globals, "live_mode", False
                                ):
                                    enhance_interval = 3
                                    if not gpen512_cadence_warned:
                                        update_status(
                                            "Live GPEN512 running in adaptive cadence (every 3 frames) for FPS."
                                        )
                                        gpen512_cadence_warned = True

                                should_enhance = (
                                    live_frame_index % enhance_interval
                                ) == 0
                                if should_enhance:
                                    if modules.globals.many_faces and cached_many_faces:
                                        for detected_face in cached_many_faces:
                                            temp_frame = frame_processor.enhance_face(
                                                temp_frame, detected_face
                                            )
                                    elif cached_target_face is not None:
                                        temp_frame = frame_processor.enhance_face(
                                            temp_frame, cached_target_face
                                        )
                        elif frame_processor.NAME == "DLC.FACE-SWAPPER":
                            # Use cached face positions from detection thread
                            swapped_bboxes = []
                            if modules.globals.many_faces and cached_many_faces:
                                result = temp_frame.copy()
                                for t_face in cached_many_faces:
                                    result = frame_processor.swap_face(
                                        source_image, t_face, result
                                    )
                                    if (
                                        hasattr(t_face, "bbox")
                                        and t_face.bbox is not None
                                    ):
                                        swapped_bboxes.append(t_face.bbox.astype(int))
                                temp_frame = result
                            elif cached_target_face is not None:
                                temp_frame = frame_processor.swap_face(
                                    source_image, cached_target_face, temp_frame
                                )
                                if (
                                    hasattr(cached_target_face, "bbox")
                                    and cached_target_face.bbox is not None
                                ):
                                    swapped_bboxes.append(
                                        cached_target_face.bbox.astype(int)
                                    )
                            # Apply post-processing (sharpening, interpolation)
                            temp_frame = frame_processor.apply_post_processing(
                                temp_frame, swapped_bboxes
                            )
                        else:
                            temp_frame = frame_processor.process_frame(
                                source_image, temp_frame
                            )
                    except Exception as e:
                        print(
                            f"Processing thread: frame processor {getattr(frame_processor, 'NAME', 'UNKNOWN')} failed: {e}"
                        )
                        continue
            else:
                modules.globals.target_path = None
                for frame_processor in frame_processors:
                    try:
                        if frame_processor.NAME == "DLC.FACE-ENHANCER":
                            if modules.globals.fp_ui["face_enhancer"]:
                                temp_frame = frame_processor.process_frame_v2(
                                    temp_frame
                                )
                        elif frame_processor.NAME in (
                            "DLC.FACE-ENHANCER-GPEN256",
                            "DLC.FACE-ENHANCER-GPEN512",
                        ):
                            fp_key = (
                                frame_processor.NAME.split(".")[-1]
                                .lower()
                                .replace("-", "_")
                            )
                            if modules.globals.fp_ui.get(fp_key, False):
                                if (
                                    frame_processor.NAME == "DLC.FACE-ENHANCER-GPEN512"
                                    and platform.system() == "Darwin"
                                    and getattr(modules.globals, "live_mode", False)
                                    and (live_frame_index % 3) != 0
                                ):
                                    pass
                                else:
                                    temp_frame = frame_processor.process_frame_v2(
                                        temp_frame
                                    )
                        else:
                            temp_frame = frame_processor.process_frame_v2(temp_frame)
                    except Exception as e:
                        print(
                            f"Processing thread: frame processor {getattr(frame_processor, 'NAME', 'UNKNOWN')} failed: {e}"
                        )
                        continue

        temp_frame = _ensure_bgr_uint8_frame(temp_frame)
        if temp_frame is None:
            temp_frame = raw_frame.copy()
        elif is_live_mode and (
            temp_frame.shape[1] != raw_frame.shape[1]
            or temp_frame.shape[0] != raw_frame.shape[0]
        ):
            try:
                upscaled = gpu_resize(
                    temp_frame,
                    (raw_frame.shape[1], raw_frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                upscaled = _ensure_bgr_uint8_frame(upscaled)
                if upscaled is not None:
                    temp_frame = upscaled
            except Exception:
                pass

        last_processed_frame = temp_frame.copy()

        # Calculate and display FPS
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= fps_update_interval:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time

        if modules.globals.show_fps:
            cv2.putText(
                temp_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # Put processed frame into output queue, dropping old frames if full
        try:
            processed_queue.put_nowait(temp_frame)
        except queue.Full:
            try:
                processed_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                processed_queue.put_nowait(temp_frame)
            except queue.Full:
                pass


def create_webcam_preview(camera_index: int):
    global preview_label, PREVIEW, preview_tk_image

    initial_source_face = None
    source_path = modules.globals.source_path
    if source_path:
        source_frame = cv2.imread(source_path)
        if source_frame is not None:
            initial_source_face = get_one_face(
                source_frame, require_embedding=True, allow_fallback=True
            )
    if not modules.globals.map_faces and initial_source_face is None:
        update_status(
            "Live started, but no source face is locked. Select a clearer frontal source image for swapping."
        )

    modules.globals.live_mode = True
    target_fps = 24 if platform.system() == "Darwin" else 60
    cap, actual_camera_index = _start_validated_camera_capture(
        camera_index,
        PREVIEW_DEFAULT_WIDTH,
        PREVIEW_DEFAULT_HEIGHT,
        target_fps,
    )
    if cap is None:
        modules.globals.live_mode = False
        update_status(
            "Failed to start camera. On macOS, allow camera access for Terminal/iTerm in System Settings > Privacy & Security > Camera."
        )
        return
    if actual_camera_index != camera_index:
        try:
            if actual_camera_index in camera_indices:
                resolved_name = camera_names[camera_indices.index(actual_camera_index)]
                if camera_variable is not None:
                    camera_variable.set(resolved_name)
                modules.globals.camera_input_combobox = resolved_name
                save_switch_states()
                update_status(
                    f"Selected camera stream looked invalid. Switched to {resolved_name}."
                )
        except Exception:
            pass

    preview_label.configure(width=PREVIEW_DEFAULT_WIDTH, height=PREVIEW_DEFAULT_HEIGHT)
    preview_tk_image = None
    if modules.globals.live_resizable:
        PREVIEW.resizable(width=True, height=True)
    else:
        PREVIEW.resizable(width=False, height=False)
        PREVIEW.geometry(f"{PREVIEW_DEFAULT_WIDTH}x{PREVIEW_DEFAULT_HEIGHT}")
    PREVIEW.deiconify()

    # Queues for decoupling capture from processing and processing from display.
    # Increased maxsize for smoother playback on macOS while keeping latency low
    capture_queue = queue.Queue(maxsize=3)
    processed_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()

    # Shared state for the detection pipeline.
    # latest_frame_holder[0] is the most recent raw frame for the detection
    # thread; detection_result holds the last detected faces for the
    # processing thread to read.  Both are guarded by detection_lock.
    detection_lock = threading.Lock()
    latest_frame_holder = [None]
    detection_result = {"target_face": None, "many_faces": None}

    # Start capture thread
    cap_thread = threading.Thread(
        target=_capture_thread_func,
        args=(cap, capture_queue, stop_event),
        daemon=True,
    )
    cap_thread.start()

    # Start detection thread — runs face detection asynchronously so the
    # processing/swap thread never blocks on it
    det_thread = threading.Thread(
        target=_detection_thread_func,
        args=(latest_frame_holder, detection_result, detection_lock, stop_event),
        daemon=True,
    )
    det_thread.start()

    # Start processing thread
    proc_thread = threading.Thread(
        target=_processing_thread_func,
        args=(
            capture_queue,
            processed_queue,
            stop_event,
            latest_frame_holder,
            detection_result,
            detection_lock,
            initial_source_face,
        ),
        daemon=True,
    )
    proc_thread.start()

    # Cleanup helper called from the display loop when preview closes
    def _cleanup():
        global preview_tk_image
        stop_event.set()
        cap_thread.join(timeout=2.0)
        det_thread.join(timeout=2.0)
        proc_thread.join(timeout=2.0)
        cap.release()
        preview_tk_image = None
        modules.globals.live_mode = False
        PREVIEW.withdraw()

    # Non-blocking display loop using ROOT.after() — avoids blocking the
    # Tk event loop which could cause UI freezes or re-entrancy issues
    def _display_next_frame():
        global preview_tk_image
        if stop_event.is_set() or PREVIEW.state() == "withdrawn":
            _cleanup()
            return

        try:
            temp_frame = processed_queue.get_nowait()
        except queue.Empty:
            ROOT.after(DISPLAY_LOOP_INTERVAL_MS, _display_next_frame)
            return

        if modules.globals.live_resizable:
            target_width = PREVIEW.winfo_width()
            target_height = PREVIEW.winfo_height()
        else:
            target_width = PREVIEW_DEFAULT_WIDTH
            target_height = PREVIEW_DEFAULT_HEIGHT

        temp_frame = fit_image_to_canvas(temp_frame, target_width, target_height)
        if temp_frame is None:
            ROOT.after(DISPLAY_LOOP_INTERVAL_MS, _display_next_frame)
            return

        # Optimized color conversion - avoid redundant operations
        try:
            # Frame is already in BGR format from processing, convert directly to RGB
            image = gpu_cvt_color(temp_frame, cv2.COLOR_BGR2RGB)
        except Exception:
            image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        preview_tk_image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=preview_tk_image)
        preview_label.image = preview_tk_image

        ROOT.after(DISPLAY_LOOP_INTERVAL_MS, _display_next_frame)

    # Kick off the non-blocking display loop
    ROOT.after(0, _display_next_frame)


def create_source_target_popup_for_webcam(
    root: ctk.CTk, map: list, camera_index: int
) -> None:
    global POPUP_LIVE, popup_status_label_live

    POPUP_LIVE = ctk.CTkToplevel(root)
    POPUP_LIVE.title(_("Source x Target Mapper"))
    POPUP_LIVE.geometry(f"{POPUP_LIVE_WIDTH}x{POPUP_LIVE_HEIGHT}")
    POPUP_LIVE.focus()

    def on_submit_click():
        if has_valid_map():
            simplify_maps()
            update_pop_live_status("Mappings successfully submitted!")
            create_webcam_preview(camera_index)  # Open the preview window
        else:
            update_pop_live_status("At least 1 source with target is required!")

    def on_add_click():
        add_blank_map()
        refresh_data(map)
        update_pop_live_status("Please provide mapping!")

    def on_clear_click():
        clear_source_target_images(map)
        refresh_data(map)
        update_pop_live_status("All mappings cleared!")

    popup_status_label_live = ctk.CTkLabel(POPUP_LIVE, text=None, justify="center")
    popup_status_label_live.grid(row=1, column=0, pady=15)

    add_button = ctk.CTkButton(
        POPUP_LIVE, text=_("Add"), command=lambda: on_add_click()
    )
    add_button.place(relx=0.1, rely=0.92, relwidth=0.2, relheight=0.05)

    clear_button = ctk.CTkButton(
        POPUP_LIVE, text=_("Clear"), command=lambda: on_clear_click()
    )
    clear_button.place(relx=0.4, rely=0.92, relwidth=0.2, relheight=0.05)

    close_button = ctk.CTkButton(
        POPUP_LIVE, text=_("Submit"), command=lambda: on_submit_click()
    )
    close_button.place(relx=0.7, rely=0.92, relwidth=0.2, relheight=0.05)


def clear_source_target_images(map: list):
    global source_label_dict_live, target_label_dict_live

    for item in map:
        if "source" in item:
            del item["source"]
        if "target" in item:
            del item["target"]

    for button_num in list(source_label_dict_live.keys()):
        source_label_dict_live[button_num].destroy()
        del source_label_dict_live[button_num]

    for button_num in list(target_label_dict_live.keys()):
        target_label_dict_live[button_num].destroy()
        del target_label_dict_live[button_num]


def refresh_data(map: list):
    global POPUP_LIVE

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP_LIVE, width=POPUP_LIVE_SCROLL_WIDTH, height=POPUP_LIVE_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

    def on_sbutton_click(map, button_num):
        map = update_webcam_source(scrollable_frame, map, button_num)

    def on_tbutton_click(map, button_num):
        map = update_webcam_target(scrollable_frame, map, button_num)

    for item in map:
        id = item["id"]

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select source image"),
            command=lambda id=id: on_sbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=0, padx=30, pady=10)

        x_label = ctk.CTkLabel(
            scrollable_frame,
            text=f"X",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=id, column=2, padx=10, pady=10)

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select target image"),
            command=lambda id=id: on_tbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=3, padx=20, pady=10)

        if "source" in item:
            image = Image.fromarray(
                gpu_cvt_color(item["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=id, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)

        if "target" in item:
            image = Image.fromarray(
                gpu_cvt_color(item["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=id, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)


def update_webcam_source(
    scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global source_label_dict_live

    source_path = ctk.filedialog.askopenfilename(
        title=_("select an source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "source" in map[button_num]:
        map[button_num].pop("source")
        source_label_dict_live[button_num].destroy()
        del source_label_dict_live[button_num]

    if source_path == "":
        return map
    else:
        cv2_img = cv2.imread(source_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["source"] = {
                "cv2": cv2_img[int(y_min) : int(y_max), int(x_min) : int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                gpu_cvt_color(map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=button_num, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)
            source_label_dict_live[button_num] = source_image
        else:
            update_pop_live_status("Face could not be detected in last upload!")
        return map


def update_webcam_target(
    scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global target_label_dict_live

    target_path = ctk.filedialog.askopenfilename(
        title=_("select an target image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "target" in map[button_num]:
        map[button_num].pop("target")
        target_label_dict_live[button_num].destroy()
        del target_label_dict_live[button_num]

    if target_path == "":
        return map
    else:
        cv2_img = cv2.imread(target_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["target"] = {
                "cv2": cv2_img[int(y_min) : int(y_max), int(x_min) : int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                gpu_cvt_color(map[button_num]["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=button_num, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)
            target_label_dict_live[button_num] = target_image
        else:
            update_pop_live_status("Face could not be detected in last upload!")
        return map
