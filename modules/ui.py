import os
import queue
import threading
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from modules.gpu_processing import gpu_cvt_color, gpu_resize, gpu_flip
from PIL import Image, ImageOps
import json
import modules.globals
import modules.metadata
from modules.face_analyser import get_one_face
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    is_image,
    is_video,
    resolve_relative_path,
    has_image_extension,
)
from modules.gettext import LanguageManager
import platform

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

# Monkey-patch CustomTkinter DropdownMenu for Tk 9.0 compatibility.
# Tk 9.0 returns "" from Menu.index("end") on an empty menu, causing TclError
# in DropdownMenu._add_menu_commands when it calls self.delete(0, "end").
import tkinter as _tk

if _tk.TkVersion >= 9.0:
    from customtkinter.windows.widgets.core_widget_classes.dropdown_menu import (
        DropdownMenu as _DropdownMenu,
    )

    _orig_add_menu_commands = _DropdownMenu._add_menu_commands

    def _patched_add_menu_commands(self):
        try:
            _orig_add_menu_commands(self)
        except _tk.TclError:
            # Empty menu — just add commands without deleting first
            import sys

            if sys.platform.startswith("linux"):
                for value in self._values:
                    self.add_command(
                        label="  " + value.ljust(self._min_character_width) + "  ",
                        command=lambda v=value: self._button_callback(v),
                        compound="left",
                    )
            else:
                for value in self._values:
                    self.add_command(
                        label=value.ljust(self._min_character_width),
                        command=lambda v=value: self._button_callback(v),
                        compound="left",
                    )

    _DropdownMenu._add_menu_commands = _patched_add_menu_commands

# Re-export moved functions for backward compatibility
from modules.ui_analysis import analyze_target, check_and_ignore_nsfw  # noqa: F401
from modules.ui_webcam import (  # noqa: F401
    webcam_preview,
    create_webcam_preview,
    _capture_thread_func,
    _processing_thread_func,
    DETECT_EVERY_N,
)
from modules.ui_mapper import (  # noqa: F401
    create_source_target_popup,
    create_source_target_popup_for_webcam,
    update_webcam_source,
    update_webcam_target,
    update_popup_source,
    clear_source_target_images,
    refresh_data,
    close_mapper_window,
    update_pop_status,
    update_pop_live_status,
    POPUP,
    POPUP_LIVE,
    source_label_dict,
    source_label_dict_live,
    target_label_dict_live,
    popup_status_label,
    popup_status_label_live,
    POPUP_WIDTH,
    POPUP_HEIGHT,
    POPUP_SCROLL_WIDTH,
    POPUP_SCROLL_HEIGHT,
    POPUP_LIVE_WIDTH,
    POPUP_LIVE_HEIGHT,
    POPUP_LIVE_SCROLL_WIDTH,
    POPUP_LIVE_SCROLL_HEIGHT,
    MAPPER_PREVIEW_MAX_HEIGHT,
    MAPPER_PREVIEW_MAX_WIDTH,
    DEFAULT_BUTTON_WIDTH,
    DEFAULT_BUTTON_HEIGHT,
)


ROOT = None
ROOT_HEIGHT = 800
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 960
PREVIEW_DEFAULT_HEIGHT = 540

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

_ = None
preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None

img_ft, vid_ft = modules.globals.file_types

# Debounce timer for responsive image scaling
_resize_timer_id = None

# Selected camera index, updated by camera detection and dropdown selection
_selected_camera_index = 0


def init(start: Callable[[], None], destroy: Callable[[], None], lang: str) -> ctk.CTk:
    global ROOT, PREVIEW, _

    lang_manager = LanguageManager(lang)
    _ = lang_manager._
    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def _state_file_path() -> str:
    import platform as _platform
    if _platform.system() == "Windows":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.path.join(os.path.expanduser("~"), ".config")
    config_dir = os.path.join(base, "deep-live-cam")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "switch_states.json")


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
        "virtual_cam": modules.globals.virtual_cam,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "source_path": modules.globals.source_path,
        "target_path": modules.globals.target_path,
    }
    with open(_state_file_path(), "w") as f:
        json.dump(switch_states, f)


def load_switch_states():
    try:
        with open(_state_file_path(), "r") as f:
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
        modules.globals.fp_ui = switch_states.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = switch_states.get("show_fps", False)
        modules.globals.virtual_cam = switch_states.get("virtual_cam", False)
        modules.globals.mouth_mask = switch_states.get("mouth_mask", False)
        modules.globals.show_mouth_mask_box = switch_states.get(
            "show_mouth_mask_box", False
        )
        # Restore last-used paths; validate existence before accepting.
        saved_source = switch_states.get("source_path")
        if saved_source and os.path.isfile(saved_source):
            modules.globals.source_path = saved_source
        saved_target = switch_states.get("target_path")
        if saved_target and os.path.isfile(saved_target):
            modules.globals.target_path = saved_target
    except FileNotFoundError:
        pass


def _restore_recent_paths() -> None:
    """Populate image labels from saved paths after labels have been created."""
    global RECENT_DIRECTORY_SOURCE, RECENT_DIRECTORY_TARGET
    if modules.globals.source_path and is_image(modules.globals.source_path):
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
    if modules.globals.target_path:
        if is_image(modules.globals.target_path):
            RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
            image = render_image_preview(modules.globals.target_path, (200, 200))
            target_label.configure(image=image)
        elif is_video(modules.globals.target_path):
            RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
            frame = render_video_preview(modules.globals.target_path, (200, 200))
            target_label.configure(image=frame)


def _setup_window(destroy: Callable) -> ctk.CTk:
    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = ctk.CTk()
    root.minsize(700, 600)
    root.title(
        f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
    )
    root.configure()
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())

    # Configure root grid: row 1 (tabview) gets all extra space
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=0)  # top_frame (images)
    root.rowconfigure(1, weight=1)  # settings_tabview
    root.rowconfigure(2, weight=0)  # action_frame
    root.rowconfigure(3, weight=0)  # status_frame

    return root


def _add_top_frame(root: ctk.CTk) -> None:
    global source_label, target_label

    top_frame = ctk.CTkFrame(root)
    top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
    top_frame.columnconfigure(0, weight=1)
    top_frame.columnconfigure(1, weight=0)
    top_frame.columnconfigure(2, weight=1)

    # Source column
    source_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
    source_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    source_frame.columnconfigure(0, weight=1)

    source_label = ctk.CTkLabel(source_frame, text=None, width=200, height=200)
    source_label.grid(row=0, column=0, pady=(5, 5))

    select_face_button = ctk.CTkButton(
        source_frame, text=_("Select a face"), cursor="hand2",
        command=lambda: select_source_path(),
    )
    select_face_button.grid(row=1, column=0, pady=(0, 5), sticky="ew", padx=10)

    # Swap button
    swap_faces_button = ctk.CTkButton(
        top_frame, text="\u2194", cursor="hand2", width=40,
        command=lambda: swap_faces_paths(),
    )
    swap_faces_button.grid(row=0, column=1, padx=5)

    # Target column
    target_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
    target_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
    target_frame.columnconfigure(0, weight=1)

    target_label = ctk.CTkLabel(target_frame, text=None, width=200, height=200)
    target_label.grid(row=0, column=0, pady=(5, 5))

    select_target_button = ctk.CTkButton(
        target_frame, text=_("Select a target"), cursor="hand2",
        command=lambda: select_target_path(),
    )
    select_target_button.grid(row=1, column=0, pady=(0, 2), sticky="ew", padx=10)

    capture_target_button = ctk.CTkButton(
        target_frame, text=_("Capture from camera"), cursor="hand2",
        command=lambda: capture_target_from_camera(),
    )
    capture_target_button.grid(row=2, column=0, pady=(0, 5), sticky="ew", padx=10)


# --- Data-driven switch definitions ---

def _get_switch_defs():
    """Return switch definitions grouped by tab.

    Each definition: (label, global_attr_or_tumbler, default_value)
    - If global_attr_or_tumbler starts with "fp_ui:", it's a tumbler key
    - Otherwise it's a modules.globals attribute name
    """
    return {
        "Processing": [
            (_("Mouth Mask"), "mouth_mask", False),
            (_("Show Mouth Mask Box"), "show_mouth_mask_box", False),
            (_("Many faces"), "many_faces", False),
            (_("Map faces"), "map_faces", False),
            (_("Poisson Blend"), "poisson_blend", False),
        ],
        "Enhancement": [
            (_("Face Enhancer"), "fp_ui:face_enhancer", False),
            (_("GPEN-256"), "fp_ui:face_enhancer_gpen256", False),
            (_("GPEN-512"), "fp_ui:face_enhancer_gpen512", False),
        ],
        "Output": [
            (_("Keep fps"), "keep_fps", True),
            (_("Keep audio"), "keep_audio", True),
            (_("Keep frames"), "keep_frames", False),
        ],
        "Live Mode": [
            (_("Fix Blueish Cam"), "color_correction", False),
            (_("Show FPS"), "show_fps", False),
            (_("Virtual Camera"), "virtual_cam", False),
        ],
    }


def _get_switch_value(attr: str) -> bool:
    if attr.startswith("fp_ui:"):
        key = attr[len("fp_ui:"):]
        return modules.globals.fp_ui.get(key, False)
    return getattr(modules.globals, attr, False)


def _create_switch(parent: ctk.CTkFrame, label: str, attr: str) -> ctk.CTkSwitch:
    value_var = ctk.BooleanVar(value=_get_switch_value(attr))

    if attr.startswith("fp_ui:"):
        key = attr[len("fp_ui:"):]
        command = lambda: (
            update_tumbler(key, value_var.get()),
            save_switch_states(),
        )
    elif attr == "map_faces":
        command = lambda: (
            setattr(modules.globals, attr, value_var.get()),
            save_switch_states(),
            close_mapper_window() if not value_var.get() else None,
        )
    else:
        command = lambda: (
            setattr(modules.globals, attr, value_var.get()),
            save_switch_states(),
        )

    switch = ctk.CTkSwitch(
        parent, text=label, variable=value_var, cursor="hand2", command=command,
    )
    return switch


def _add_settings_tabview(root: ctk.CTk, live_button: ctk.CTkButton) -> None:
    tabview = ctk.CTkTabview(root)
    tabview.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

    switch_defs = _get_switch_defs()

    for tab_name, switches in switch_defs.items():
        tab = tabview.add(tab_name)
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)

        for i, (label, attr, _default) in enumerate(switches):
            row = i // 2
            col = i % 2
            sw = _create_switch(tab, label, attr)
            sw.grid(row=row, column=col, sticky="w", padx=15, pady=8)

    # Enhancement tab: add sliders
    enhancement_tab = tabview.tab("Enhancement")
    _add_sliders_to_tab(enhancement_tab, len(switch_defs["Enhancement"]))

    # Live Mode tab: add camera dropdown (live button is in action bar)
    live_tab = tabview.tab("Live Mode")
    _add_camera_to_tab(live_tab, root, len(switch_defs["Live Mode"]), live_button)


def _add_sliders_to_tab(tab: ctk.CTkFrame, num_switches: int) -> None:
    start_row = (num_switches + 1) // 2 + 1

    transparency_var = ctk.DoubleVar(value=1.0)

    def on_transparency_change(value: float):
        val = float(value)
        modules.globals.opacity = val
        percentage = int(val * 100)

        if percentage == 0:
            modules.globals.fp_ui["face_enhancer"] = False
            update_status("Transparency set to 0% - Face swapping disabled.")
        elif percentage == 100:
            modules.globals.face_swapper_enabled = True
            update_status("Transparency set to 100%.")
        else:
            modules.globals.face_swapper_enabled = True
            update_status(f"Transparency set to {percentage}%")

    transparency_label = ctk.CTkLabel(tab, text="Transparency:")
    transparency_label.grid(row=start_row, column=0, sticky="w", padx=15, pady=(15, 2))

    transparency_slider = ctk.CTkSlider(
        tab, from_=0.0, to=1.0, variable=transparency_var,
        command=on_transparency_change,
        fg_color="#E0E0E0", progress_color="#007BFF",
        button_color="#FFFFFF", button_hover_color="#CCCCCC",
        height=5, border_width=1, corner_radius=3,
    )
    transparency_slider.grid(
        row=start_row, column=1, sticky="ew", padx=(0, 15), pady=(15, 2),
    )

    sharpness_var = ctk.DoubleVar(value=0.0)

    def on_sharpness_change(value: float):
        modules.globals.sharpness = float(value)
        update_status(f"Sharpness set to {value:.1f}")

    sharpness_label = ctk.CTkLabel(tab, text="Sharpness:")
    sharpness_label.grid(row=start_row + 1, column=0, sticky="w", padx=15, pady=2)

    sharpness_slider = ctk.CTkSlider(
        tab, from_=0.0, to=5.0, variable=sharpness_var,
        command=on_sharpness_change,
        fg_color="#E0E0E0", progress_color="#007BFF",
        button_color="#FFFFFF", button_hover_color="#CCCCCC",
        height=5, border_width=1, corner_radius=3,
    )
    sharpness_slider.grid(
        row=start_row + 1, column=1, sticky="ew", padx=(0, 15), pady=2,
    )


def _add_camera_to_tab(
    tab: ctk.CTkFrame, root: ctk.CTk, num_switches: int, live_button: ctk.CTkButton,
) -> None:
    start_row = (num_switches + 1) // 2 + 1

    camera_label = ctk.CTkLabel(tab, text=_("Select Camera:"))
    camera_label.grid(row=start_row, column=0, sticky="w", padx=15, pady=(15, 5))

    camera_variable = ctk.StringVar(value=_("Detecting cameras..."))
    camera_optionmenu = ctk.CTkOptionMenu(
        tab, variable=camera_variable,
        values=[_("Detecting cameras...")], state="disabled",
    )
    camera_optionmenu.grid(
        row=start_row, column=1, sticky="ew", padx=(0, 15), pady=(15, 5),
    )

    camera_indices: list = []
    camera_names: list = []

    def _on_camera_selected(choice):
        global _selected_camera_index
        if camera_names and choice in camera_names:
            _selected_camera_index = camera_indices[camera_names.index(choice)]

    # Wire up the live button command to use the camera selection from this tab
    live_button.configure(
        command=lambda: webcam_preview(
            root,
            (
                camera_indices[camera_names.index(camera_variable.get())]
                if camera_names and camera_names[0] != "No cameras found"
                else None
            ),
        ),
    )
    camera_optionmenu.configure(command=_on_camera_selected)

    def _finish_camera_probe(indices, names):
        global _selected_camera_index
        camera_indices.clear()
        camera_indices.extend(indices)
        camera_names.clear()
        camera_names.extend(names)
        if names and names[0] != "No cameras found":
            camera_variable.set(names[0])
            _selected_camera_index = indices[0]
            camera_optionmenu.configure(values=names, state="normal")
            live_button.configure(state="normal")
        else:
            camera_variable.set(_("No cameras found"))
            camera_optionmenu.configure(values=[_("No cameras found")], state="disabled")

    _camera_queue: queue.Queue = queue.Queue()

    def _poll_camera_queue():
        try:
            indices, names = _camera_queue.get_nowait()
            _finish_camera_probe(indices, names)
        except queue.Empty:
            root.after(100, _poll_camera_queue)

    if platform.system() == "Darwin":
        def _enumerate_cameras():
            _camera_queue.put(([0, 1], ["Camera 0", "Camera 1"]))
    else:
        def _enumerate_cameras():
            indices, names = get_available_cameras()
            _camera_queue.put((indices, names))

    threading.Thread(target=_enumerate_cameras, daemon=True).start()
    root.after(100, _poll_camera_queue)


def _add_action_buttons(root: ctk.CTk, start: Callable, destroy: Callable) -> ctk.CTkButton:
    """Create action bar with Start, Stop, Preview, Live buttons. Returns the Live button."""
    action_frame = ctk.CTkFrame(root, fg_color="transparent")
    action_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
    action_frame.columnconfigure(0, weight=1)
    action_frame.columnconfigure(1, weight=1)
    action_frame.columnconfigure(2, weight=1)
    action_frame.columnconfigure(3, weight=1)

    start_button = ctk.CTkButton(
        action_frame, text=_("Start"), cursor="hand2",
        command=lambda: analyze_target(start, root),
    )
    start_button.grid(row=0, column=0, sticky="ew", padx=5)

    stop_button = ctk.CTkButton(
        action_frame, text=_("Destroy"), cursor="hand2",
        command=lambda: destroy(),
    )
    stop_button.grid(row=0, column=1, sticky="ew", padx=5)

    preview_button = ctk.CTkButton(
        action_frame, text=_("Preview"), cursor="hand2",
        command=lambda: toggle_preview(),
    )
    preview_button.grid(row=0, column=2, sticky="ew", padx=5)

    # Live button — command and state wired up by _add_camera_to_tab after detection
    live_button = ctk.CTkButton(
        action_frame, text=_("Live"), cursor="hand2", state="disabled",
    )
    live_button.grid(row=0, column=3, sticky="ew", padx=5)

    return live_button


def _add_status_bar(root: ctk.CTk) -> None:
    global status_label

    status_frame = ctk.CTkFrame(root, fg_color="transparent")
    status_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 5))
    status_frame.columnconfigure(0, weight=1)
    status_frame.columnconfigure(1, weight=1)

    status_label = ctk.CTkLabel(status_frame, text=None, justify="left")
    status_label.grid(row=0, column=0, sticky="w")

    donate_label = ctk.CTkLabel(
        status_frame, text="Deep Live Cam", justify="right", cursor="hand2",
    )
    donate_label.grid(row=0, column=1, sticky="e")
    donate_label.configure(
        text_color=ctk.ThemeManager.theme.get("URL").get("text_color")
    )
    donate_label.bind(
        "<Button>", lambda event: webbrowser.open("https://deeplivecam.net")
    )


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    load_switch_states()
    root = _setup_window(destroy)
    _add_top_frame(root)
    live_button = _add_action_buttons(root, start, destroy)
    _add_settings_tabview(root, live_button)
    _add_status_bar(root)
    _restore_recent_paths()
    return root


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title(_("Preview"))
    preview.configure()
    preview.protocol("WM_DELETE_WINDOW", lambda: toggle_preview())
    preview.resizable(width=True, height=True)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)

    preview_slider = ctk.CTkSlider(
        preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value)
    )

    return preview


def update_status(text: str) -> None:
    # May be called from background threads (e.g. face swapper model loading).
    # Tkinter is not thread-safe: schedule the label update on the main thread.
    ROOT.after(0, lambda t=text: status_label.configure(text=_(t)))


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    if PREVIEW.state() == "normal":
        global frame_processors
        frame_processors = get_frame_processors_modules(
            modules.globals.frame_processors
        )


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
        save_switch_states()
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)


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
    save_switch_states()


def capture_target_from_camera() -> None:
    """Open a live camera preview window with a Capture button."""
    cap = cv2.VideoCapture(_selected_camera_index)
    if not cap.isOpened():
        update_status("Failed to open camera.")
        return

    capture_window = ctk.CTkToplevel(ROOT)
    capture_window.title(_("Camera Capture"))
    capture_window.minsize(480, 400)
    capture_window.resizable(width=True, height=True)
    capture_window.protocol("WM_DELETE_WINDOW", lambda: _close_capture())

    feed_label = ctk.CTkLabel(capture_window, text=None)
    feed_label.pack(fill="both", expand=True, padx=5, pady=5)

    button_frame = ctk.CTkFrame(capture_window, fg_color="transparent")
    button_frame.pack(fill="x", padx=10, pady=(0, 10))
    button_frame.columnconfigure(0, weight=1)
    button_frame.columnconfigure(1, weight=1)

    capture_btn = ctk.CTkButton(
        button_frame, text=_("Capture"), cursor="hand2",
        command=lambda: _do_capture(),
    )
    capture_btn.grid(row=0, column=0, sticky="ew", padx=5)

    cancel_btn = ctk.CTkButton(
        button_frame, text=_("Cancel"), cursor="hand2",
        command=lambda: _close_capture(),
    )
    cancel_btn.grid(row=0, column=1, sticky="ew", padx=5)

    # Mutable state shared between callbacks
    _running = [True]
    _last_frame = [None]

    def _update_feed():
        if not _running[0]:
            return
        ret, frame = cap.read()
        if ret and frame is not None:
            _last_frame[0] = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            # Scale to fit the label while preserving aspect ratio
            w = feed_label.winfo_width() or 480
            h = feed_label.winfo_height() or 360
            pil_img = ImageOps.contain(pil_img, (max(w, 1), max(h, 1)), Image.LANCZOS)
            ctk_img = ctk.CTkImage(pil_img, size=pil_img.size)
            feed_label.configure(image=ctk_img)
            feed_label._ctk_img = ctk_img  # prevent GC
        capture_window.after(33, _update_feed)  # ~30 fps

    def _do_capture():
        frame = _last_frame[0]
        if frame is None:
            update_status("No frame captured yet.")
            return
        _close_capture()

        tmp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        capture_path = os.path.join(tmp_dir, "camera_capture.png")
        cv2.imwrite(capture_path, frame)

        global RECENT_DIRECTORY_TARGET
        modules.globals.target_path = capture_path
        RECENT_DIRECTORY_TARGET = tmp_dir
        image = render_image_preview(capture_path, (200, 200))
        target_label.configure(image=image)
        save_switch_states()
        update_status("Camera capture set as target.")

    def _close_capture():
        _running[0] = False
        cap.release()
        capture_window.destroy()

    # Start the feed after a short delay so the window has geometry
    capture_window.after(100, _update_feed)


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
        save_switch_states()
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
        save_switch_states()
    else:
        modules.globals.target_path = None
        target_label.configure(image=None)


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


def fit_image_to_size(image, width: int, height: int):
    if width is None and height is None:
        return image
    h, w, _ = image.shape
    ratio_h = 0.0
    ratio_w = 0.0
    if width > height:
        ratio_h = height / h
    else:
        ratio_w = width / w
    ratio = max(ratio_w, ratio_h)
    new_size = (int(ratio * w), int(ratio * h))
    return gpu_resize(image, dsize=new_size)


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
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        update_status("Processing succeed!")
        PREVIEW.deiconify()


def get_available_cameras():
    """Returns a list of available camera names and indices.

    On Windows, uses pygrabber FilterGraph for named device enumeration.
    On Linux, uses a bounded cv2.VideoCapture probe loop with CAP_ANY.
    On macOS, this function is not used — see _add_camera_to_tab which
    defaults to [0, 1] to avoid the OBSENSOR segfault.
    """
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            camera_indices = list(range(len(devices)))
            camera_names = devices

            if not camera_names:
                # Fallback: probe indices 0 and 1
                camera_indices = []
                camera_names = []
                for idx in range(2):
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        camera_indices.append(idx)
                        camera_names.append(f"Camera {idx}")
                        cap.release()

            if not camera_names:
                return [], ["No cameras found"]
            return camera_indices, camera_names
        except Exception as e:
            print(f"Error detecting cameras: {str(e)}")
            return [], ["No cameras found"]
    else:
        # Linux only (macOS uses defaults; see _add_camera_to_tab).
        camera_indices = []
        camera_names = []
        consecutive_failures = 0
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_indices.append(i)
                camera_names.append(f"Camera {i}")
                cap.release()
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break

        if not camera_names:
            return [], ["No cameras found"]
        return camera_indices, camera_names
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
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


# How often to run full face detection. On intermediate frames the last
# detected face positions are reused, which significantly reduces the
# per-frame cost of the processing thread.
DETECT_EVERY_N = 2


def _processing_thread_func(capture_queue, processed_queue, stop_event):
    """Processing thread: takes raw frames from capture_queue, applies face
    processing, and puts results into processed_queue. Drops processed frames
    when the output queue is full so the UI always gets the latest result.

    Uses DETECT_EVERY_N to skip expensive face detection on intermediate
    frames, reusing cached face positions instead."""
    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    source_image = None
    prev_time = time.time()
    fps_update_interval = 0.5
    frame_count = 0
    fps = 0
    proc_frame_index = 0
    cached_target_face = None  # cached single-face result
    cached_many_faces = None   # cached many-faces result

    while not stop_event.is_set():
        try:
            frame = capture_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        temp_frame = frame.copy()
        run_detection = (proc_frame_index % DETECT_EVERY_N == 0)
        proc_frame_index += 1

        if modules.globals.live_mirror:
            temp_frame = gpu_flip(temp_frame, 1)

        if not modules.globals.map_faces:
            if source_image is None and modules.globals.source_path:
                source_image = get_one_face(cv2.imread(modules.globals.source_path))

            # Update face detection cache on detection frames
            if run_detection or (cached_target_face is None and cached_many_faces is None):
                if modules.globals.many_faces:
                    cached_many_faces = get_many_faces(temp_frame)
                    cached_target_face = None
                else:
                    cached_target_face = get_one_face(temp_frame)
                    cached_many_faces = None

            for frame_processor in frame_processors:
                if frame_processor.NAME == "DLC.FACE-ENHANCER":
                    if modules.globals.fp_ui["face_enhancer"]:
                        temp_frame = frame_processor.process_frame(None, temp_frame)
                elif frame_processor.NAME == "DLC.FACE-SWAPPER":
                    # Use cached face positions to skip redundant detection
                    swapped_bboxes = []
                    if modules.globals.many_faces and cached_many_faces:
                        result = temp_frame.copy()
                        for t_face in cached_many_faces:
                            result = frame_processor.swap_face(source_image, t_face, result)
                            if hasattr(t_face, 'bbox') and t_face.bbox is not None:
                                swapped_bboxes.append(t_face.bbox.astype(int))
                        temp_frame = result
                    elif cached_target_face is not None:
                        temp_frame = frame_processor.swap_face(source_image, cached_target_face, temp_frame)
                        if hasattr(cached_target_face, 'bbox') and cached_target_face.bbox is not None:
                            swapped_bboxes.append(cached_target_face.bbox.astype(int))
                    # Apply post-processing (sharpening, interpolation)
                    temp_frame = frame_processor.apply_post_processing(temp_frame, swapped_bboxes)
                else:
                    temp_frame = frame_processor.process_frame(source_image, temp_frame)
        else:
            modules.globals.target_path = None
            for frame_processor in frame_processors:
                if frame_processor.NAME == "DLC.FACE-ENHANCER":
                    if modules.globals.fp_ui["face_enhancer"]:
                        temp_frame = frame_processor.process_frame_v2(temp_frame)
                else:
                    temp_frame = frame_processor.process_frame_v2(temp_frame)

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
    global preview_label, PREVIEW

    cap = VideoCapturer(camera_index)
    if not cap.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, 60):
        update_status("Failed to start camera")
        return

    preview_label.configure(width=PREVIEW_DEFAULT_WIDTH, height=PREVIEW_DEFAULT_HEIGHT)
    PREVIEW.deiconify()

    # Queues for decoupling capture from processing and processing from display.
    # Small maxsize ensures we always work on recent frames and drop stale ones.
    capture_queue = queue.Queue(maxsize=2)
    processed_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    # Start capture thread
    cap_thread = threading.Thread(
        target=_capture_thread_func,
        args=(cap, capture_queue, stop_event),
        daemon=True,
    )
    cap_thread.start()

    # Start processing thread
    proc_thread = threading.Thread(
        target=_processing_thread_func,
        args=(capture_queue, processed_queue, stop_event),
        daemon=True,
    )
    proc_thread.start()

    # Main (UI) thread: pull processed frames and update the display
    while not stop_event.is_set():
        try:
            temp_frame = processed_queue.get(timeout=0.03)
        except queue.Empty:
            ROOT.update()
            continue

        if modules.globals.live_resizable:
            temp_frame = fit_image_to_size(
                temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height()
            )
        else:
            temp_frame = fit_image_to_size(
                temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height()
            )

        image = gpu_cvt_color(temp_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageOps.contain(
            image, (temp_frame.shape[1], temp_frame.shape[0]), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        ROOT.update()

        if PREVIEW.state() == "withdrawn":
            break

    # Signal threads to stop and wait for them
    stop_event.set()
    cap_thread.join(timeout=2.0)
    proc_thread.join(timeout=2.0)
    cap.release()
    PREVIEW.withdraw()


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

    add_button = ctk.CTkButton(POPUP_LIVE, text=_("Add"), command=lambda: on_add_click())
    add_button.place(relx=0.1, rely=0.92, relwidth=0.2, relheight=0.05)

    clear_button = ctk.CTkButton(POPUP_LIVE, text=_("Clear"), command=lambda: on_clear_click())
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
                "cv2": cv2_img[int(y_min): int(y_max), int(x_min): int(x_max)],
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
                "cv2": cv2_img[int(y_min): int(y_max), int(x_min): int(x_max)],
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
