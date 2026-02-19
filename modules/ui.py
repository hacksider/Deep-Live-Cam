import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from cv2_enumerate_cameras import enumerate_cameras
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

# --- Tk 9.0 compatibility patch ---
# In Tk 9.0, Menu.index("end") returns "" instead of raising TclError
# when the menu is empty. CustomTkinter's CTkOptionMenu doesn't handle
# this, causing crashes. This patch adds the missing guard.
try:
    from customtkinter.windows.widgets.core_widget_classes import DropdownMenu as _DropdownMenu

    _original_add_menu_commands = _DropdownMenu._add_menu_commands

    def _patched_add_menu_commands(self, *args, **kwargs):
        try:
            end_index = self._menu.index("end")
            if end_index == "" or end_index is None:
                return
        except Exception:
            pass
        _original_add_menu_commands(self, *args, **kwargs)

    _DropdownMenu._add_menu_commands = _patched_add_menu_commands
except (ImportError, AttributeError):
    pass  # CustomTkinter version doesn't have this class path
# --- End Tk 9.0 patch ---

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
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
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
        modules.globals.mouth_mask = switch_states.get("mouth_mask", False)
        modules.globals.show_mouth_mask_box = switch_states.get(
            "show_mouth_mask_box", False
        )
    except FileNotFoundError:
        pass


def _setup_window(destroy: Callable) -> ctk.CTk:
    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(
        f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}"
    )
    root.configure()
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())

    return root


def _add_image_labels(root: ctk.CTk) -> None:
    global source_label, target_label

    source_label = ctk.CTkLabel(root, text=None)
    source_label.place(relx=0.1, rely=0.05, relwidth=0.275, relheight=0.225)

    target_label = ctk.CTkLabel(root, text=None)
    target_label.place(relx=0.6, rely=0.05, relwidth=0.275, relheight=0.225)


def _add_file_buttons(root: ctk.CTk) -> None:
    select_face_button = ctk.CTkButton(
        root, text=_("Select a face"), cursor="hand2", command=lambda: select_source_path()
    )
    select_face_button.place(relx=0.1, rely=0.30, relwidth=0.3, relheight=0.1)

    swap_faces_button = ctk.CTkButton(
        root, text="\u2194", cursor="hand2", command=lambda: swap_faces_paths()
    )
    swap_faces_button.place(relx=0.45, rely=0.30, relwidth=0.1, relheight=0.1)

    select_target_button = ctk.CTkButton(
        root,
        text=_("Select a target"),
        cursor="hand2",
        command=lambda: select_target_path(),
    )
    select_target_button.place(relx=0.6, rely=0.30, relwidth=0.3, relheight=0.1)


def _add_toggle_switches(root: ctk.CTk) -> None:
    keep_fps_value = ctk.BooleanVar(value=modules.globals.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(
        root,
        text=_("Keep fps"),
        variable=keep_fps_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "keep_fps", keep_fps_value.get()),
            save_switch_states(),
        ),
    )
    keep_fps_checkbox.place(relx=0.1, rely=0.5)

    keep_frames_value = ctk.BooleanVar(value=modules.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(
        root,
        text=_("Keep frames"),
        variable=keep_frames_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "keep_frames", keep_frames_value.get()),
            save_switch_states(),
        ),
    )
    keep_frames_switch.place(relx=0.1, rely=0.55)

    enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui["face_enhancer"])
    enhancer_switch = ctk.CTkSwitch(
        root,
        text=_("Face Enhancer"),
        variable=enhancer_value,
        cursor="hand2",
        command=lambda: (
            update_tumbler("face_enhancer", enhancer_value.get()),
            save_switch_states(),
        ),
    )
    enhancer_switch.place(relx=0.1, rely=0.6)

    gpen256_value = ctk.BooleanVar(value=modules.globals.fp_ui.get("face_enhancer_gpen256", False))
    gpen256_switch = ctk.CTkSwitch(
        root,
        text=_("GPEN Enhancer 256"),
        variable=gpen256_value,
        cursor="hand2",
        command=lambda: (
            update_tumbler("face_enhancer_gpen256", gpen256_value.get()),
            save_switch_states(),
        ),
    )
    gpen256_switch.place(relx=0.1, rely=0.65)

    gpen512_value = ctk.BooleanVar(value=modules.globals.fp_ui.get("face_enhancer_gpen512", False))
    gpen512_switch = ctk.CTkSwitch(
        root,
        text=_("GPEN Enhancer 512"),
        variable=gpen512_value,
        cursor="hand2",
        command=lambda: (
            update_tumbler("face_enhancer_gpen512", gpen512_value.get()),
            save_switch_states(),
        ),
    )
    gpen512_switch.place(relx=0.1, rely=0.7)

    keep_audio_value = ctk.BooleanVar(value=modules.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(
        root,
        text=_("Keep audio"),
        variable=keep_audio_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "keep_audio", keep_audio_value.get()),
            save_switch_states(),
        ),
    )
    keep_audio_switch.place(relx=0.6, rely=0.5)

    many_faces_value = ctk.BooleanVar(value=modules.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(
        root,
        text=_("Many faces"),
        variable=many_faces_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "many_faces", many_faces_value.get()),
            save_switch_states(),
        ),
    )
    many_faces_switch.place(relx=0.6, rely=0.55)

    color_correction_value = ctk.BooleanVar(value=modules.globals.color_correction)
    color_correction_switch = ctk.CTkSwitch(
        root,
        text=_("Fix Blueish Cam"),
        variable=color_correction_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "color_correction", color_correction_value.get()),
            save_switch_states(),
        ),
    )
    color_correction_switch.place(relx=0.6, rely=0.6)

    map_faces = ctk.BooleanVar(value=modules.globals.map_faces)
    map_faces_switch = ctk.CTkSwitch(
        root,
        text=_("Map faces"),
        variable=map_faces,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "map_faces", map_faces.get()),
            save_switch_states(),
            close_mapper_window() if not map_faces.get() else None
        ),
    )
    map_faces_switch.place(relx=0.1, rely=0.75)

    poisson_blend_value = ctk.BooleanVar(value=modules.globals.poisson_blend)
    poisson_blend_switch = ctk.CTkSwitch(
        root,
        text=_("Poisson Blend"),
        variable=poisson_blend_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "poisson_blend", poisson_blend_value.get()),
            save_switch_states(),
        ),
    )
    poisson_blend_switch.place(relx=0.1, rely=0.8)

    show_fps_value = ctk.BooleanVar(value=modules.globals.show_fps)
    show_fps_switch = ctk.CTkSwitch(
        root,
        text=_("Show FPS"),
        variable=show_fps_value,
        cursor="hand2",
        command=lambda: (
            setattr(modules.globals, "show_fps", show_fps_value.get()),
            save_switch_states(),
        ),
    )
    show_fps_switch.place(relx=0.6, rely=0.65)

    mouth_mask_var = ctk.BooleanVar(value=modules.globals.mouth_mask)
    mouth_mask_switch = ctk.CTkSwitch(
        root,
        text=_("Mouth Mask"),
        variable=mouth_mask_var,
        cursor="hand2",
        command=lambda: setattr(modules.globals, "mouth_mask", mouth_mask_var.get()),
    )
    mouth_mask_switch.place(relx=0.1, rely=0.45)

    show_mouth_mask_box_var = ctk.BooleanVar(value=modules.globals.show_mouth_mask_box)
    show_mouth_mask_box_switch = ctk.CTkSwitch(
        root,
        text=_("Show Mouth Mask Box"),
        variable=show_mouth_mask_box_var,
        cursor="hand2",
        command=lambda: setattr(
            modules.globals, "show_mouth_mask_box", show_mouth_mask_box_var.get()
        ),
    )
    show_mouth_mask_box_switch.place(relx=0.6, rely=0.45)


def _add_action_buttons(root: ctk.CTk, start: Callable, destroy: Callable) -> None:
    start_button = ctk.CTkButton(
        root, text=_("Start"), cursor="hand2", command=lambda: analyze_target(start, root)
    )
    start_button.place(relx=0.15, rely=0.86, relwidth=0.2, relheight=0.05)

    stop_button = ctk.CTkButton(
        root, text=_("Destroy"), cursor="hand2", command=lambda: destroy()
    )
    stop_button.place(relx=0.4, rely=0.86, relwidth=0.2, relheight=0.05)

    preview_button = ctk.CTkButton(
        root, text=_("Preview"), cursor="hand2", command=lambda: toggle_preview()
    )
    preview_button.place(relx=0.65, rely=0.86, relwidth=0.2, relheight=0.05)


def _add_camera_row(root: ctk.CTk) -> None:
    camera_label = ctk.CTkLabel(root, text=_("Select Camera:"))
    camera_label.place(relx=0.1, rely=0.92, relwidth=0.2, relheight=0.05)

    available_cameras = get_available_cameras()
    camera_indices, camera_names = available_cameras

    if not camera_names or camera_names[0] == "No cameras found":
        camera_variable = ctk.StringVar(value="No cameras found")
        camera_optionmenu = ctk.CTkOptionMenu(
            root,
            variable=camera_variable,
            values=["No cameras found"],
            state="disabled",
        )
    else:
        camera_variable = ctk.StringVar(value=camera_names[0])
        camera_optionmenu = ctk.CTkOptionMenu(
            root, variable=camera_variable, values=camera_names
        )

    camera_optionmenu.place(relx=0.35, rely=0.92, relwidth=0.25, relheight=0.05)

    live_button = ctk.CTkButton(
        root,
        text=_("Live"),
        cursor="hand2",
        command=lambda: webcam_preview(
            root,
            (
                camera_indices[camera_names.index(camera_variable.get())]
                if camera_names and camera_names[0] != "No cameras found"
                else None
            ),
        ),
        state=(
            "normal"
            if camera_names and camera_names[0] != "No cameras found"
            else "disabled"
        ),
    )
    live_button.place(relx=0.65, rely=0.92, relwidth=0.2, relheight=0.05)


def _add_sliders(root: ctk.CTk) -> None:
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

    transparency_label = ctk.CTkLabel(root, text="Transparency:")
    transparency_label.place(relx=0.15, rely=0.75, relwidth=0.2, relheight=0.05)

    transparency_slider = ctk.CTkSlider(
        root,
        from_=0.0,
        to=1.0,
        variable=transparency_var,
        command=on_transparency_change,
        fg_color="#E0E0E0",
        progress_color="#007BFF",
        button_color="#FFFFFF",
        button_hover_color="#CCCCCC",
        height=5,
        border_width=1,
        corner_radius=3,
    )
    transparency_slider.place(relx=0.35, rely=0.77, relwidth=0.5, relheight=0.02)

    sharpness_var = ctk.DoubleVar(value=0.0)

    def on_sharpness_change(value: float):
        modules.globals.sharpness = float(value)
        update_status(f"Sharpness set to {value:.1f}")

    sharpness_label = ctk.CTkLabel(root, text="Sharpness:")
    sharpness_label.place(relx=0.15, rely=0.80, relwidth=0.2, relheight=0.05)

    sharpness_slider = ctk.CTkSlider(
        root,
        from_=0.0,
        to=5.0,
        variable=sharpness_var,
        command=on_sharpness_change,
        fg_color="#E0E0E0",
        progress_color="#007BFF",
        button_color="#FFFFFF",
        button_hover_color="#CCCCCC",
        height=5,
        border_width=1,
        corner_radius=3,
    )
    sharpness_slider.place(relx=0.35, rely=0.82, relwidth=0.5, relheight=0.02)


def _add_status_bar(root: ctk.CTk) -> None:
    global status_label

    status_label = ctk.CTkLabel(root, text=None, justify="center")
    status_label.place(relx=0.1, rely=0.96, relwidth=0.8)

    donate_label = ctk.CTkLabel(
        root, text="Deep Live Cam", justify="center", cursor="hand2"
    )
    donate_label.place(relx=0.1, rely=0.98, relwidth=0.8)
    donate_label.configure(
        text_color=ctk.ThemeManager.theme.get("URL").get("text_color")
    )
    donate_label.bind(
        "<Button>", lambda event: webbrowser.open("https://deeplivecam.net")
    )


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    load_switch_states()
    root = _setup_window(destroy)
    _add_image_labels(root)
    _add_file_buttons(root)
    _add_toggle_switches(root)
    _add_action_buttons(root, start, destroy)
    _add_camera_row(root)
    _add_sliders(root)
    _add_status_bar(root)
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
    status_label.configure(text=_(text))
    ROOT.update()


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
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
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
    """Returns a list of available camera names and indices."""
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()

            camera_indices = list(range(len(devices)))
            camera_names = devices

            if not camera_names:
                test_indices = [-1, 0]
                working_cameras = []

                for idx in test_indices:
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        working_cameras.append(f"Camera {idx}")
                        cap.release()

                if working_cameras:
                    return test_indices[: len(working_cameras)], working_cameras

            if not camera_names:
                return [], ["No cameras found"]

            return camera_indices, camera_names

        except Exception as e:
            print(f"Error detecting cameras: {str(e)}")
            return [], ["No cameras found"]
    else:
        camera_indices = []
        camera_names = []

        if platform.system() == "Darwin":
            try:
                for cam in enumerate_cameras(cv2.CAP_AVFOUNDATION):
                    camera_indices.append(cam.index)
                    camera_names.append(cam.name if cam.name else f"Camera {cam.index}")
            except Exception:
                for i in range(2):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        camera_indices.append(i)
                        camera_names.append(f"Camera {i}")
                        cap.release()
        else:
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_indices.append(i)
                    camera_names.append(f"Camera {i}")
                    cap.release()

        if not camera_names:
            return [], ["No cameras found"]

        return camera_indices, camera_names

