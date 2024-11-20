import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from cv2_enumerate_cameras import enumerate_cameras  # Add this import
from PIL import Image, ImageOps
import time
import json
from numpy import ndarray
from modules.predicter import predict_image, predict_video, predict_frame

import modules.globals
import modules.metadata
from modules.face_analyser import (
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    add_blank_map,
    has_valid_map,
    simplify_maps,
)
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    is_image,
    is_video,
    resolve_relative_path,
    has_image_extension,
)

ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 960
PREVIEW_DEFAULT_HEIGHT = 540

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

img_ft, vid_ft = modules.globals.file_types


def save_switch_states():
    switch_states = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box
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
        modules.globals.color_correction = switch_states.get("color_correction", False)
        modules.globals.nsfw_filter = switch_states.get("nsfw_filter", False)
        modules.globals.live_mirror = switch_states.get("live_mirror", False)
        modules.globals.live_resizable = switch_states.get("live_resizable", False)
        modules.globals.fp_ui = switch_states.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = switch_states.get("show_fps", False)
        modules.globals.mouth_mask = switch_states.get("mouth_mask", False)
        modules.globals.show_mouth_mask_box = switch_states.get("show_mouth_mask_box", False)
    except FileNotFoundError:
        # If the file doesn't exist, use default values
        pass
    
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
    return cv2.resize(image, dsize=new_size)

def get_available_cameras():
        """Returns a list of available camera names and indices."""
        camera_indices = []
        camera_names = []

        for camera in enumerate_cameras():
            cap = cv2.VideoCapture(camera.index)
            if cap.isOpened():
                camera_indices.append(camera.index)
                camera_names.append(camera.name)
                cap.release()
        return (camera_indices, camera_names)

class DeepFakeUI:    
    preview_label: ctk.CTkLabel
    source_label: ctk.CTkLabel
    target_label: ctk.CTkLabel
    status_label: ctk.CTkLabel
    popup_status_label: ctk.CTkLabel
    popup_status_label_live: ctk.CTkLabel
    preview_slider: ctk.CTkSlider
    source_label_dict: dict[int, ctk.CTkLabel] = {}
    source_label_dict_live: dict[int, ctk.CTkLabel] = {}
    target_label_dict_live: dict[int, ctk.CTkLabel] = {}
    source_label_dict_live = {}
    target_label_dict_live = {}
    popup_live: ctk.CTkToplevel
    popup: ctk.CTkToplevel = None
    
    recent_directory_source: str = os.path.expanduser("~")
    recent_directory_target: str = os.path.expanduser("~")
    recent_directory_output: str = os.path.expanduser("~")

    def __init__(self, start: Callable[[], None], destroy: Callable[[], None]) -> None:
        self.root = self.create_root(start, destroy)
        self.preview = self.create_preview(self.root)

    def create_root(self, start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
        load_switch_states()

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

        source_label = ctk.CTkLabel(root, text=None)
        source_label.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.25)

        target_label = ctk.CTkLabel(root, text=None)
        target_label.place(relx=0.6, rely=0.1, relwidth=0.3, relheight=0.25)

        select_face_button = ctk.CTkButton(
            root, text="Select a face", cursor="hand2", command=lambda: self.select_source_path()
        )
        select_face_button.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)

        swap_faces_button = ctk.CTkButton(
            root, text="↔", cursor="hand2", command=lambda: self.swap_faces_paths()
        )
        swap_faces_button.place(relx=0.45, rely=0.4, relwidth=0.1, relheight=0.1)

        select_target_button = ctk.CTkButton(
            root,
            text="Select a target",
            cursor="hand2",
            command=lambda: self.select_target_path(),
        )
        select_target_button.place(relx=0.6, rely=0.4, relwidth=0.3, relheight=0.1)

        keep_fps_value = ctk.BooleanVar(value=modules.globals.keep_fps)
        keep_fps_checkbox = ctk.CTkSwitch(
            root,
            text="Keep fps",
            variable=keep_fps_value,
            cursor="hand2",
            command=lambda: (
                setattr(modules.globals, "keep_fps", keep_fps_value.get()),
                save_switch_states(),
            ),
        )
        keep_fps_checkbox.place(relx=0.1, rely=0.6)

        keep_frames_value = ctk.BooleanVar(value=modules.globals.keep_frames)
        keep_frames_switch = ctk.CTkSwitch(
            root,
            text="Keep frames",
            variable=keep_frames_value,
            cursor="hand2",
            command=lambda: (
                setattr(modules.globals, "keep_frames", keep_frames_value.get()),
                save_switch_states(),
            ),
        )
        keep_frames_switch.place(relx=0.1, rely=0.65)

        enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui["face_enhancer"])
        enhancer_switch = ctk.CTkSwitch(
            root,
            text="Face Enhancer",
            variable=enhancer_value,
            cursor="hand2",
            command=lambda: (
                self.update_tumbler("face_enhancer", enhancer_value.get()),
                save_switch_states(),
            ),
        )
        enhancer_switch.place(relx=0.1, rely=0.7)

        keep_audio_value = ctk.BooleanVar(value=modules.globals.keep_audio)
        keep_audio_switch = ctk.CTkSwitch(
            root,
            text="Keep audio",
            variable=keep_audio_value,
            cursor="hand2",
            command=lambda: (
                setattr(modules.globals, "keep_audio", keep_audio_value.get()),
                save_switch_states(),
            ),
        )
        keep_audio_switch.place(relx=0.6, rely=0.6)

        many_faces_value = ctk.BooleanVar(value=modules.globals.many_faces)
        many_faces_switch = ctk.CTkSwitch(
            root,
            text="Many faces",
            variable=many_faces_value,
            cursor="hand2",
            command=lambda: (
                setattr(modules.globals, "many_faces", many_faces_value.get()),
                save_switch_states(),
            ),
        )
        many_faces_switch.place(relx=0.6, rely=0.65)

        color_correction_value = ctk.BooleanVar(value=modules.globals.color_correction)
        color_correction_switch = ctk.CTkSwitch(
            root,
            text="Fix Blueish Cam",
            variable=color_correction_value,
            cursor="hand2",
            command=lambda: (
                setattr(modules.globals, "color_correction", color_correction_value.get()),
                save_switch_states(),
            ),
        )
        color_correction_switch.place(relx=0.6, rely=0.70)

        #    nsfw_value = ctk.BooleanVar(value=modules.globals.nsfw_filter)
        #    nsfw_switch = ctk.CTkSwitch(root, text='NSFW filter', variable=nsfw_value, cursor='hand2', command=lambda: setattr(modules.globals, 'nsfw_filter', nsfw_value.get()))
        #    nsfw_switch.place(relx=0.6, rely=0.7)

        map_faces = ctk.BooleanVar(value=modules.globals.map_faces)
        map_faces_switch = ctk.CTkSwitch(
            root,
            text="Map faces",
            variable=map_faces,
            cursor="hand2",
            command=lambda: (
                setattr(modules.globals, "map_faces", map_faces.get()),
                save_switch_states(),
            ),
        )
        map_faces_switch.place(relx=0.1, rely=0.75)

        show_fps_value = ctk.BooleanVar(value=modules.globals.show_fps)
        show_fps_switch = ctk.CTkSwitch(
            root,
            text="Show FPS",
            variable=show_fps_value,
            cursor="hand2",
            command=lambda: (
                setattr(modules.globals, "show_fps", show_fps_value.get()),
                save_switch_states(),
            ),
        )
        show_fps_switch.place(relx=0.6, rely=0.75)

        mouth_mask_var = ctk.BooleanVar(value=modules.globals.mouth_mask)
        mouth_mask_switch = ctk.CTkSwitch(
            root,
            text="Mouth Mask",
            variable=mouth_mask_var,
            cursor="hand2",
            command=lambda: setattr(modules.globals, "mouth_mask", mouth_mask_var.get()),
        )
        mouth_mask_switch.place(relx=0.1, rely=0.55)

        show_mouth_mask_box_var = ctk.BooleanVar(value=modules.globals.show_mouth_mask_box)
        show_mouth_mask_box_switch = ctk.CTkSwitch(
            root,
            text="Show Mouth Mask Box",
            variable=show_mouth_mask_box_var,
            cursor="hand2",
            command=lambda: setattr(
                modules.globals, "show_mouth_mask_box", show_mouth_mask_box_var.get()
            ),
        )
        show_mouth_mask_box_switch.place(relx=0.6, rely=0.55)

        start_button = ctk.CTkButton(
            root, text="Start", cursor="hand2", command=lambda: self.analyze_target(start, root)
        )
        start_button.place(relx=0.15, rely=0.80, relwidth=0.2, relheight=0.05)

        stop_button = ctk.CTkButton(
            root, text="Destroy", cursor="hand2", command=lambda: destroy()
        )
        stop_button.place(relx=0.4, rely=0.80, relwidth=0.2, relheight=0.05)

        preview_button = ctk.CTkButton(
            root, text="Preview", cursor="hand2", command=lambda: self.toggle_preview()
        )
        preview_button.place(relx=0.65, rely=0.80, relwidth=0.2, relheight=0.05)

        # --- Camera Selection ---
        camera_label = ctk.CTkLabel(root, text="Select Camera:")
        camera_label.place(relx=0.1, rely=0.86, relwidth=0.2, relheight=0.05)

        available_cameras = get_available_cameras()
        # Convert camera indices to strings for CTkOptionMenu
        available_camera_indices, available_camera_strings = available_cameras
        camera_variable = ctk.StringVar(
            value=(
                available_camera_strings[0]
                if available_camera_strings
                else "No cameras found"
            )
        )
        camera_optionmenu = ctk.CTkOptionMenu(
            root, variable=camera_variable, values=available_camera_strings
        )
        camera_optionmenu.place(relx=0.35, rely=0.86, relwidth=0.25, relheight=0.05)

        live_button = ctk.CTkButton(
            root,
            text="Live",
            cursor="hand2",
            command=lambda: self.webcam_preview(
                root,
                available_camera_indices[
                    available_camera_strings.index(camera_variable.get())
                ],
            ),
        )
        live_button.place(relx=0.65, rely=0.86, relwidth=0.2, relheight=0.05)
        # --- End Camera Selection ---

        status_label = ctk.CTkLabel(root, text=None, justify="center")
        status_label.place(relx=0.1, rely=0.9, relwidth=0.8)

        donate_label = ctk.CTkLabel(
            root, text="Deep Live Cam", justify="center", cursor="hand2"
        )
        donate_label.place(relx=0.1, rely=0.95, relwidth=0.8)
        donate_label.configure(
            text_color=ctk.ThemeManager.theme.get("URL").get("text_color")
        )
        donate_label.bind(
            "<Button>", lambda event: webbrowser.open("https://paypal.me/hacksider")
        )
        
        self.source_label = source_label
        self.target_label = target_label
        self.status_label = status_label

        return root


    def analyze_target(self, start: Callable[[], None], root: ctk.CTk):
        if self.popup != None and self.popup.winfo_exists():
            self.update_status("Please complete pop-up or close it.")
            return

        if modules.globals.map_faces:
            modules.globals.souce_target_map = []

            if is_image(modules.globals.target_path):
                self.update_status("Getting unique faces")
                get_unique_faces_from_target_image()
            elif is_video(modules.globals.target_path):
                self.update_status("Getting unique faces")
                get_unique_faces_from_target_video()

            if len(modules.globals.souce_target_map) > 0:
                self.create_source_target_popup(start, root, modules.globals.souce_target_map)
            else:
                self.update_status("No faces found in target")
        else:
            self.select_output_path(start)


    def create_source_target_popup(
        self, start: Callable[[], None], root: ctk.CTk, map: list
    ) -> None:
        popup = ctk.CTkToplevel(root)
        popup.title("Source x Target Mapper")
        popup.geometry(f"{POPUP_WIDTH}x{POPUP_HEIGHT}")
        popup.focus()

        def on_submit_click(start):
            if has_valid_map():
                popup.destroy()
                self.select_output_path(start)
            else:
                self.update_pop_status("Atleast 1 source with target is required!")

        scrollable_frame = ctk.CTkScrollableFrame(
            popup, width=POPUP_SCROLL_WIDTH, height=POPUP_SCROLL_HEIGHT
        )
        scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        def on_button_click(map, button_num):
            map = self.update_popup_source(scrollable_frame, map, button_num)

        for item in map:
            id = item["id"]

            button = ctk.CTkButton(
                scrollable_frame,
                text="Select source image",
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

            image = Image.fromarray(cv2.cvtColor(item["target"]["cv2"], cv2.COLOR_BGR2RGB))
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

        popup_status_label = ctk.CTkLabel(popup, text=None, justify="center")
        popup_status_label.grid(row=1, column=0, pady=15)

        close_button = ctk.CTkButton(
            popup, text="Submit", command=lambda: on_submit_click(start)
        )
        close_button.grid(row=2, column=0, pady=10)

        self.popup_status_label = popup_status_label
        self.popup = popup


    def update_popup_source(
        self, scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
    ) -> list:
        source_path = ctk.filedialog.askopenfilename(
            title="select an source image",
            initialdir=self.recent_directory_source,
            filetypes=[img_ft],
        )

        if "source" in map[button_num]:
            map[button_num].pop("source")
            self.source_label_dict[button_num].destroy()
            del self.source_label_dict[button_num]

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
                    cv2.cvtColor(map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
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
                self.source_label_dict[button_num] = source_image
            else:
                self.update_pop_status("Face could not be detected in last upload!")
            return map


    def create_preview(self, parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
        preview = ctk.CTkToplevel(parent)
        preview.withdraw()
        preview.title("Preview")
        preview.configure()
        preview.protocol("WM_DELETE_WINDOW", lambda: self.toggle_preview())
        preview.resizable(width=True, height=True)

        self.preview_label = ctk.CTkLabel(preview, text=None)
        self.preview_label.pack(fill="both", expand=True)

        self.preview_slider = ctk.CTkSlider(
            preview, from_=0, to=0, command=lambda frame_value: self.update_preview(frame_value)
        )

        return preview


    def update_status(self, text: str) -> None:
        self.status_label.configure(text=text)
        self.root.update()


    def update_pop_status(self, text: str) -> None:
        self.popup_status_label.configure(text=text)


    def update_pop_live_status(self, text: str) -> None:
        self.popup_status_label_live.configure(text=text)


    def update_tumbler(self, var: str, value: bool) -> None:
        modules.globals.fp_ui[var] = value
        save_switch_states()
        # If we're currently in a live preview, update the frame processors
        if self.preview.state() == "normal":
            self.frame_processors = get_frame_processors_modules(
                modules.globals.frame_processors
            )


    def select_source_path(self) -> None:
        self.preview.withdraw()
        source_path = ctk.filedialog.askopenfilename(
            title="select an source image",
            initialdir=self.recent_directory_source,
            filetypes=[img_ft],
        )
        if is_image(source_path):
            modules.globals.source_path = source_path
            self.recent_directory_source = os.path.dirname(modules.globals.source_path)
            image = self.render_image_preview(modules.globals.source_path, (200, 200))
            self.source_label.configure(image=image)
        else:
            modules.globals.source_path = None
            self.source_label.configure(image=None)


    def swap_faces_paths(self) -> None:
        source_path = modules.globals.source_path
        target_path = modules.globals.target_path

        if not is_image(source_path) or not is_image(target_path):
            return

        modules.globals.source_path = target_path
        modules.globals.target_path = source_path

        self.recent_directory_source = os.path.dirname(modules.globals.source_path)
        self.recent_directory_target = os.path.dirname(modules.globals.target_path)

        self.preview.withdraw()

        source_image = self.render_image_preview(modules.globals.source_path, (200, 200))
        self.source_label.configure(image=source_image)

        target_image = self.render_image_preview(modules.globals.target_path, (200, 200))
        self.target_label.configure(image=target_image)


    def select_target_path(self) -> None:
        self.preview.withdraw()
        target_path = ctk.filedialog.askopenfilename(
            title="select an target image or video",
            initialdir=self.recent_directory_target,
            filetypes=[img_ft, vid_ft],
        )
        if is_image(target_path):
            modules.globals.target_path = target_path
            self.recent_directory_target = os.path.dirname(modules.globals.target_path)
            image = self.render_image_preview(modules.globals.target_path, (200, 200))
            self.target_label.configure(image=image)
        elif is_video(target_path):
            modules.globals.target_path = target_path
            self.recent_directory_target = os.path.dirname(modules.globals.target_path)
            video_frame = self.render_video_preview(target_path, (200, 200))
            self.target_label.configure(image=video_frame)
        else:
            modules.globals.target_path = None
            self.target_label.configure(image=None)


    def select_output_path(self, start: Callable[[], None]) -> None:
        if is_image(modules.globals.target_path):
            output_path = ctk.filedialog.asksaveasfilename(
                title="save image output file",
                filetypes=[img_ft],
                defaultextension=".png",
                initialfile="output.png",
                initialdir=self.recent_directory_output,
            )
        elif is_video(modules.globals.target_path):
            output_path = ctk.filedialog.asksaveasfilename(
                title="save video output file",
                filetypes=[vid_ft],
                defaultextension=".mp4",
                initialfile="output.mp4",
                initialdir=self.recent_directory_output,
            )
        else:
            output_path = None
        if output_path:
            modules.globals.output_path = output_path
            self.recent_directory_output = os.path.dirname(modules.globals.output_path)
            start()

    def check_and_ignore_nsfw(self, target: str | ndarray, destroy: Callable | None = None) -> bool:
        """Check if the target is NSFW.
        TODO: Consider to make blur the target.
        """

        if type(target) is str:  # image/video file path
            check_nsfw = predict_image if has_image_extension(target) else predict_video
        elif type(target) is ndarray:  # frame object
            check_nsfw = predict_frame
        if check_nsfw and check_nsfw(target):
            if destroy:
                destroy(
                    to_quit=False
                )  # Do not need to destroy the window frame if the target is NSFW
            self.update_status("Processing ignored!")
            return True
        else:
            return False

    def render_image_preview(self, image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
        image = Image.open(image_path)
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)


    def render_video_preview(
        self, video_path: str, size: Tuple[int, int], frame_number: int = 0
    ) -> None:
        capture = cv2.VideoCapture(video_path)
        if frame_number:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        has_frame, frame = capture.read()
        if has_frame:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if size:
                image = ImageOps.fit(image, size, Image.LANCZOS)
            return ctk.CTkImage(image, size=image.size)
        capture.release()
        cv2.destroyAllWindows()


    def toggle_preview(self) -> None:
        if self.preview.state() == "normal":
            self.preview.withdraw()
        elif modules.globals.source_path and modules.globals.target_path:
            self.init_preview()
            self.update_preview()


    def init_preview(self) -> None:
        if is_image(modules.globals.target_path):
            self.preview_slider.pack_forget()
        if is_video(modules.globals.target_path):
            video_frame_total = get_video_frame_total(modules.globals.target_path)
            self.preview_slider.configure(to=video_frame_total)
            self.preview_slider.pack(fill="x")
            self.preview_slider.set(0)


    def update_preview(self, frame_number: int = 0) -> None:
        if modules.globals.source_path and modules.globals.target_path:
            self.update_status("Processing...")
            temp_frame = get_video_frame(modules.globals.target_path, frame_number)
            if modules.globals.nsfw_filter and self.check_and_ignore_nsfw(temp_frame):
                return
            for frame_processor in get_frame_processors_modules(
                modules.globals.frame_processors
            ):
                temp_frame = frame_processor.process_frame(
                    get_one_face(cv2.imread(modules.globals.source_path)), temp_frame
                )
            image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
            image = ImageOps.contain(
                image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            image = ctk.CTkImage(image, size=image.size)
            self.preview_label.configure(image=image)
            self.update_status("Processing succeed!")
            self.preview.deiconify()


    def webcam_preview(self, root: ctk.CTk, camera_index: int):
        if not modules.globals.map_faces:
            if modules.globals.source_path is None:
                # No image selected
                return
            self.create_webcam_preview(camera_index)
        else:
            modules.globals.souce_target_map = []
            self.create_source_target_popup_for_webcam(
                root, modules.globals.souce_target_map, camera_index
            )

    def create_webcam_preview(self, camera_index: int):
        camera = cv2.VideoCapture(camera_index)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_DEFAULT_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_DEFAULT_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, 60)

        self.preview_label.configure(width=PREVIEW_DEFAULT_WIDTH, height=PREVIEW_DEFAULT_HEIGHT)

        self.preview.deiconify()

        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)

        source_image = None
        prev_time = time.time()
        fps_update_interval = 0.5  # Update FPS every 0.5 seconds
        frame_count = 0
        fps = 0

        while camera:
            ret, frame = camera.read()
            if not ret:
                break

            temp_frame = frame.copy()

            if modules.globals.live_mirror:
                temp_frame = cv2.flip(temp_frame, 1)

            if modules.globals.live_resizable:
                temp_frame = fit_image_to_size(
                    temp_frame, self.preview.winfo_width(), self.preview.winfo_height()
                )

            if not modules.globals.map_faces:
                if source_image is None and modules.globals.source_path:
                    source_image = get_one_face(cv2.imread(modules.globals.source_path))

                for frame_processor in frame_processors:
                    if frame_processor.NAME == "DLC.FACE-ENHANCER":
                        if modules.globals.fp_ui["face_enhancer"]:
                            temp_frame = frame_processor.process_frame(None, temp_frame)
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

            image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageOps.contain(
                image, (temp_frame.shape[1], temp_frame.shape[0]), Image.LANCZOS
            )
            image = ctk.CTkImage(image, size=image.size)
            self.preview_label.configure(image=image)
            self.root.update()

            if self.preview.state() == "withdrawn":
                break

        camera.release()
        self.preview.withdraw()


    def create_source_target_popup_for_webcam(
        self, root: ctk.CTk, map: list, camera_index: int
    ) -> None:
        self.popup_live = ctk.CTkToplevel(root)
        self.popup_live.title("Source x Target Mapper")
        self.popup_live.geometry(f"{POPUP_LIVE_WIDTH}x{POPUP_LIVE_HEIGHT}")
        self.popup_live.focus()

        def on_submit_click():
            if has_valid_map():
                self.popup_live.destroy()
                simplify_maps()
                self.create_webcam_preview(camera_index)
            else:
                self.update_pop_live_status("At least 1 source with target is required!")

        def on_add_click():
            add_blank_map()
            self.refresh_data(map)
            self.update_pop_live_status("Please provide mapping!")

        popup_status_label_live = ctk.CTkLabel(self.popup_live, text=None, justify="center")
        popup_status_label_live.grid(row=1, column=0, pady=15)

        add_button = ctk.CTkButton(self.popup_live, text="Add", command=lambda: on_add_click())
        add_button.place(relx=0.2, rely=0.92, relwidth=0.2, relheight=0.05)

        close_button = ctk.CTkButton(
            self.popup_live, text="Submit", command=lambda: on_submit_click()
        )
        close_button.place(relx=0.6, rely=0.92, relwidth=0.2, relheight=0.05)
        
        self.popup_status_label_live = popup_status_label_live


    def refresh_data(self, map: list):
        scrollable_frame = ctk.CTkScrollableFrame(
            self.popup_live, width=POPUP_LIVE_SCROLL_WIDTH, height=POPUP_LIVE_SCROLL_HEIGHT
        )
        scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

        def on_sbutton_click(map, button_num):
            map = self.update_webcam_source(scrollable_frame, map, button_num)

        def on_tbutton_click(map, button_num):
            map = self.update_webcam_target(scrollable_frame, map, button_num)

        for item in map:
            id = item["id"]

            button = ctk.CTkButton(
                scrollable_frame,
                text="Select source image",
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
                text="Select target image",
                command=lambda id=id: on_tbutton_click(map, id),
                width=DEFAULT_BUTTON_WIDTH,
                height=DEFAULT_BUTTON_HEIGHT,
            )
            button.grid(row=id, column=3, padx=20, pady=10)

            if "source" in item:
                image = Image.fromarray(
                    cv2.cvtColor(item["source"]["cv2"], cv2.COLOR_BGR2RGB)
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
                    cv2.cvtColor(item["target"]["cv2"], cv2.COLOR_BGR2RGB)
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
        self, scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
    ) -> list:
        source_path = ctk.filedialog.askopenfilename(
            title="select an source image",
            initialdir=self.recent_directory_source,
            filetypes=[img_ft],
        )

        if "source" in map[button_num]:
            map[button_num].pop("source")
            self.source_label_dict_live[button_num].destroy()
            del self.source_label_dict_live[button_num]

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
                    cv2.cvtColor(map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
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
                self.source_label_dict_live[button_num] = source_image
            else:
                self.update_pop_live_status("Face could not be detected in last upload!")
            return map


    def update_webcam_target(
        self, scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
    ) -> list:
        target_path = ctk.filedialog.askopenfilename(
            title="select an target image",
            initialdir=self.recent_directory_source,
            filetypes=[img_ft],
        )

        if "target" in map[button_num]:
            map[button_num].pop("target")
            self.target_label_dict_live[button_num].destroy()
            del self.target_label_dict_live[button_num]

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
                    cv2.cvtColor(map[button_num]["target"]["cv2"], cv2.COLOR_BGR2RGB)
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
                self.target_label_dict_live[button_num] = target_image
            else:
                self.update_pop_live_status("Face could not be detected in last upload!")
            return map
