import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from PIL import Image, ImageOps

import roop.globals
import roop.metadata
from roop.face_analyser import get_one_face
from roop.capturer import get_video_frame, get_video_frame_total
from roop.predicter import predict_frame
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import is_image, is_video, resolve_relative_path

import cv2

ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(f'{roop.metadata.name} {roop.metadata.version}')
    root.configure()
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())

    source_label = ctk.CTkLabel(root, text=None)
    source_label.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.25)

    target_label = ctk.CTkLabel(root, text=None)
    target_label.place(relx=0.6, rely=0.1, relwidth=0.3, relheight=0.25)

    source_button = ctk.CTkButton(root, text='Select a face', cursor='hand2', command=lambda: select_source_path())
    source_button.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)

    target_button = ctk.CTkButton(root, text='Select a target', cursor='hand2', command=lambda: select_target_path())
    target_button.place(relx=0.6, rely=0.4, relwidth=0.3, relheight=0.1)

    keep_fps_value = ctk.BooleanVar(value=roop.globals.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(root, text='Keep fps', variable=keep_fps_value, cursor='hand2', command=lambda: setattr(roop.globals, 'keep_fps', not roop.globals.keep_fps))
    keep_fps_checkbox.place(relx=0.1, rely=0.6)

    keep_frames_value = ctk.BooleanVar(value=roop.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(root, text='Keep frames', variable=keep_frames_value, cursor='hand2', command=lambda: setattr(roop.globals, 'keep_frames', keep_frames_value.get()))
    keep_frames_switch.place(relx=0.1, rely=0.65)

    keep_audio_value = ctk.BooleanVar(value=roop.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(root, text='Keep audio', variable=keep_audio_value, cursor='hand2', command=lambda: setattr(roop.globals, 'keep_audio', keep_audio_value.get()))
    keep_audio_switch.place(relx=0.6, rely=0.6)

    many_faces_value = ctk.BooleanVar(value=roop.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(root, text='Many faces', variable=many_faces_value, cursor='hand2', command=lambda: setattr(roop.globals, 'many_faces', many_faces_value.get()))
    many_faces_switch.place(relx=0.6, rely=0.65)

    start_button = ctk.CTkButton(root, text='Start', cursor='hand2', command=lambda: select_output_path(start))
    start_button.place(relx=0.15, rely=0.75, relwidth=0.2, relheight=0.05)

    stop_button = ctk.CTkButton(root, text='Destroy', cursor='hand2', command=lambda: destroy())
    stop_button.place(relx=0.4, rely=0.75, relwidth=0.2, relheight=0.05)

    preview_button = ctk.CTkButton(root, text='Preview', cursor='hand2', command=lambda: toggle_preview())
    preview_button.place(relx=0.65, rely=0.75, relwidth=0.2, relheight=0.05)
    
    live_button = ctk.CTkButton(root, text='Live', cursor='hand2', command=lambda: webcam_preview())
    live_button.place(relx=0.40, rely=0.83, relwidth=0.2, relheight=0.05)

    status_label = ctk.CTkLabel(root, text=None, justify='center')
    status_label.place(relx=0.1, rely=0.9, relwidth=0.8)

    donate_label = ctk.CTkLabel(root, text='Send some love!', justify='center', cursor='hand2')
    donate_label.place(relx=0.1, rely=0.95, relwidth=0.8)
    donate_label.configure(text_color=ctk.ThemeManager.theme.get('RoopDonate').get('text_color'))
    donate_label.bind('<Button>', lambda event: webbrowser.open('https://paypal.me/hacksider'))

    return root


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title('Preview')
    preview.configure()
    preview.protocol('WM_DELETE_WINDOW', lambda: toggle_preview())
    preview.resizable(width=False, height=False)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill='both', expand=True)

    preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value))

    return preview


def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(title='select an source image', initialdir=RECENT_DIRECTORY_SOURCE)
    if is_image(source_path):
        roop.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(roop.globals.source_path)
        image = render_image_preview(roop.globals.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        roop.globals.source_path = None
        source_label.configure(image=None)


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(title='select an target image or video', initialdir=RECENT_DIRECTORY_TARGET)
    if is_image(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
        image = render_image_preview(roop.globals.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        roop.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT

    if is_image(roop.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save image output file', defaultextension='.png', initialfile='output.png', initialdir=RECENT_DIRECTORY_OUTPUT)
    elif is_video(roop.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='save video output file', defaultextension='.mp4', initialfile='output.mp4', initialdir=RECENT_DIRECTORY_OUTPUT)
    else:
        output_path = None
    if output_path:
        roop.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(roop.globals.output_path)
        start()


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(video_path: str, size: Tuple[int, int], frame_number: int = 0) -> ctk.CTkImage:
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


def toggle_preview() -> None:
    if PREVIEW.state() == 'normal':
        PREVIEW.withdraw()
    elif roop.globals.source_path and roop.globals.target_path:
        init_preview()
        update_preview()
        PREVIEW.deiconify()


def init_preview() -> None:
    if is_image(roop.globals.target_path):
        preview_slider.pack_forget()
    if is_video(roop.globals.target_path):
        video_frame_total = get_video_frame_total(roop.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill='x')
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if roop.globals.source_path and roop.globals.target_path:
        temp_frame = get_video_frame(roop.globals.target_path, frame_number)
        if predict_frame(temp_frame):
            quit()
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            temp_frame = frame_processor.process_frame(
                get_one_face(cv2.imread(roop.globals.source_path)),
                temp_frame
            )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        

def webcam_preview():
    if roop.globals.source_path is None:
        # No image selected
        return
    
    global preview_label, PREVIEW

    cap = cv2.VideoCapture(0)  # Use index for the webcam (adjust the index accordingly if necessary)    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set the width of the resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set the height of the resolution
    cap.set(cv2.CAP_PROP_FPS, 60)  # Set the frame rate of the webcam
    PREVIEW_MAX_HEIGHT = 720
    PREVIEW_MAX_WIDTH = 1280

    preview_label.configure(image=None)  # Reset the preview image before startup

    PREVIEW.deiconify()  # Open preview window

    frame_processors = get_frame_processors_modules(roop.globals.frame_processors)

    source_image = None  # Initialize variable for the selected face image

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Select and save face image only once
        if source_image is None and roop.globals.source_path:
            source_image = get_one_face(cv2.imread(roop.globals.source_path))

        temp_frame = frame.copy()  #Create a copy of the frame

        for frame_processor in frame_processors:
            temp_frame = frame_processor.process_frame(source_image, temp_frame)

        image = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)  # Convert the image to RGB format to display it with Tkinter
        image = Image.fromarray(image)
        image = ImageOps.contain(image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        ROOT.update()

    cap.release()
    PREVIEW.withdraw()  # Close preview window when loop is finished