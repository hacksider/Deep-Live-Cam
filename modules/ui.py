import os
import platform
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple, List, Any
from types import ModuleType
import cv2
from PIL import Image, ImageOps
import pyvirtualcam

# Import OS-specific modules only when necessary
if platform.system() == 'Darwin':  # macOS
    import AVFoundation

if os.name == 'nt':  # Windows
    from pygrabber.dshow_graph import FilterGraph


import modules.globals
import modules.metadata
from modules.face_analyser import get_one_face
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import is_image, is_video, resolve_relative_path

ROOT = None
ROOT_HEIGHT = 800
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH  = 1200
PREVIEW_DEFAULT_WIDTH  = 960
PREVIEW_DEFAULT_HEIGHT = 540

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None

img_ft, vid_ft = modules.globals.file_types

camera = None

def check_camera_permissions():
    """Check and request camera access permission on macOS."""
    if platform.system() == 'Darwin':  # macOS-specific
        auth_status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(AVFoundation.AVMediaTypeVideo)

        if auth_status == AVFoundation.AVAuthorizationStatusNotDetermined:
            # Request access to the camera
            def completion_handler(granted):
                if granted:
                    print("Access granted to the camera.")
                else:
                    print("Access denied to the camera.")
            
            AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVFoundation.AVMediaTypeVideo, completion_handler)
        elif auth_status == AVFoundation.AVAuthorizationStatusAuthorized:
            print("Camera access already authorized.")
        elif auth_status == AVFoundation.AVAuthorizationStatusDenied:
            print("Camera access denied. Please enable it in System Preferences.")
        elif auth_status == AVFoundation.AVAuthorizationStatusRestricted:
            print("Camera access restricted. The app is not allowed to use the camera.")


def select_camera(camera_name: str):
    """Select the appropriate camera based on its name (cross-platform)."""
    if platform.system() == 'Darwin':  # macOS-specific
        devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
        for device in devices:
            if device.localizedName() == camera_name:
                return device
    elif platform.system() == 'Windows' or platform.system() == 'Linux':
        # On Windows/Linux, simply return the camera name as OpenCV can handle it by index
        return camera_name
    return None


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    if platform.system() == 'Darwin':  # macOS-specific
        check_camera_permissions()  # Check camera permissions before initializing the UI
    
    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    print("Creating root window...")
    
    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(f'{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}')
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())

    source_label = ctk.CTkLabel(root, text=None)
    source_label.place(relx=0.1, rely=0.0875, relwidth=0.3, relheight=0.25)

    target_label = ctk.CTkLabel(root, text=None)
    target_label.place(relx=0.6, rely=0.0875, relwidth=0.3, relheight=0.25)

    source_button = ctk.CTkButton(root, text='Select a face', cursor='hand2', command=select_source_path)
    source_button.place(relx=0.1, rely=0.35, relwidth=0.3, relheight=0.1)

    swap_faces_button = ctk.CTkButton(root, text='↔', cursor='hand2', command=lambda: swap_faces_paths())
    swap_faces_button.place(relx=0.45, rely=0.4, relwidth=0.1, relheight=0.1)

    target_button = ctk.CTkButton(root, text='Select a target', cursor='hand2', command=select_target_path)
    target_button.place(relx=0.6, rely=0.35, relwidth=0.3, relheight=0.1)

    keep_fps_value = ctk.BooleanVar(value=modules.globals.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(root, text='Keep fps', variable=keep_fps_value, cursor='hand2', command=lambda: setattr(modules.globals, 'keep_fps', not modules.globals.keep_fps))
    keep_fps_checkbox.place(relx=0.1, rely=0.525)

    keep_frames_value = ctk.BooleanVar(value=modules.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(root, text='Keep frames', variable=keep_frames_value, cursor='hand2', command=lambda: setattr(modules.globals, 'keep_frames', keep_frames_value.get()))
    keep_frames_switch.place(relx=0.1, rely=0.56875)

    enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui['face_enhancer'])
    enhancer_switch = ctk.CTkSwitch(root, text='Face Enhancer', variable=enhancer_value, cursor='hand2', command=lambda: update_tumbler('face_enhancer', enhancer_value.get()))
    enhancer_switch.place(relx=0.1, rely=0.6125)

    keep_audio_value = ctk.BooleanVar(value=modules.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(root, text='Keep audio', variable=keep_audio_value, cursor='hand2', command=lambda: setattr(modules.globals, 'keep_audio', keep_audio_value.get()))
    keep_audio_switch.place(relx=0.6, rely=0.525)

    many_faces_value = ctk.BooleanVar(value=modules.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(root, text='Many faces', variable=many_faces_value, cursor='hand2', command=lambda: setattr(modules.globals, 'many_faces', many_faces_value.get()))
    many_faces_switch.place(relx=0.6, rely=0.56875)

    nsfw_value = ctk.BooleanVar(value=modules.globals.nsfw)
    nsfw_switch = ctk.CTkSwitch(root, text='NSFW', variable=nsfw_value, cursor='hand2', command=lambda: setattr(modules.globals, 'nsfw', nsfw_value.get()))
    nsfw_switch.place(relx=0.6, rely=0.6125)

    start_button = ctk.CTkButton(root, text='Start', cursor='hand2', command=lambda: select_output_path(start))
    start_button.place(relx=0.15, rely=0.7, relwidth=0.2, relheight=0.05)

    stop_button = ctk.CTkButton(root, text='Destroy', cursor='hand2', command=destroy)
    stop_button.place(relx=0.4, rely=0.7, relwidth=0.2, relheight=0.05)

    preview_button = ctk.CTkButton(root, text='Preview', cursor='hand2', command=toggle_preview)
    preview_button.place(relx=0.65, rely=0.7, relwidth=0.2, relheight=0.05)

    camera_label = ctk.CTkLabel(root, text="Select Camera:")
    camera_label.place(relx=0.4, rely=0.7525, relwidth=0.2, relheight=0.05)

    available_cameras = get_available_cameras()
    available_camera_strings = [str(cam) for cam in available_cameras]

    camera_variable = ctk.StringVar(value=available_camera_strings[0] if available_camera_strings else "No cameras found")
    camera_optionmenu = ctk.CTkOptionMenu(root, variable=camera_variable, values=available_camera_strings)
    camera_optionmenu.place(relx=0.65, rely=0.7525, relwidth=0.2, relheight=0.05)

    virtual_cam_out_value = ctk.BooleanVar(value=False)
    virtual_cam_out_switch = ctk.CTkSwitch(root, text='Virtual Cam Output (OBS)', variable=virtual_cam_out_value, cursor='hand2')
    virtual_cam_out_switch.place(relx=0.4, rely=0.805)

    live_button = ctk.CTkButton(root, text='Live', cursor='hand2', command=lambda: webcam_preview(camera_variable.get(), virtual_cam_out_value.get()))
    live_button.place(relx=0.15, rely=0.7525, relwidth=0.2, relheight=0.05)

    status_label = ctk.CTkLabel(root, text=None, justify='center')
    status_label.place(relx=0.1, relwidth=0.8, rely=0.875)

    donate_label = ctk.CTkLabel(root, text='Deep Live Cam', justify='center', cursor='hand2')
    donate_label.place(relx=0.1, rely=0.95, relwidth=0.8)
    donate_label.configure(text_color=ctk.ThemeManager.theme.get('URL').get('text_color'))
    donate_label.bind('<Button-1>', lambda event: webbrowser.open('https://paypal.me/hacksider'))

    return root


def create_preview(parent: ctk.CTk) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title('Preview')
    preview.protocol('WM_DELETE_WINDOW', toggle_preview)
    preview.resizable(width=True, height=True)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill='both', expand=True)

    preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=update_preview)

    return preview


def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(title='Select a source image', initialdir=RECENT_DIRECTORY_SOURCE, filetypes=[img_ft])
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
    global RECENT_DIRECTORY_TARGET

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(title='Select a target image or video', initialdir=RECENT_DIRECTORY_TARGET, filetypes=[img_ft, vid_ft])
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
    global RECENT_DIRECTORY_OUTPUT

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='Save image output file', filetypes=[img_ft], defaultextension='.png', initialfile='output.png', initialdir=RECENT_DIRECTORY_OUTPUT)
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='Save video output file', filetypes=[vid_ft], defaultextension='.mp4', initialfile='output.mp4', initialdir=RECENT_DIRECTORY_OUTPUT)
    else:
        output_path = None
    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
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
    capture.release()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    return None


def toggle_preview() -> None:
    if PREVIEW.state() == 'normal':
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()
        PREVIEW.deiconify()
    global camera
    if PREVIEW.state() == 'withdrawn':
        if camera and camera.isOpened():
            camera.release()
            camera = None


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    elif is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill='x')
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if not modules.globals.nsfw:
            from modules.predicter import predict_frame
            if predict_frame(temp_frame):
                quit()
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            temp_frame = frame_processor.process_frame(
                get_one_face(cv2.imread(modules.globals.source_path)),
                temp_frame
            )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)

def webcam_preview_loop(camera: cv2.VideoCapture, source_image: Any, frame_processors: List[ModuleType], virtual_cam: pyvirtualcam.Camera = None) -> bool:
    global preview_label, PREVIEW

    ret, frame = camera.read()
    if not ret:
        update_status(f"Error: Frame not received from camera.")
        return False

    temp_frame = frame.copy()

    if modules.globals.live_mirror:
        temp_frame = cv2.flip(temp_frame, 1) # horizontal flipping

    if modules.globals.live_resizable:
        temp_frame = fit_image_to_size(temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height())

    for frame_processor in frame_processors:
        temp_frame = frame_processor.process_frame(source_image, temp_frame)

    image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
    image = ImageOps.contain(image, (temp_frame.shape[1], temp_frame.shape[0]), Image.LANCZOS)
    image = ctk.CTkImage(image, size=image.size)
    preview_label.configure(image=image)
    if virtual_cam:
        virtual_cam.send(temp_frame)
        virtual_cam.sleep_until_next_frame()
    ROOT.update()

    if PREVIEW.state() == 'withdrawn':
        return False
    
    return True

def fit_image_to_size(image, width: int, height: int):
    if width is None and height is None:
      return image
    h, w, _ = image.shape
    ratio_h = 0.0
    ratio_w = 0.0
    if width > height:
        ratio_h = height / h
    else:
        ratio_w = width  / w
    ratio = max(ratio_w, ratio_h)
    new_size = (int(ratio * w), int(ratio * h))
    return cv2.resize(image, dsize=new_size)

def webcam_preview(camera_name: str, virtual_cam_output: bool):
    if modules.globals.source_path is None:
        return

    global preview_label, PREVIEW

    WIDTH = 960
    HEIGHT = 540
    FPS = 60

    # Select the camera by its name
    selected_camera = select_camera(camera_name)
    if selected_camera is None:
        update_status(f"No suitable camera found.")
        return

    # Use OpenCV's camera index for cross-platform compatibility
    camera_index = get_camera_index_by_name(camera_name)

    global camera
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        update_status(f"Error: Could not open camera {camera_name}")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, FPS)

    PREVIEW_MAX_WIDTH = WIDTH
    PREVIEW_MAX_HEIGHT = HEIGHT

    preview_label.configure(width=PREVIEW_DEFAULT_WIDTH, height=PREVIEW_DEFAULT_HEIGHT)
    PREVIEW.deiconify()

    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    source_image = get_one_face(cv2.imread(modules.globals.source_path))

    preview_running = True

    if virtual_cam_output:
        with pyvirtualcam.Camera(width=WIDTH, height=HEIGHT, fps=FPS, fmt=pyvirtualcam.PixelFormat.BGR) as virtual_cam:
            while preview_running:
                preview_running = webcam_preview_loop(camera, source_image, frame_processors, virtual_cam)

    while preview_running:
        preview_running = webcam_preview_loop(camera, source_image, frame_processors)

    if camera: camera.release()
    PREVIEW.withdraw()


def get_camera_index_by_name(camera_name: str) -> int:
    """Map camera name to index for OpenCV."""
    if platform.system() == 'Darwin':  # macOS-specific
        if "FaceTime" in camera_name:
            return 0  # Assuming FaceTime is at index 0
        elif "iPhone" in camera_name:
            return 1  # Assuming iPhone camera is at index 1
    elif platform.system() == 'Windows' or platform.system() == 'Linux':
        # Map camera name to index dynamically (OpenCV on these platforms usually starts with 0)
        return get_available_cameras().index(camera_name)
    return -1


def get_available_cameras():
    """Get available camera names (cross-platform)."""
    available_cameras = []
    if platform.system() == 'Darwin':  # macOS-specific
        devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
        
        for device in devices:
            if device.deviceType() == AVFoundation.AVCaptureDeviceTypeBuiltInWideAngleCamera:
                print(f"Found Built-In Camera: {device.localizedName()}")
                available_cameras.append(device.localizedName())
            elif device.deviceType() == "AVCaptureDeviceTypeExternal":
                print(f"Found External Camera: {device.localizedName()}")
                available_cameras.append(device.localizedName())
            elif device.deviceType() == "AVCaptureDeviceTypeContinuityCamera":
                print(f"Skipping Continuity Camera: {device.localizedName()}")
    elif platform.system() == 'Windows' or platform.system() == 'Linux':
        try:  
            devices = FilterGraph().get_input_devices()  
        except Exception as e:  
            # Use OpenCV to detect camera indexes
            index = 0
            devices = [] 
            while True:
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    break
                devices.append(f"Camera {index}")
                cap.release()
                index += 1

        available_cameras = devices
    return available_cameras
