import cv2
import time
import queue
import threading
from PIL import Image
import customtkinter as ctk

import modules.globals
from modules.gpu_processing import gpu_cvt_color, gpu_flip
from modules.face_analyser import get_one_face, get_many_faces, set_det_size, _LIVE_DET_SIZE, _DEFAULT_DET_SIZE
from modules.processors.frame.core import get_frame_processors_modules
from modules.video_capture import VideoCapturer


# DETECT_EVERY_N is kept for backward-compatibility with any external imports
# but is no longer used by the processing thread — detection now runs in its
# own dedicated thread.
DETECT_EVERY_N = 2


def _capture_thread_func(cap, capture_queue, stop_event):
    """Capture thread: reads frames from camera and puts them into the queue.
    Drops frames when the queue is full to avoid backpressure on the camera."""
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


def _detection_thread_func(latest_frame_holder, detection_result, detection_lock, stop_event):
    """Detection thread (producer): continuously reads the most recently
    captured raw frame and runs face detection on it, storing results in
    *detection_result* under *detection_lock*.

    latest_frame_holder is a one-element list [frame | None] written by the
    processing thread so the detection thread always works on the newest frame
    without queuing overhead.  The detection thread never touches Tkinter
    widgets — all UI updates go through ROOT.after() in the display loop.
    """
    while not stop_event.is_set():
        with detection_lock:
            frame = latest_frame_holder[0]

        if frame is None:
            time.sleep(0.005)
            continue

        if modules.globals.many_faces:
            many = get_many_faces(frame)
            with detection_lock:
                detection_result['target_face'] = None
                detection_result['many_faces'] = many
        else:
            face = get_one_face(frame)
            with detection_lock:
                detection_result['target_face'] = face
                detection_result['many_faces'] = None


def _processing_thread_func(capture_queue, processed_queue, stop_event,
                             latest_frame_holder, detection_result, detection_lock):
    """Processing thread (consumer): takes raw frames from capture_queue,
    reads the latest detection result from the shared detection_result dict,
    applies face swap/enhancement, and puts results into processed_queue.

    Face detection is no longer performed here — it runs concurrently in
    _detection_thread_func and the most recent result is consumed lock-free
    (under a brief lock copy) so the swap loop never blocks on detection."""
    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    source_image = None
    last_source_path = None
    prev_time = time.time()
    fps_update_interval = 0.5
    frame_count = 0
    fps = 0

    while not stop_event.is_set():
        try:
            frame = capture_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        temp_frame = frame

        if modules.globals.live_mirror:
            temp_frame = gpu_flip(temp_frame, 1)

        # Publish the mirrored frame for the detection thread to pick up
        with detection_lock:
            latest_frame_holder[0] = temp_frame

        if not modules.globals.map_faces:
            if modules.globals.source_path and modules.globals.source_path != last_source_path:
                last_source_path = modules.globals.source_path
                source_image = get_one_face(cv2.imread(modules.globals.source_path))

            # Read latest detection results — brief lock copy so we don't
            # block the detection thread longer than necessary
            with detection_lock:
                cached_target_face = detection_result.get('target_face')
                cached_many_faces = detection_result.get('many_faces')

            for frame_processor in frame_processors:
                if frame_processor.NAME == "DLC.FACE-ENHANCER":
                    if modules.globals.fp_ui.get("face_enhancer"):
                        temp_frame = frame_processor.process_frame(None, temp_frame)
                elif frame_processor.NAME == "DLC.FACE-ENHANCER-GPEN256":
                    if modules.globals.fp_ui.get("face_enhancer_gpen256"):
                        temp_frame = frame_processor.process_frame(None, temp_frame)
                elif frame_processor.NAME == "DLC.FACE-ENHANCER-GPEN512":
                    if modules.globals.fp_ui.get("face_enhancer_gpen512"):
                        temp_frame = frame_processor.process_frame(None, temp_frame)
                elif frame_processor.NAME == "DLC.FACE-SWAPPER":
                    # Use cached face positions to skip redundant detection
                    swapped_bboxes = []
                    if modules.globals.many_faces and cached_many_faces:
                        opacity = getattr(modules.globals, "opacity", 1.0)
                        result = temp_frame if opacity >= 1.0 else temp_frame.copy()
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
                    if modules.globals.fp_ui.get("face_enhancer"):
                        temp_frame = frame_processor.process_frame_v2(temp_frame)
                elif frame_processor.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                    fp_key = "face_enhancer_gpen256" if "256" in frame_processor.NAME else "face_enhancer_gpen512"
                    if modules.globals.fp_ui.get(fp_key):
                        temp_frame = frame_processor.process_frame_v2(temp_frame)
                else:
                    temp_frame = frame_processor.process_frame(None, temp_frame)

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
    from modules.ui import (
        preview_label, PREVIEW, ROOT,
        PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT,
        update_status, fit_image_to_size,
    )

    set_det_size(_LIVE_DET_SIZE)

    cap = VideoCapturer(camera_index)
    if not cap.start(PREVIEW_DEFAULT_WIDTH, PREVIEW_DEFAULT_HEIGHT, 60):
        set_det_size(_DEFAULT_DET_SIZE)
        update_status("Failed to start camera")
        return

    preview_label.configure(width=PREVIEW_DEFAULT_WIDTH, height=PREVIEW_DEFAULT_HEIGHT)
    PREVIEW.deiconify()

    # Queues for decoupling capture from processing and processing from display.
    # Small maxsize ensures we always work on recent frames and drop stale ones.
    capture_queue = queue.Queue(maxsize=2)
    processed_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    # Shared state for the producer-consumer detection pipeline.
    # latest_frame_holder[0] is the most recent raw frame for the detection
    # thread to consume; detection_result holds the last detected faces for
    # the processing thread to read.  Both are guarded by detection_lock.
    detection_lock = threading.Lock()
    latest_frame_holder = [None]  # one-element list so inner functions can rebind
    detection_result = {'target_face': None, 'many_faces': None}

    # Start capture thread
    cap_thread = threading.Thread(
        target=_capture_thread_func,
        args=(cap, capture_queue, stop_event),
        daemon=True,
    )
    cap_thread.start()

    # Start detection thread — runs face detection asynchronously on the
    # latest raw frame so the processing/swap thread never blocks on it.
    det_thread = threading.Thread(
        target=_detection_thread_func,
        args=(latest_frame_holder, detection_result, detection_lock, stop_event),
        daemon=True,
    )
    det_thread.start()

    # Start processing thread
    proc_thread = threading.Thread(
        target=_processing_thread_func,
        args=(capture_queue, processed_queue, stop_event,
              latest_frame_holder, detection_result, detection_lock),
        daemon=True,
    )
    proc_thread.start()

    def _cleanup():
        stop_event.set()
        cap_thread.join(timeout=2.0)
        det_thread.join(timeout=2.0)
        proc_thread.join(timeout=2.0)
        cap.release()
        set_det_size(_DEFAULT_DET_SIZE)
        PREVIEW.withdraw()

    def _display_next_frame():
        """Non-blocking display step — reschedules itself via ROOT.after()."""
        if stop_event.is_set() or PREVIEW.state() == "withdrawn":
            _cleanup()
            return


        try:
            temp_frame = processed_queue.get_nowait()
        except queue.Empty:
            ROOT.after(16, _display_next_frame)
            return

        if modules.globals.live_resizable:
            temp_frame = fit_image_to_size(
                temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height()
            )
        # live_resizable=False: display at native camera resolution

        image = gpu_cvt_color(temp_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)

        ROOT.after(16, _display_next_frame)

    # Kick off the non-blocking display loop
    ROOT.after(16, _display_next_frame)


def webcam_preview(root: ctk.CTk, camera_index: int):
    from modules.ui import POPUP_LIVE, update_status
    from modules.ui_mapper import create_source_target_popup_for_webcam

    if POPUP_LIVE is not None and POPUP_LIVE.winfo_exists():
        update_status("Source x Target Mapper is already open.")
        POPUP_LIVE.focus()
        return

    if not modules.globals.map_faces:
        if modules.globals.source_path is None:
            update_status("Please select a source image first")
            return
        create_webcam_preview(camera_index)
    else:
        with modules.globals.MAP_LOCK:
            modules.globals.source_target_map = []
        create_source_target_popup_for_webcam(
            root, modules.globals.source_target_map, camera_index
        )
