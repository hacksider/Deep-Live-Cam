"""
Deep-Live-Cam Web Interface
Flask-based web UI for face swapping on images + live webcam streaming.
"""

import os
import sys
import base64
import logging
import threading
import queue
import time
import urllib.request

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '4'

import cv2
import numpy as np

try:
    from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
except ImportError:
    print("Flask not found. Install it with:  pip install flask")
    sys.exit(1)

import modules.globals as g
from modules.face_analyser import get_one_face, get_many_faces, detect_one_face_fast, detect_many_faces_fast
from modules.processors.frame.face_swapper import get_face_swapper, swap_face

# Must be set BEFORE any module calls update_status(), otherwise it tries
# to call ui.status_label.configure() which is None in headless/web mode.
g.headless = True

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder="web_static", static_url_path="/static")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DLC-Web")

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------
g.execution_providers = ["CPUExecutionProvider"]
g.frame_processors = ["face_swapper"]
g.many_faces = False
g.map_faces = False
g.mouth_mask = False
g.poisson_blend = False
g.color_correction = False
g.opacity = 1.0
g.sharpness = 0.0
g.mouth_mask_size = 0.0
g.fp_ui = {
    "face_enhancer": False,
    "face_enhancer_gpen256": False,
    "face_enhancer_gpen512": False,
}

# Pre-load face swapper model
_model_ready = threading.Event()

def _preload_model():
    try:
        logger.info("Pre-loading face swapper model…")
        get_face_swapper()
        logger.info("Model ready.")
    except Exception as exc:
        logger.warning(f"Model pre-load failed (will retry on first request): {exc}")
    finally:
        _model_ready.set()

threading.Thread(target=_preload_model, daemon=True).start()

# ---------------------------------------------------------------------------
# Live cam state  (one session at a time)
# ---------------------------------------------------------------------------
_live_lock = threading.Lock()

class LiveSession:
    def __init__(self):
        self.running = False
        self.stop_event = threading.Event()
        self.source_face = None          # insightface Face object
        self.many_faces = False
        # live settings (applied per-frame)
        self.opacity = 1.0
        self.sharpness = 0.0
        self.mouth_mask_size = 0.0
        self.enhancer = "none"
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.out_queue: queue.Queue = queue.Queue(maxsize=4)  # shared output queue
        self.cap_thread: threading.Thread = None
        self.proc_thread: threading.Thread = None
        self.cap: cv2.VideoCapture = None
        # cached face detection
        self._cached_face = None
        self._cached_many = None
        self._det_counter = 0
        self._det_interval = 1           # detect every frame (safe default; tune up for perf)

_live: LiveSession = LiveSession()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(data_url_or_bytes) -> np.ndarray:
    if isinstance(data_url_or_bytes, str):
        if "," in data_url_or_bytes:
            data_url_or_bytes = data_url_or_bytes.split(",", 1)[1]
        raw = base64.b64decode(data_url_or_bytes)
    else:
        raw = data_url_or_bytes
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_image(img: np.ndarray, fmt: str = ".jpg", quality: int = 90) -> str:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if fmt == ".jpg" else []
    ok, buf = cv2.imencode(fmt, img, params)
    if not ok:
        raise ValueError("Failed to encode result image")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


def _apply_enhancer(frame: np.ndarray, enhancer: str) -> np.ndarray:
    if enhancer == "none":
        return frame
    key_map = {
        "gfpgan": "face_enhancer",
        "gpen512": "face_enhancer_gpen512",
        "gpen256": "face_enhancer_gpen256",
    }
    processor_name = key_map.get(enhancer)
    if processor_name is None:
        return frame
    for k in g.fp_ui:
        g.fp_ui[k] = False
    g.fp_ui[processor_name] = True
    g.frame_processors = ["face_swapper", processor_name]
    try:
        from modules.processors.frame.core import get_frame_processors_modules
        processors = get_frame_processors_modules(g.frame_processors)
        for proc in processors:
            if proc.NAME != "DLC.FACE-SWAPPER":
                frame = proc.process_frame(None, frame)
    except Exception as exc:
        logger.warning(f"Enhancer '{enhancer}' failed: {exc}")
    finally:
        g.fp_ui[processor_name] = False
        g.frame_processors = ["face_swapper"]
    return frame


def _list_cameras() -> list:
    """Return list of {index, name} for available cameras."""
    cameras = []
    for i in range(6):
        # Try MSMF first (more reliable on Windows), then CAP_ANY
        for backend in [cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    cameras.append({"index": i, "name": f"Camera {i}"})
                    break
            else:
                cap.release()
    return cameras


# ---------------------------------------------------------------------------
# Live cam background threads
# ---------------------------------------------------------------------------

def _cap_thread(session: LiveSession):
    """Capture thread: read frames from camera → capture queue."""
    while not session.stop_event.is_set():
        ret, frame = session.cap.read()
        if not ret:
            session.stop_event.set()
            break
        try:
            session.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                session.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                session.frame_queue.put_nowait(frame)
            except queue.Full:
                pass


def _proc_thread(session: LiveSession):
    """Processing thread: swap face on each frame → shared out_queue."""
    _log_interval = 30   # print a status line every N frames
    _frame_n = 0
    logger.info("[proc_thread] started")
    while not session.stop_event.is_set():
        try:
            frame = session.frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        _frame_n += 1
        result = frame.copy()

        if session.source_face is None:
            if _frame_n % _log_interval == 0:
                logger.info(f"[proc_thread] frame {_frame_n}: no source_face yet, passing through")
        else:
            session._det_counter += 1
            if session._det_counter % session._det_interval == 0:
                if session.many_faces:
                    session._cached_many = get_many_faces(frame)
                    session._cached_face = None
                    if _frame_n % _log_interval == 0:
                        logger.info(f"[proc_thread] frame {_frame_n}: many_faces detected={len(session._cached_many) if session._cached_many else 0}")
                else:
                    session._cached_face = get_one_face(frame)
                    session._cached_many = None
                    if _frame_n % _log_interval == 0:
                        logger.info(f"[proc_thread] frame {_frame_n}: one_face detected={session._cached_face is not None}")

            # apply globals for opacity/sharpness/mouth_mask
            g.opacity = session.opacity
            g.sharpness = session.sharpness
            g.mouth_mask_size = session.mouth_mask_size
            g.mouth_mask = session.mouth_mask_size > 0

            try:
                if session.many_faces and session._cached_many:
                    for tf in session._cached_many:
                        result = swap_face(session.source_face, tf, result)
                    if _frame_n % _log_interval == 0:
                        logger.info(f"[proc_thread] frame {_frame_n}: swapped {len(session._cached_many)} faces")
                elif session._cached_face is not None:
                    result = swap_face(session.source_face, session._cached_face, result)
                    if _frame_n % _log_interval == 0:
                        logger.info(f"[proc_thread] frame {_frame_n}: swapped 1 face OK")
                else:
                    if _frame_n % _log_interval == 0:
                        logger.info(f"[proc_thread] frame {_frame_n}: source_face set but no target face cached yet")
            except Exception as exc:
                logger.warning(f"[proc_thread] frame {_frame_n}: swap error: {exc}")

        # Encode to JPEG
        ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            continue

        jpg = buf.tobytes()
        try:
            session.out_queue.put_nowait(jpg)
        except queue.Full:
            try:
                session.out_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                session.out_queue.put_nowait(jpg)
            except queue.Full:
                pass


def _stop_live():
    """Stop any running live session."""
    global _live
    _live.stop_event.set()
    _live.running = False
    if _live.cap_thread and _live.cap_thread.is_alive():
        _live.cap_thread.join(timeout=2.0)
    if _live.proc_thread and _live.proc_thread.is_alive():
        _live.proc_thread.join(timeout=2.0)
    if _live.cap:
        _live.cap.release()
        _live.cap = None
    _live = LiveSession()


# ---------------------------------------------------------------------------
# Routes — static
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("web_static", "index.html")


# ---------------------------------------------------------------------------
# Routes — image swap (existing)
# ---------------------------------------------------------------------------

@app.route("/api/random_face", methods=["GET"])
def random_face():
    """Fetch a random face from thispersondoesnotexist.com.
    Retries up to 5 times to ensure the returned image actually contains
    a detectable face (GAN images occasionally produce undetectable results).
    """
    last_error = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                "https://thispersondoesnotexist.com/",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                last_error = "Failed to decode image"
                continue
            # Verify a face is actually detectable before returning
            face = get_one_face(img)
            if face is None:
                logger.info(f"[random_face] attempt {attempt+1}: no face detected, retrying…")
                continue
            logger.info(f"[random_face] attempt {attempt+1}: face detected (score={face.det_score:.3f})")
            return jsonify({"image": _encode_image(img, ".jpg", 85)})
        except Exception as exc:
            last_error = str(exc)
            logger.warning(f"[random_face] attempt {attempt+1} error: {exc}")
    return jsonify({"error": f"Could not get a usable random face after 5 attempts: {last_error}"}), 500


@app.route("/api/swap", methods=["POST"])
def swap():
    data = request.get_json(force=True)
    if not data.get("source"):
        return jsonify({"error": "Missing source image"}), 400
    if not data.get("target"):
        return jsonify({"error": "Missing target image"}), 400

    opacity          = max(0.0, min(1.0,   float(data.get("opacity", 1.0))))
    sharpness        = max(0.0, min(5.0,   float(data.get("sharpness", 0.0))))
    mouth_mask_size  = max(0.0, min(100.0, float(data.get("mouth_mask_size", 0.0))))
    many_faces       = bool(data.get("many_faces", False))
    enhancer         = str(data.get("enhancer", "none")).lower()

    g.opacity = opacity
    g.sharpness = sharpness
    g.mouth_mask_size = mouth_mask_size
    g.mouth_mask = mouth_mask_size > 0
    g.many_faces = many_faces

    try:
        source_img = _decode_image(data["source"])
        target_img = _decode_image(data["target"])
    except Exception as exc:
        return jsonify({"error": f"Image decode error: {exc}"}), 400

    if source_img is None:
        return jsonify({"error": "Could not decode source image"}), 400
    if target_img is None:
        return jsonify({"error": "Could not decode target image"}), 400

    source_face = get_one_face(source_img)
    if source_face is None:
        # Retry with lower threshold
        from modules.face_analyser import get_face_analyser
        fa = get_face_analyser()
        orig_thresh = fa.det_model.det_thresh
        fa.det_model.det_thresh = 0.3
        source_face = get_one_face(source_img)
        fa.det_model.det_thresh = orig_thresh
    if source_face is None:
        return jsonify({"error": "No face detected in source image."}), 422

    try:
        result = target_img.copy()
        if many_faces:
            target_faces = get_many_faces(target_img)
            if not target_faces:
                return jsonify({"error": "No faces detected in target image"}), 422
            for tf in target_faces:
                result = swap_face(source_face, tf, result)
        else:
            target_face = get_one_face(target_img)
            if target_face is None:
                return jsonify({"error": "No face detected in target image"}), 422
            result = swap_face(source_face, target_face, result)
    except Exception as exc:
        logger.exception("Swap failed")
        return jsonify({"error": f"Swap failed: {exc}"}), 500

    try:
        result = _apply_enhancer(result, enhancer)
    except Exception as exc:
        logger.warning(f"Enhancer failed: {exc}")

    try:
        return jsonify({"result": _encode_image(result, ".jpg", 92)})
    except Exception as exc:
        return jsonify({"error": f"Encoding result failed: {exc}"}), 500


# ---------------------------------------------------------------------------
# Routes — Live Cam
# ---------------------------------------------------------------------------

@app.route("/api/cameras", methods=["GET"])
def cameras():
    """Return list of available cameras."""
    return jsonify({"cameras": _list_cameras()})


@app.route("/api/live/start", methods=["POST"])
def live_start():
    """
    POST JSON:
    {
        "camera_index": 0,
        "source": "<base64 data-URL>",   // optional – can be set later via /api/live/source
        "many_faces": false
    }
    """
    global _live
    data = request.get_json(force=True) or {}

    with _live_lock:
        # Stop any existing session first
        if _live.running:
            _stop_live()

        camera_index = int(data.get("camera_index", 0))
        many_faces   = bool(data.get("many_faces", False))

        # Open camera - try MSMF first, then CAP_ANY, then no backend hint
        cap = None
        for backend in [cv2.CAP_MSMF, cv2.CAP_ANY, -1]:
            try:
                if backend == -1:
                    cap = cv2.VideoCapture(camera_index)
                else:
                    cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        break
                cap.release()
                cap = None
            except Exception:
                if cap:
                    cap.release()
                cap = None

        if cap is None or not cap.isOpened():
            return jsonify({"error": f"Cannot open camera {camera_index}"}), 500

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        _live.cap = cap
        _live.many_faces = many_faces
        _live.opacity = float(data.get("opacity", 1.0))
        _live.sharpness = float(data.get("sharpness", 0.0))
        _live.mouth_mask_size = float(data.get("mouth_mask_size", 0.0))
        _live.enhancer = str(data.get("enhancer", "none")).lower()
        _live.running = True
        _live.stop_event.clear()

        # Set source face if provided
        if data.get("source"):
            try:
                src_img = _decode_image(data["source"])
                face = get_one_face(src_img)
                if face is None:
                    logger.warning("No face detected in source image at start, will retry when source is set via /api/live/source")
                else:
                    _live.source_face = face
            except Exception as exc:
                logger.warning(f"Source decode error at start: {exc}")

        # Start capture + processing threads
        _live.cap_thread = threading.Thread(
            target=_cap_thread, args=(_live,), daemon=True
        )
        _live.cap_thread.start()

        _live.proc_thread = threading.Thread(
            target=_proc_thread, args=(_live,), daemon=True
        )
        _live.proc_thread.start()

    return jsonify({"status": "started", "camera_index": camera_index})


@app.route("/api/live/source", methods=["POST"])
def live_set_source():
    """Update the source face while live session is running."""
    global _live
    data = request.get_json(force=True) or {}
    if not data.get("source"):
        return jsonify({"error": "Missing source"}), 400
    try:
        src_img = _decode_image(data["source"])
        if src_img is None:
            return jsonify({"error": "Image decode failed (result is None). Check image format."}), 400
        h, w = src_img.shape[:2]
        logger.info(f"[live/source] decoded image: {w}x{h}, dtype={src_img.dtype}, channels={src_img.shape[2] if src_img.ndim==3 else 1}")

        # Try with original image first
        face = get_one_face(src_img)

        # If not found, try resizing to 640x640 (det_size) — sometimes very small or very large images fail
        if face is None and (w < 100 or h < 100 or w > 2000 or h > 2000):
            scale = 640 / max(w, h)
            resized = cv2.resize(src_img, (int(w*scale), int(h*scale)))
            logger.info(f"[live/source] retrying with resized image: {resized.shape[1]}x{resized.shape[0]}")
            face = get_one_face(resized)

        # If still not found, try lowering detection threshold temporarily
        if face is None:
            from modules.face_analyser import get_face_analyser
            fa = get_face_analyser()
            orig_thresh = fa.det_model.det_thresh
            fa.det_model.det_thresh = 0.3
            logger.info(f"[live/source] retrying with lower det_thresh=0.3")
            face = get_one_face(src_img)
            fa.det_model.det_thresh = orig_thresh

        if face is None:
            logger.warning(f"[live/source] No face detected in {w}x{h} image after all retries")
            return jsonify({"error": "No face detected in source image. Please use a clear frontal face photo."}), 422

        logger.info(f"[live/source] face detected, det_score={face.det_score:.3f}")
        _live.source_face = face
        _live.many_faces = bool(data.get("many_faces", _live.many_faces))
        return jsonify({"status": "source updated"})
    except Exception as exc:
        logger.exception("[live/source] exception")
        return jsonify({"error": str(exc)}), 400


@app.route("/api/live/settings", methods=["POST"])
def live_settings():
    """Update live processing settings on the fly (no restart needed)."""
    global _live
    data = request.get_json(force=True) or {}
    if "opacity" in data:
        _live.opacity = max(0.0, min(1.0, float(data["opacity"])))
    if "sharpness" in data:
        _live.sharpness = max(0.0, min(5.0, float(data["sharpness"])))
    if "mouth_mask_size" in data:
        _live.mouth_mask_size = max(0.0, min(100.0, float(data["mouth_mask_size"])))
    if "many_faces" in data:
        _live.many_faces = bool(data["many_faces"])
        # reset cached faces when mode changes
        _live._cached_face = None
        _live._cached_many = None
    if "enhancer" in data:
        _live.enhancer = str(data["enhancer"]).lower()
    return jsonify({"status": "settings updated"})


@app.route("/api/live/stop", methods=["POST"])
def live_stop():
    global _live
    with _live_lock:
        _stop_live()
    return jsonify({"status": "stopped"})


@app.route("/api/live/stream")
def live_stream():
    """MJPEG stream — reads from the shared out_queue on the live session."""
    global _live

    if not _live.running:
        return jsonify({"error": "No live session running"}), 400

    def generate():
        try:
            while _live.running and not _live.stop_event.is_set():
                try:
                    jpg_bytes = _live.out_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    jpg_bytes +
                    b"\r\n"
                )
        except GeneratorExit:
            pass

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Deep-Live-Cam Web Interface")
    print("  Open http://127.0.0.1:7860 in your browser")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
