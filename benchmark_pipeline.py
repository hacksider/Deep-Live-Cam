"""Standalone pipeline benchmark — no UI required.

Captures 200 frames from the webcam and runs the full face swap pipeline,
printing per-stage timing and effective FPS.
"""
import os, sys, time, cv2, numpy as np, queue, threading

# PATH fix for cuDNN (Windows only)
if sys.platform == "win32":
    _sp = os.path.join(sys.prefix, "Lib", "site-packages")
    _torch_lib = os.path.join(_sp, "torch", "lib")
    if os.path.isdir(_torch_lib):
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ["PATH"]

import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from modules.processors.frame.face_swapper import _fast_paste_back
from modules import platform_info

platform_info.print_banner()

# Pick providers based on what's actually available on this machine.
if platform_info.HAS_CUDA_PROVIDER:
    _providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
elif platform_info.HAS_COREML_PROVIDER:
    _providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
else:
    _providers = ["CPUExecutionProvider"]

# --- Init models (same as the app) ---
print(f"Loading models with providers={_providers}...")
fa = FaceAnalysis(
    name="buffalo_l",
    providers=_providers,
    allowed_modules=["detection", "recognition", "landmark_2d_106"],
)
fa.prepare(ctx_id=0, det_size=(640, 640))
swap_model = insightface.model_zoo.get_model(
    "models/inswapper_128.onnx",
    providers=_providers,
)
face_size = swap_model.input_size[0]
aimg_dummy = np.empty((face_size, face_size, 3), dtype=np.uint8)

# --- Camera setup ---
# Windows: DirectShow explicit for MJPEG 1080p60 support.
# macOS/Linux: default backend (AVFoundation / V4L2).
print("Opening camera at 1080p60 MJPEG...")
if sys.platform == "win32":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
time.sleep(0.5)

# Warmup + get source face
for _ in range(15):
    cap.read()
ret, src_frame = cap.read()
faces = fa.get(src_frame)
if not faces:
    print("ERROR: No face detected in warmup frame")
    cap.release()
    sys.exit(1)
source_face = faces[0]
print(f"Source face acquired. Frame: {src_frame.shape}")

# --- Capture thread (same as app) ---
capture_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()

def capture_thread():
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            capture_queue.put_nowait(frame)
        except queue.Full:
            try: capture_queue.get_nowait()
            except queue.Empty: pass
            try: capture_queue.put_nowait(frame)
            except queue.Full: pass

cap_t = threading.Thread(target=capture_thread, daemon=True)
cap_t.start()

# --- Warmup processing ---
print("Warming up pipeline...")
for _ in range(20):
    try:
        frame = capture_queue.get(timeout=0.1)
    except queue.Empty:
        continue
    f = frame.copy()
    det_faces = fa.get(f)
    if det_faces:
        tgt = min(det_faces, key=lambda x: x.bbox[0])
        bgr_fake, M = swap_model.get(f, tgt, source_face, paste_back=False)
        _fast_paste_back(f, bgr_fake, aimg_dummy, M)

# --- Benchmark ---
N = 200
print(f"\nBenchmarking {N} frames...")

t_queue, t_det, t_onnx, t_paste, t_copy, t_cvt, t_total = [], [], [], [], [], [], []
det_count = 0
cached_face = None

for i in range(N):
    tt = time.perf_counter()

    t0 = time.perf_counter()
    try:
        frame = capture_queue.get(timeout=0.1)
    except queue.Empty:
        continue
    t_queue.append((time.perf_counter() - t0) * 1000)

    # Detection every 3rd frame — det-only (no landmark/recognition)
    det_count += 1
    if det_count % 3 == 0:
        t0 = time.perf_counter()
        from insightface.app.common import Face as _Face
        bboxes, kpss = fa.det_model.detect(frame, max_num=0, metric='default')
        if bboxes.shape[0] > 0:
            idx = int(bboxes[:, 0].argmin())
            cached_face = _Face(bbox=bboxes[idx, :4], kps=kpss[idx], det_score=bboxes[idx, 4])
        t_det.append((time.perf_counter() - t0) * 1000)

    if cached_face is not None:
        # No frame.copy() — _fast_paste_back writes in-place, we own the frame
        t0 = time.perf_counter()
        bgr_fake, M = swap_model.get(frame, cached_face, source_face, paste_back=False)
        t_onnx.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        result = _fast_paste_back(frame, bgr_fake, aimg_dummy, M)
        t_paste.append((time.perf_counter() - t0) * 1000)

        # Display prep — resize then flip (no cvtColor needed)
        t0 = time.perf_counter()
        small = cv2.resize(result, (640, 360))
        _ = small[:, :, ::-1]  # BGR→RGB zero-copy
        t_cvt.append((time.perf_counter() - t0) * 1000)

    t_total.append((time.perf_counter() - tt) * 1000)

stop_event.set()
cap.release()

# --- Results ---
def s(name, arr):
    if not arr:
        return
    avg = sum(arr) / len(arr)
    print(f"  {name:25s}: avg={avg:6.1f}ms  min={min(arr):5.1f}ms  max={max(arr):6.1f}ms  n={len(arr)}")

print(f"\n{'='*55}")
print(f"  1080p Pipeline Benchmark ({len(t_total)} frames)")
print(f"{'='*55}")
s("queue.get (wait for cam)", t_queue)
s("detection (fa.get)", t_det)
s("frame.copy()", t_copy)
s("ONNX swap", t_onnx)
s("_fast_paste_back", t_paste)
s("cvtColor BGR->RGB", t_cvt)
s("TOTAL per frame", t_total)

avg_total = sum(t_total) / len(t_total)
avg_queue = sum(t_queue) / len(t_queue)
print(f"\n  Effective FPS:          {1000/avg_total:.1f}")
print(f"  FPS (excl. cam wait):   {1000/(avg_total - avg_queue):.1f}")
print(f"{'='*55}")
