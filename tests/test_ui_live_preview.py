"""Regression tests for live-preview teardown and source handling in modules.ui.

Runs the real widgets and worker threads under Qt's offscreen platform. No
model is ever loaded: face detection and the frame-processor list are stubbed,
so the tests exercise control flow only and stay fast.
"""
import os
import queue
import sys
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import modules.globals  # noqa: E402
import modules.ui as ui  # noqa: E402

_APP = None


def setUpModule():
    global _APP
    _APP = QApplication.instance() or QApplication([])


def _frame():
    return np.zeros((16, 16, 3), dtype=np.uint8)


class _StubCapture:
    """Stands in for VideoCapturer. read() is deliberately slow so that
    shutdown has to wait for a worker that is mid-operation."""

    read_delay = 0.3

    def __init__(self, device_index=0):
        self.actual_width = 16
        self.actual_height = 16
        self.actual_fps = 30.0
        self.released = False

    def start(self, width=0, height=0, fps=0):
        return True

    def read(self):
        time.sleep(self.read_delay)
        return True, _frame()

    def release(self):
        self.released = True


class CloseEventShutdownTest(unittest.TestCase):
    """A worker that outlives the grace period must still be joined.

    Parking it instead only defers Qt's fatal 'QThread: Destroyed while thread
    is still running' to interpreter exit.
    """

    def test_close_event_leaves_no_running_worker(self):
        modules.globals.source_path = None
        modules.globals.map_faces = False
        with patch.object(ui, "VideoCapturer", _StubCapture), \
             patch.object(ui, "get_frame_processors_modules", lambda *_: []), \
             patch.object(ui, "detect_one_face_fast", lambda *_: None), \
             patch.object(ui, "detect_many_faces_fast", lambda *_: None), \
             patch.object(ui, "WORKER_SHUTDOWN_GRACE_MS", 50):
            window = ui.WebcamPreviewWindow(0)
            self.addCleanup(window.deleteLater)
            time.sleep(0.2)
            window.close()

            for name in ("_capture_worker", "_processing_worker"):
                worker = getattr(window, name)
                self.assertTrue(
                    worker.isFinished(),
                    f"{name} was still running when closeEvent returned",
                )
            self.assertTrue(window._cap.released)
            self.assertFalse(
                hasattr(ui, "_STRAY_WORKERS"),
                "workers must be joined, not parked: a parked QThread is still "
                "destroyed at interpreter exit, which aborts the process",
            )

    def test_close_event_survives_failed_camera_start(self):
        """__init__ returns early when the camera will not open, so closeEvent
        runs against a half-built window."""
        class _DeadCapture(_StubCapture):
            def start(self, width=0, height=0, fps=0):
                return False

        with patch.object(ui, "VideoCapturer", _DeadCapture):
            window = ui.WebcamPreviewWindow(0)
            self.addCleanup(window.deleteLater)
            window.close()  # must not raise AttributeError


class ShutdownExitCodeTest(unittest.TestCase):
    """The end-to-end property: a worker that outlives the grace period must
    not take the process down on the way out."""

    CHILD = r"""
import os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, {root!r})
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import modules.globals, modules.ui as ui

class Slow:
    def __init__(self, i=0):
        self.actual_width = 16; self.actual_height = 16; self.actual_fps = 30.0
    def start(self, *a, **k): return True
    def read(self):
        time.sleep(0.4)
        return True, np.zeros((16, 16, 3), dtype=np.uint8)
    def release(self): pass

ui.VideoCapturer = Slow
ui.get_frame_processors_modules = lambda *_: []
ui.detect_one_face_fast = lambda *_: None
ui.detect_many_faces_fast = lambda *_: None
ui.WORKER_SHUTDOWN_GRACE_MS = 50
modules.globals.source_path = None
modules.globals.map_faces = False

app = QApplication([])
win = ui.WebcamPreviewWindow(0)
win.show()
def finish():
    win.close()
    app.quit()
QTimer.singleShot(300, finish)
app.exec()
"""

    def test_process_exits_cleanly(self):
        import subprocess
        import tempfile
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(self.CHILD.format(root=root))
            child = handle.name
        self.addCleanup(lambda: os.path.exists(child) and os.remove(child))

        result = subprocess.run(
            [sys.executable, child], capture_output=True, text=True, timeout=120
        )
        output = result.stdout + result.stderr
        self.assertNotIn("QThread: Destroyed", output, output[-2000:])
        self.assertEqual(result.returncode, 0, output[-2000:])


class SourceReloadTest(unittest.TestCase):
    """Picking an unreadable source must not disable swapping for a source
    that does work: A -> unreadable B -> A has to reload A."""

    def test_switching_back_to_a_working_source_reloads_it(self):
        reads = []
        lock = threading.Lock()

        def fake_imread(path, *_a, **_kw):
            with lock:
                reads.append(path)
            return None if path == "B" else _frame()

        capture_q = queue.Queue(maxsize=2)
        processed_q = queue.Queue(maxsize=2)
        stop = threading.Event()

        modules.globals.map_faces = False
        modules.globals.source_path = "A"

        def count(path):
            with lock:
                return reads.count(path)

        def feed_until(predicate, timeout=5.0):
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    capture_q.put_nowait(_frame())
                except queue.Full:
                    try:
                        processed_q.get_nowait()
                    except queue.Empty:
                        pass
                if predicate():
                    return True
                time.sleep(0.01)
            return False

        with patch.object(ui, "imread_unicode", fake_imread), \
             patch.object(ui, "get_one_face", lambda *_: object()), \
             patch.object(ui, "get_frame_processors_modules", lambda *_: []), \
             patch.object(ui, "detect_one_face_fast", lambda *_: None), \
             patch.object(ui, "detect_many_faces_fast", lambda *_: None), \
             patch.object(ui, "update_status", lambda *_: None):
            worker = ui._ProcessingWorker(capture_q, processed_q, stop, 30.0)
            worker.start()
            try:
                self.assertTrue(feed_until(lambda: count("A") >= 1), "A never loaded")
                modules.globals.source_path = "B"
                self.assertTrue(feed_until(lambda: count("B") >= 1), "B never tried")
                modules.globals.source_path = "A"
                self.assertTrue(
                    feed_until(lambda: count("A") >= 2),
                    "A was not reloaded after an unreadable source; the cached "
                    "path kept swapping disabled",
                )
            finally:
                stop.set()
                worker.wait(5000)


class RandomFaceDownloadTest(unittest.TestCase):
    """A failed fetch must not damage the image already in use."""

    class _Label:
        def setPixmap(self, _pixmap): pass
        def setText(self, _text): pass

    class _Response:
        def __init__(self, content, content_type="image/jpeg"):
            self.content = content
            self.headers = {"content-type": content_type}

        def raise_for_status(self):
            return None

    def _good_jpeg(self):
        import cv2
        ok, buf = cv2.imencode(".jpg", np.full((32, 32, 3), 128, dtype=np.uint8))
        self.assertTrue(ok)
        return buf.tobytes()

    def test_failed_download_leaves_previous_image_intact(self):
        import tempfile
        target = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
        staging = f"{target}.part"
        good = self._good_jpeg()
        with open(target, "wb") as handle:
            handle.write(good)
        self.addCleanup(lambda: os.path.exists(target) and os.remove(target))
        modules.globals.source_path = target

        stub = type("Stub", (), {"source_label": self._Label()})()
        # Claims to be an image, but the bytes do not decode.
        bad = self._Response(b"<!DOCTYPE html><html>not an image</html>")
        with patch.object(ui.requests, "get", lambda *_a, **_kw: bad):
            ui.MainWindow._on_random_face(stub)

        with open(target, "rb") as handle:
            self.assertEqual(handle.read(), good, "the image in use was overwritten")
        self.assertEqual(modules.globals.source_path, target)
        self.assertFalse(os.path.exists(staging), "staging file was left behind")

    def test_successful_download_replaces_the_image(self):
        import tempfile
        target = os.path.join(tempfile.gettempdir(), "deep_live_cam_random_face.jpg")
        self.addCleanup(lambda: os.path.exists(target) and os.remove(target))
        fresh = self._good_jpeg()

        stub = type("Stub", (), {"source_label": self._Label()})()
        with patch.object(ui.requests, "get", lambda *_a, **_kw: self._Response(fresh)):
            ui.MainWindow._on_random_face(stub)

        self.assertEqual(modules.globals.source_path, target)
        with open(target, "rb") as handle:
            self.assertEqual(handle.read(), fresh)


if __name__ == "__main__":
    unittest.main()
