"""Tests for the cloud /live WebSocket protocol and the LiveSession frame loop.

No GPU or models needed: the face pipeline is replaced by stubs. Skipped when
the cloud extras (fastapi, httpx) aren't installed.
"""

import importlib
import json
import struct
import types
import unittest
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    HAS_DEPS = True
except ImportError:  # pragma: no cover
    HAS_DEPS = False

HEADER = struct.Struct(">I")


def _jpeg(color=(10, 20, 30), size=(48, 64)):
    frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    frame[:] = color
    return cv2.imencode(".jpg", frame)[1].tobytes()


class _StubSession:
    instances = []

    def __init__(self):
        self.configured = []
        _StubSession.instances.append(self)

    def configure(self, source_image=None, **options):
        from modules.live import decode_image

        if source_image is not None:
            decode_image(source_image)
        self.configured.append(options)

    def process(self, frame):
        return cv2.bitwise_not(frame)


@unittest.skipUnless(HAS_DEPS, "cloud extras not installed")
class LiveProtocolTests(unittest.TestCase):
    def setUp(self):
        import base64

        import modules.live

        _StubSession.instances.clear()
        self.source = base64.b64encode(_jpeg()).decode()
        patcher = patch.object(modules.live, "LiveSession", _StubSession)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _client(self, token=None):
        with patch.dict("os.environ", {"DLC_API_TOKEN": token} if token else {}, clear=False):
            import cloud.server
            server = importlib.reload(cloud.server)
        if not token:
            server.API_TOKEN = None
        return TestClient(server.app), server

    def _start(self, ws, options=None):
        ws.send_text(json.dumps({"type": "start", "source": self.source, "options": options or {}}))
        return json.loads(ws.receive_text())

    def test_frames_round_trip_in_order_with_ids(self):
        client, _ = self._client()
        with client.websocket_connect("/live") as ws:
            self.assertEqual(self._start(ws)["type"], "ready")
            for frame_id in (7, 8, 2**32 - 1):
                ws.send_bytes(HEADER.pack(frame_id) + _jpeg((200, 100, 50)))
                reply = ws.receive_bytes()
                self.assertEqual(HEADER.unpack(reply[:4])[0], frame_id)
                frame = cv2.imdecode(np.frombuffer(reply[4:], np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(frame.shape, (48, 64, 3))
                self.assertLess(abs(int(frame[0, 0, 0]) - 55), 8)  # inverted 200

    def test_update_reconfigures_and_bad_frame_reports_its_id(self):
        client, _ = self._client()
        with client.websocket_connect("/live") as ws:
            self._start(ws, {"frame_processors": ["face_swapper"]})
            ws.send_text(json.dumps({"type": "update", "options": {"many_faces": True}}))
            self.assertEqual(json.loads(ws.receive_text()), {"type": "updated"})
            self.assertEqual(_StubSession.instances[-1].configured[-1], {"many_faces": True})

            ws.send_bytes(HEADER.pack(42) + b"not a jpeg")
            error = json.loads(ws.receive_text())
            self.assertEqual((error["type"], error["frame_id"]), ("error", 42))

            ws.send_text(json.dumps({"type": "update", "options": {"bogus": 1}}))
            self.assertIn("Unknown option", json.loads(ws.receive_text())["message"])

    def test_invalid_start_is_rejected(self):
        client, _ = self._client()
        with client.websocket_connect("/live") as ws:
            ws.send_text(json.dumps({"type": "start", "source": "!!notbase64!!"}))
            self.assertEqual(json.loads(ws.receive_text())["type"], "error")
            with self.assertRaises(WebSocketDisconnect) as closed:
                ws.receive_text()
            self.assertEqual(closed.exception.code, 1003)

    def test_second_session_is_refused_while_busy(self):
        client, _ = self._client()
        with client.websocket_connect("/live") as first:
            self._start(first)
            with client.websocket_connect("/live") as second:
                self.assertIn("busy", json.loads(second.receive_text())["message"])
                with self.assertRaises(WebSocketDisconnect) as closed:
                    second.receive_text()
                self.assertEqual(closed.exception.code, 1013)
        # lock released once the first session ends
        with client.websocket_connect("/live") as third:
            self.assertEqual(self._start(third)["type"], "ready")

    def test_token_required_when_configured(self):
        client, _ = self._client(token="s3cret")
        with client.websocket_connect("/live?token=wrong") as ws:
            self.assertIn("token", json.loads(ws.receive_text())["message"])
        with client.websocket_connect("/live?token=s3cret") as ws:
            self.assertEqual(self._start(ws)["type"], "ready")
        self.assertEqual(client.post("/swap").status_code, 401)


@unittest.skipUnless(HAS_DEPS, "cloud extras not installed")
class LiveSessionFrameLoopTests(unittest.TestCase):
    """LiveSession.process must follow ui.py's live loop."""

    def _session(self, detect_every=1, many=False, enhancer_on=True):
        import modules.globals
        from modules.live import LiveSession

        calls = []
        swapper = types.SimpleNamespace(
            NAME="DLC.FACE-SWAPPER",
            swap_face=lambda src, face, frame: calls.append(("swap", face)) or frame + 1,
            apply_post_processing=lambda frame, boxes: calls.append(("post", len(boxes))) or frame,
        )
        enhancer = types.SimpleNamespace(
            NAME="DLC.FACE-ENHANCER",
            process_frame=lambda src, frame, detected_faces=None: calls.append(("enhance", detected_faces)) or frame,
        )
        session = LiveSession()
        session.source_face = "SRC"
        session.processors = [swapper, enhancer]
        session.detect_every = detect_every
        modules.globals.many_faces = many
        modules.globals.mouth_mask = False
        modules.globals.fp_ui["face_enhancer"] = enhancer_on
        return session, calls

    def _face(self, name):
        return types.SimpleNamespace(bbox=np.array([0, 0, 1, 1]), name=name)

    def test_detects_every_n_frames_and_reuses_cached_face(self):
        session, calls = self._session(detect_every=3)
        detections = []
        with patch("modules.face_analyser.detect_one_face_fast",
                   side_effect=lambda f: detections.append(1) or self._face("a")):
            for _ in range(7):
                session.process(np.zeros((4, 4, 3), np.uint8))
        self.assertEqual(len(detections), 3)  # frames 0, 3, 6
        self.assertEqual(sum(1 for c in calls if c[0] == "swap"), 7)

    def test_many_faces_and_enhancer_toggle(self):
        session, calls = self._session(many=True, enhancer_on=False)
        faces = [self._face("a"), self._face("b")]
        with patch("modules.face_analyser.detect_many_faces_fast", return_value=faces):
            session.process(np.zeros((4, 4, 3), np.uint8))
        self.assertEqual([c for c in calls if c[0] == "swap"], [("swap", faces[0]), ("swap", faces[1])])
        self.assertIn(("post", 2), calls)
        self.assertFalse(any(c[0] == "enhance" for c in calls))

    def test_no_face_passes_frame_through(self):
        session, calls = self._session()
        frame = np.full((4, 4, 3), 9, np.uint8)
        with patch("modules.face_analyser.detect_one_face_fast", return_value=None):
            out = session.process(frame)
        self.assertTrue((out == frame).all())
        self.assertEqual([c[0] for c in calls], ["post", "enhance"])


if __name__ == "__main__":
    unittest.main()
