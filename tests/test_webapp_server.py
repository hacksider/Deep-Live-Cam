import base64
import unittest

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from webapp.engine import MAX_SESSIONS, LiveEngine
from webapp.server import create_app


def _engine():
    def decode_image(data: bytes):
        if data == b"noface":
            return data
        return data

    def get_one_face(image: bytes):
        if image == b"noface":
            return None
        return image

    def detect_one_face(frame: bytes):
        return b"target"

    def swap_face(source, target, frame):
        return b"swapped"

    def encode_jpeg(frame: bytes) -> bytes:
        return b"OUT"

    return LiveEngine(
        get_one_face=get_one_face,
        detect_one_face=detect_one_face,
        swap_face=swap_face,
        decode_image=decode_image,
        encode_jpeg=encode_jpeg,
    )


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.engine = _engine()
        self._client_cm = TestClient(create_app(self.engine, "secret"))
        self.client = self._client_cm.__enter__()

    def tearDown(self):
        self._client_cm.__exit__(None, None, None)

    def test_wrong_password_is_401_and_sets_no_cookie(self):
        response = self.client.post("/login", json={"password": "nope"})
        self.assertEqual(response.status_code, 401)
        self.assertNotIn("dlc_session", response.cookies)

    def test_right_password_sets_session_cookie_without_max_age(self):
        response = self.client.post("/login", json={"password": "secret"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("dlc_session", response.cookies)
        set_cookie = response.headers["set-cookie"].lower()
        self.assertIn("httponly", set_cookie)
        self.assertIn("samesite=lax", set_cookie)
        self.assertNotIn("max-age", set_cookie)
        self.assertEqual(self.client.get("/session").status_code, 204)

    def test_session_without_cookie_is_401(self):
        self.assertEqual(self.client.get("/session").status_code, 401)

    def test_websocket_without_cookie_closes(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("/live") as socket:
                socket.receive_text()

    def test_source_and_frame_roundtrip(self):
        self.client.post("/login", json={"password": "secret"})
        image = base64.b64encode(b"faceA").decode("ascii")
        with self.client.websocket_connect("/live") as socket:
            socket.send_json({"type": "source", "image": image})
            self.assertEqual(socket.receive_json(), {"type": "ready"})
            socket.send_bytes(b"frame")
            self.assertEqual(socket.receive_bytes(), b"OUT")

    def test_no_face_keeps_socket_open(self):
        self.client.post("/login", json={"password": "secret"})
        image = base64.b64encode(b"noface").decode("ascii")
        with self.client.websocket_connect("/live") as socket:
            socket.send_json({"type": "source", "image": image})
            self.assertEqual(
                socket.receive_json(),
                {"type": "error", "message": "no face found"},
            )

    def test_ninth_socket_is_rejected(self):
        self.client.post("/login", json={"password": "secret"})
        for index in range(MAX_SESSIONS):
            self.assertTrue(self.engine.open_session(f"held-{index}"))
        with self.client.websocket_connect("/live") as socket:
            self.assertEqual(
                socket.receive_json(),
                {"type": "error", "message": "server is full"},
            )
        self.assertEqual(self.engine.session_count(), MAX_SESSIONS)


if __name__ == "__main__":
    unittest.main()
