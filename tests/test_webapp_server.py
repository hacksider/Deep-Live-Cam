import base64
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from webapp.engine import MAX_SESSIONS, MAX_SOURCE_BYTES, LiveEngine
from webapp.server import LOGIN_FAIL_LIMIT, create_app, login_client_key


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


class LoginClientKeyTests(unittest.TestCase):
    def test_forwarded_last_hop_ignores_visitor_prefix(self):
        self.assertEqual(
            login_client_key("127.0.0.1", "spoofed, 203.0.113.5"),
            "203.0.113.5",
        )

    def test_forwarded_last_hop_of_many(self):
        self.assertEqual(
            login_client_key("127.0.0.1", "a, b, 203.0.113.5"),
            "203.0.113.5",
        )

    def test_forwarded_single_value(self):
        self.assertEqual(
            login_client_key("127.0.0.1", "203.0.113.5"), "203.0.113.5"
        )

    def test_forwarded_last_hop_is_what_proxy_saw(self):
        self.assertEqual(
            login_client_key("127.0.0.1", "203.0.113.5, 127.0.0.1"),
            "127.0.0.1",
        )

    def test_forwarded_ipv6_loopback_peer(self):
        self.assertEqual(
            login_client_key("::1", "spoofed, 203.0.113.5"), "203.0.113.5"
        )

    def test_forwarded_empty_last_entry_falls_back_to_peer(self):
        self.assertEqual(
            login_client_key("127.0.0.1", "203.0.113.5, "), "127.0.0.1"
        )

    def test_forwarded_empty_header_falls_back_to_peer(self):
        self.assertEqual(login_client_key("127.0.0.1", ""), "127.0.0.1")

    def test_different_prefixes_same_last_hop_same_key(self):
        self.assertEqual(
            login_client_key("127.0.0.1", "rand1, 203.0.113.5"),
            login_client_key("127.0.0.1", "rand2, rand3, 203.0.113.5"),
        )

    def test_missing_peer_ignores_forwarded(self):
        self.assertEqual(
            login_client_key(None, "spoofed, 203.0.113.5"), "unknown"
        )

    def test_localhost_peer_without_forwarded(self):
        self.assertEqual(login_client_key("127.0.0.1", None), "127.0.0.1")

    def test_non_localhost_peer_ignores_forwarded(self):
        self.assertEqual(
            login_client_key("testclient", "spoofed, 203.0.113.5"),
            "testclient",
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

    def test_odd_password_input_is_401_not_an_error(self):
        for given in ("caf\u00e9", "", "\u65e5\u672c"):
            response = self.client.post("/login", json={"password": given})
            self.assertEqual(response.status_code, 401, repr(given))
        # A JSON escape for a lone surrogate decodes to a str json.loads accepts.
        response = self.client.post(
            "/login",
            content=b'{"password": "bad\\ud800pass"}',
            headers={"content-type": "application/json"},
        )
        self.assertEqual(response.status_code, 401)

    def test_non_ascii_and_surrogate_cookies_fail_cleanly(self):
        from webapp.server import _cookie_ok

        self.assertFalse(_cookie_ok({"dlc_session": "caf\u00e9"}, "secret"))
        self.assertFalse(_cookie_ok({"dlc_session": "x\ud800y"}, "secret"))
        self.assertFalse(_cookie_ok({"dlc_session": "x"}, "pass\ud800word"))
        response = self.client.get(
            "/session", headers={"cookie": "dlc_session=caf\u00e9".encode("utf-8")}
        )
        self.assertEqual(response.status_code, 401)

    def test_non_ascii_cookie_closes_websocket(self):
        self.client.cookies.clear()
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect(
                "/live", headers={"cookie": "dlc_session=caf\u00e9".encode("utf-8")}
            ) as socket:
                socket.receive_text()

    def test_surrogate_in_configured_password_still_logs_in(self):
        with TestClient(create_app(self.engine, "pass\ud800word")) as client:
            self.assertEqual(
                client.post("/login", json={"password": "nope"}).status_code, 401
            )

    def test_cookie_is_not_secure_by_default(self):
        response = self.client.post("/login", json={"password": "secret"})
        self.assertNotIn("secure", response.headers["set-cookie"].lower().split("; "))

    def test_secure_cookie_flag_sets_secure(self):
        with TestClient(create_app(self.engine, "secret", secure_cookie=True)) as client:
            response = client.post("/login", json={"password": "secret"})
        set_cookie = response.headers["set-cookie"].lower()
        self.assertIn("secure", set_cookie.split("; "))
        self.assertIn("httponly", set_cookie)
        self.assertIn("samesite=lax", set_cookie)
        self.assertIn("path=/", set_cookie)
        self.assertNotIn("max-age", set_cookie)

    def test_wrong_password_logs_a_warning(self):
        with self.assertLogs("webapp.server", level="WARNING"):
            self.client.post("/login", json={"password": "nope"})

    def test_repeated_failures_are_rate_limited_with_429(self):
        for _ in range(LOGIN_FAIL_LIMIT):
            response = self.client.post("/login", json={"password": "nope"})
            self.assertEqual(response.status_code, 401)
        response = self.client.post("/login", json={"password": "nope"})
        self.assertEqual(response.status_code, 429)
        response = self.client.post("/login", json={"password": "secret"})
        self.assertEqual(response.status_code, 429)

    def test_failure_window_expires(self):
        from webapp import server

        for _ in range(LOGIN_FAIL_LIMIT):
            self.client.post("/login", json={"password": "nope"})
        real = server.time.monotonic
        with mock.patch.object(
            server.time, "monotonic", lambda: real() + server.LOGIN_FAIL_WINDOW + 1
        ):
            response = self.client.post("/login", json={"password": "secret"})
        self.assertEqual(response.status_code, 200)

    def test_oversized_source_reports_image_too_large(self):
        self.client.post("/login", json={"password": "secret"})
        image = base64.b64encode(b"x" * (MAX_SOURCE_BYTES + 1)).decode("ascii")
        with self.client.websocket_connect("/live") as socket:
            socket.send_json({"type": "source", "image": image})
            self.assertEqual(
                socket.receive_json(),
                {"type": "error", "message": "image too large"},
            )

    def test_worker_survives_a_failing_drain(self):
        state = {"armed": False, "raised": False, "after": 0}
        real_drain = self.engine.drain

        def flaky():
            if state["armed"] and not state["raised"]:
                state["raised"] = True
                raise RuntimeError("bad pass")
            if state["raised"]:
                state["after"] += 1
            real_drain()

        self.engine.drain = flaky
        with self.assertLogs("webapp.server", level="ERROR"):
            state["armed"] = True
            with TestClient(create_app(self.engine, "secret")) as client:
                client.post("/login", json={"password": "secret"})
                image = base64.b64encode(b"faceA").decode("ascii")
                with client.websocket_connect("/live") as socket:
                    socket.send_json({"type": "source", "image": image})
                    self.assertEqual(socket.receive_json(), {"type": "ready"})
        self.assertTrue(state["raised"])
        self.assertGreater(state["after"], 0)

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
