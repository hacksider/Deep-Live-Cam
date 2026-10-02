import unittest

from webapp.engine import MAX_JPEG_BYTES, MAX_SESSIONS, MAX_SOURCE_BYTES, LiveEngine


class LiveEngineTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def decode_image(data: bytes):
            if data in (b"bad", b"bad-source"):
                return None
            return data

        def encode_jpeg(frame: bytes) -> bytes:
            return b"enc:" + frame

        def get_one_face(image: bytes):
            if image == b"noface":
                return None
            return image

        def detect_one_face(frame: bytes):
            if frame == b"nobody":
                return None
            return b"target:" + frame

        def swap_face(source, target, frame):
            self.calls.append((source, target, frame))
            if frame == b"boom":
                raise RuntimeError("boom")
            return b"swapped:" + source + b":" + frame

        self.engine = LiveEngine(
            get_one_face=get_one_face,
            detect_one_face=detect_one_face,
            swap_face=swap_face,
            decode_image=decode_image,
            encode_jpeg=encode_jpeg,
        )

    def _ready(self, session_id: str, face: bytes) -> None:
        self.assertTrue(self.engine.open_session(session_id))
        self.engine.submit_source(session_id, face)
        self.engine.drain()
        self.assertEqual(self.engine.source_message(session_id), "ready")

    def test_newer_unread_frame_is_the_only_one_swapped(self):
        self._ready("a", b"faceA")
        self.engine.submit_frame("a", b"old")
        self.engine.submit_frame("a", b"new")
        self.engine.drain()
        self.assertEqual(self.calls, [(b"faceA", b"target:new", b"new")])
        self.assertEqual(self.engine.frame_result("a", 1), ("jpeg", b"enc:swapped:faceA:new"))

    def test_two_sessions_keep_their_own_source_faces(self):
        self._ready("a", b"faceA")
        self._ready("b", b"faceB")
        self.engine.submit_frame("a", b"fa")
        self.engine.submit_frame("b", b"fb")
        self.engine.drain()
        self.assertEqual(
            self.calls,
            [
                (b"faceA", b"target:fa", b"fa"),
                (b"faceB", b"target:fb", b"fb"),
            ],
        )

    def test_close_removes_only_that_session(self):
        self._ready("a", b"faceA")
        self._ready("b", b"faceB")
        self.engine.close_session("a")
        self.engine.submit_frame("b", b"fb")
        self.engine.drain()
        self.assertEqual(self.engine.session_count(), 1)
        self.assertEqual(self.calls, [(b"faceB", b"target:fb", b"fb")])
        self.assertIsNone(self.engine.source_message("a"))

    def test_ninth_session_is_rejected_and_not_counted(self):
        for index in range(MAX_SESSIONS):
            self.assertTrue(self.engine.open_session(str(index)))
        self.assertFalse(self.engine.open_session("extra"))
        self.assertEqual(self.engine.session_count(), MAX_SESSIONS)

    def test_source_with_no_face_is_not_swappable(self):
        self.assertTrue(self.engine.open_session("a"))
        self.engine.submit_source("a", b"noface")
        self.engine.drain()
        self.assertEqual(self.engine.source_message("a"), "no face found")
        seq, rejected = self.engine.submit_frame("a", b"frame")
        self.engine.drain()
        self.assertIsNone(rejected)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.engine.frame_result("a", seq), ("error", "no face found"))

    def test_unreadable_source(self):
        self.assertTrue(self.engine.open_session("a"))
        self.engine.submit_source("a", b"bad-source")
        self.engine.drain()
        self.assertEqual(self.engine.source_message("a"), "unreadable image")

    def test_oversized_source_is_image_too_large(self):
        self.assertTrue(self.engine.open_session("a"))
        self.engine.submit_source("a", b"x" * (MAX_SOURCE_BYTES + 1))
        self.engine.drain()
        self.assertEqual(self.engine.source_message("a"), "image too large")

    def test_oversized_jpeg_is_rejected_without_swap(self):
        self._ready("a", b"faceA")
        seq, rejected = self.engine.submit_frame("a", b"x" * (MAX_JPEG_BYTES + 1))
        self.engine.drain()
        self.assertEqual(seq, 0)
        self.assertEqual(rejected, "bad frame")
        self.assertEqual(self.calls, [])

    def test_undecodable_jpeg_returns_bad_frame(self):
        self._ready("a", b"faceA")
        seq, rejected = self.engine.submit_frame("a", b"bad")
        self.assertIsNone(rejected)
        self.engine.drain()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.engine.frame_result("a", seq), ("error", "bad frame"))

    def test_missing_target_face_returns_original_jpeg(self):
        self._ready("a", b"faceA")
        seq, _rejected = self.engine.submit_frame("a", b"nobody")
        self.engine.drain()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.engine.frame_result("a", seq), ("jpeg", b"nobody"))

    def test_swap_exception_returns_original_and_later_frame_still_runs(self):
        self._ready("a", b"faceA")
        boom_seq, _rejected = self.engine.submit_frame("a", b"boom")
        self.engine.drain()
        self.assertEqual(self.engine.frame_result("a", boom_seq), ("jpeg", b"boom"))
        ok_seq, _rejected = self.engine.submit_frame("a", b"ok")
        self.engine.drain()
        self.assertEqual(
            self.engine.frame_result("a", ok_seq),
            ("jpeg", b"enc:swapped:faceA:ok"),
        )

    def test_source_decode_exception_does_not_abort_drain(self):
        def decode_image(data: bytes):
            if data == b"raise-source":
                raise ValueError("decode blew up")
            if data in (b"bad", b"bad-source"):
                return None
            return data

        engine = LiveEngine(
            get_one_face=lambda image: image,
            detect_one_face=lambda frame: b"target:" + frame,
            swap_face=lambda source, target, frame: b"swapped",
            decode_image=decode_image,
            encode_jpeg=lambda frame: b"enc:" + frame,
        )
        self.assertTrue(engine.open_session("bad"))
        engine.submit_source("bad", b"raise-source")
        self.assertTrue(engine.open_session("good"))
        engine.submit_source("good", b"faceB")
        seq, _rejected = engine.submit_frame("good", b"fb")
        engine.drain()
        self.assertEqual(engine.source_message("bad"), "unreadable image")
        self.assertEqual(engine.source_message("good"), "ready")
        self.assertEqual(engine.frame_result("good", seq), ("jpeg", b"enc:swapped"))

    def test_get_one_face_exception_does_not_abort_drain(self):
        def get_one_face(image: bytes):
            if image == b"raise-face":
                raise RuntimeError("face detector blew up")
            return image

        engine = LiveEngine(
            get_one_face=get_one_face,
            detect_one_face=lambda frame: b"target:" + frame,
            swap_face=lambda source, target, frame: b"swapped",
            decode_image=lambda data: data,
            encode_jpeg=lambda frame: b"enc:" + frame,
        )
        self.assertTrue(engine.open_session("bad"))
        engine.submit_source("bad", b"raise-face")
        self.assertTrue(engine.open_session("good"))
        engine.submit_source("good", b"faceB")
        seq, _rejected = engine.submit_frame("good", b"fb")
        engine.drain()
        self.assertEqual(engine.source_message("bad"), "unreadable image")
        self.assertEqual(engine.source_message("good"), "ready")
        self.assertEqual(engine.frame_result("good", seq), ("jpeg", b"enc:swapped"))


if __name__ == "__main__":
    unittest.main()
