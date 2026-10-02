import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from webapp.engine import LiveEngine
from webapp.server import create_app

PAGE = Path("webapp/static/index.html")


class PageTests(unittest.TestCase):
    def test_page_contains_the_live_protocol(self):
        text = PAGE.read_text(encoding="utf-8")
        self.assertIn("getUserMedia", text)
        self.assertIn("/live", text)
        self.assertIn("0.7", text)
        self.assertIn("640", text)
        self.assertIn("Use a face only with that person's consent", text)
        self.assertIn("label anything you share as a deepfake", text)
        compact = "".join(text.split())
        self.assertIn('{type:"source",image:', compact)

    def test_page_treats_image_too_large_as_fatal_and_checks_size(self):
        text = PAGE.read_text(encoding="utf-8")
        self.assertIn('message.message === "image too large"', text)
        self.assertIn("file.size > 8000000", text)
        self.assertLess(text.index("file.size > 8000000"), text.index("new WebSocket("))

    def test_root_serves_the_page_without_a_cookie(self):
        engine = LiveEngine(
            get_one_face=lambda image: None,
            detect_one_face=lambda frame: None,
            swap_face=lambda source, target, frame: frame,
            decode_image=lambda data: None,
            encode_jpeg=lambda frame: b"",
        )
        with TestClient(create_app(engine, "secret")) as client:
            response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("getUserMedia", response.text)
        self.assertIn('const basePath = "";', response.text)
        self.assertNotIn("__BASE_PATH_JSON__", response.text)


if __name__ == "__main__":
    unittest.main()
