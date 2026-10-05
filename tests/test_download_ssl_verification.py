import importlib
import os
import ssl
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class _FakeTqdm:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n):
        pass


def _install_import_stubs():
    sys.modules.setdefault("cv2", types.SimpleNamespace(IMREAD_COLOR=1))
    sys.modules.setdefault(
        "numpy",
        types.SimpleNamespace(uint8=object, fromfile=lambda *_args, **_kwargs: b""),
    )
    sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=_FakeTqdm))


class _FakeHeaders:
    def __init__(self, values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


class _FakeResponse:
    def __init__(self, payload=b""):
        self.headers = _FakeHeaders({"Content-Length": str(len(payload))})
        self._payload = payload
        self.status = 200

    def read(self, size=-1):
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _load_model_downloader():
    _install_import_stubs()
    sys.modules.pop("modules.model_downloader", None)
    return importlib.import_module("modules.model_downloader")


def _load_utilities():
    _install_import_stubs()
    sys.modules.pop("modules.utilities", None)
    return importlib.import_module("modules.utilities")


class SslContextTests(unittest.TestCase):
    def test_ssl_context_verifies_certificates(self):
        model_downloader = _load_model_downloader()
        context = model_downloader._ssl_context()
        self.assertIsInstance(context, ssl.SSLContext)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_ssl_context_verifies_on_every_platform(self):
        # Regression test for #1890: macOS used to get an unverified context.
        model_downloader = _load_model_downloader()
        for system in ("Darwin", "Windows", "Linux"):
            with patch("platform.system", return_value=system):
                context = model_downloader._ssl_context()
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED, system)
            self.assertTrue(context.check_hostname, system)


class ConditionalDownloadTests(unittest.TestCase):
    def test_passes_verifying_context_to_urlopen(self):
        utilities = _load_utilities()
        seen = {}

        def fake_urlopen(request, context=None, **kwargs):
            seen["context"] = context
            return _FakeResponse(b"fake-model-bytes")

        with tempfile.TemporaryDirectory() as tmp:
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                utilities.conditional_download(tmp, ["https://example.com/x.onnx"])
            self.assertTrue(os.path.isfile(os.path.join(tmp, "x.onnx")))

        context = seen.get("context")
        self.assertIsInstance(context, ssl.SSLContext)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)


class ModelDownloaderTests(unittest.TestCase):
    def test_download_passes_verifying_context_to_urlopen(self):
        model_downloader = _load_model_downloader()
        seen = {}

        def fake_urlopen(request, context=None, **kwargs):
            seen["context"] = context
            return _FakeResponse(b"fake-model-bytes")

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "x.onnx")
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                ok = model_downloader._download("x.onnx", "https://example.com/x.onnx", target, None)
            self.assertTrue(ok)
            self.assertTrue(os.path.isfile(target))

        context = seen.get("context")
        self.assertIsInstance(context, ssl.SSLContext)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)


if __name__ == "__main__":
    unittest.main()
