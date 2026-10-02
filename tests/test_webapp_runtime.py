import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from webapp.runtime import (
    BIND_HOST,
    apply_swap_defaults,
    base_path,
    choose_providers,
    ensure_swapper,
    load_web_env,
    require_password,
    web_port,
)


class RuntimeTests(unittest.TestCase):
    def test_env_file_fills_missing_values_only(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("WEB_PASSWORD=from-file\nWEB_PORT=3003\n", encoding="utf-8")
            saved = {name: os.environ.get(name) for name in ("WEB_PASSWORD", "WEB_PORT")}
            os.environ.pop("WEB_PASSWORD", None)
            os.environ["WEB_PORT"] = "9000"
            try:
                load_web_env(env_file)
                self.assertEqual(os.environ["WEB_PASSWORD"], "from-file")
                self.assertEqual(os.environ["WEB_PORT"], "9000")
            finally:
                for name, value in saved.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    def test_password_is_required(self):
        with self.assertRaises(SystemExit):
            require_password(None)
        with self.assertRaises(SystemExit):
            require_password("")
        self.assertEqual(require_password("secret"), "secret")

    def test_port_defaults_to_8000(self):
        self.assertEqual(web_port(None), 8000)
        self.assertEqual(web_port(""), 8000)
        self.assertEqual(web_port("9000"), 9000)

    def test_base_path_normalizes_a_prefix(self):
        self.assertEqual(base_path(None), "")
        self.assertEqual(base_path(""), "")
        self.assertEqual(base_path("/"), "")
        self.assertEqual(base_path("/DEEPFAKE/"), "/DEEPFAKE")
        self.assertEqual(base_path("DEEPFAKE"), "/DEEPFAKE")
        with self.assertRaises(SystemExit):
            base_path("/DEEPFAKE/../other")
        with self.assertRaises(SystemExit):
            base_path("/DEEP FAKE")

    def test_bind_host_is_loopback(self):
        self.assertEqual(BIND_HOST, "127.0.0.1")

    def test_cuda_wins_over_cpu(self):
        available = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        self.assertEqual(choose_providers(available), ["CUDAExecutionProvider"])
        self.assertEqual(choose_providers(["CPUExecutionProvider"]), ["CPUExecutionProvider"])

    def test_defaults_are_process_wide_constants(self):
        ns = SimpleNamespace(mouth_mask=True, opacity=0.5, poisson_blend=True)
        apply_swap_defaults(ns)
        self.assertFalse(ns.mouth_mask)
        self.assertEqual(ns.opacity, 1.0)
        self.assertFalse(ns.poisson_blend)

    def test_missing_model_exits(self):
        with self.assertRaises(SystemExit):
            ensure_swapper(lambda: False)
        ensure_swapper(lambda: True)


if __name__ == "__main__":
    unittest.main()
