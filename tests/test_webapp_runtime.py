import unittest

import modules.globals as globals_module
from webapp.runtime import (
    BIND_HOST,
    apply_swap_defaults,
    choose_providers,
    ensure_swapper,
    require_password,
    web_port,
)


class RuntimeTests(unittest.TestCase):
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

    def test_bind_host_is_loopback(self):
        self.assertEqual(BIND_HOST, "127.0.0.1")

    def test_cuda_wins_over_cpu(self):
        available = ["CPUExecutionProvider", "CUDAExecutionProvider"]
        self.assertEqual(choose_providers(available), ["CUDAExecutionProvider"])
        self.assertEqual(choose_providers(["CPUExecutionProvider"]), ["CPUExecutionProvider"])

    def test_defaults_are_process_wide_constants(self):
        apply_swap_defaults(globals_module)
        self.assertFalse(globals_module.mouth_mask)
        self.assertEqual(globals_module.opacity, 1.0)
        self.assertFalse(globals_module.poisson_blend)

    def test_missing_model_exits(self):
        with self.assertRaises(SystemExit):
            ensure_swapper(lambda: False)
        ensure_swapper(lambda: True)


if __name__ == "__main__":
    unittest.main()
