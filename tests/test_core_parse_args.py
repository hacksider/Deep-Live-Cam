import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

from test_core_map_faces_fallback import _patched_core_import_stubs


def _parse(core, argv):
    # The stubbed modules.metadata is only in sys.modules, not on the package.
    with patch.object(sys, "argv", ["run.py", *argv]), patch.object(
        core.modules, "metadata", sys.modules["modules.metadata"], create=True
    ):
        core.parse_args()


class ParseArgsTests(unittest.TestCase):
    def test_cli_mode_without_target_exits_with_usage_error(self):
        for argv in (["-o", "out.mp4"], ["-s", "source.jpg"]):
            with self.subTest(argv=argv), _patched_core_import_stubs([]) as core:
                stderr = StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    _parse(core, argv)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("-s/--source and -t/--target", stderr.getvalue())

    def test_cli_mode_with_source_and_target(self):
        with _patched_core_import_stubs([]) as core:
            _parse(core, ["-s", "source.jpg", "-t", "target.mp4"])
            self.assertTrue(core.modules.globals.headless)

    def test_gui_mode_without_arguments(self):
        with _patched_core_import_stubs([]) as core:
            _parse(core, [])
            self.assertFalse(core.modules.globals.headless)


if __name__ == "__main__":
    unittest.main()
