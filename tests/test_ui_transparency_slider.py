"""Regression test for what the Transparency slider is allowed to touch.

Only the handler is exercised: no widget is built and no model is loaded, so
the test is fast and needs no display.
"""
import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.globals  # noqa: E402
import modules.ui as ui  # noqa: E402


class TransparencySliderTest(unittest.TestCase):
    """Transparency blends the swapped face. It must not disable a processor.

    The swap is already skipped at 0% by the `opacity == 0` guards in
    face_swapper.process_frame / process_frame_v2, so there is nothing for the
    slider to switch off, and nothing ever switched the enhancer back on.
    """

    def setUp(self):
        saved_fp_ui = dict(modules.globals.fp_ui)
        saved_opacity = modules.globals.opacity

        def restore():
            modules.globals.fp_ui.clear()
            modules.globals.fp_ui.update(saved_fp_ui)
            modules.globals.opacity = saved_opacity

        self.addCleanup(restore)

    def _drag(self, *values):
        # The handler ignores `self`, so no window is needed.
        with patch.object(ui, "update_status", lambda *_: None):
            for value in values:
                ui.MainWindow._on_transparency_change(None, value)

    def test_enhancer_survives_a_pass_through_zero(self):
        modules.globals.fp_ui["face_enhancer"] = True

        self._drag(0.0, 1.0)

        self.assertTrue(
            modules.globals.fp_ui["face_enhancer"],
            "the transparency slider switched GFPGAN off on its way through "
            "0% and never switched it back on, while the enhancer dropdown "
            "still reads GFPGAN",
        )

    def test_slider_still_sets_opacity(self):
        self._drag(0.0)
        self.assertEqual(modules.globals.opacity, 0.0)

        self._drag(0.4)
        self.assertAlmostEqual(modules.globals.opacity, 0.4)


if __name__ == "__main__":
    unittest.main()
