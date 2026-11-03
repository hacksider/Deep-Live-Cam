"""The NSFW check must hand opennsfw2 an RGB frame.

Frames in the pipeline are BGR, and ``opennsfw2.preprocess_image`` expects an
RGB ``PIL.Image`` (it flips to BGR and subtracts the VGG mean itself). The
model is stubbed, so nothing is downloaded and only the channel order on the
way in is exercised.
"""
import importlib
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import modules.globals  # noqa: E402


def _load_predicter(seen, added):
    """Import modules.predicter with opennsfw2 stubbed to record its input.

    Stubs are installed only for modules that are not importable here, and the
    names added are collected so the caller can undo them.
    """

    def preprocess_image(pil_image, _preprocessing):
        seen.append(np.array(pil_image))
        return np.zeros((224, 224, 3), dtype=np.float32)

    insightface = types.ModuleType("insightface")
    insightface_app = types.ModuleType("insightface.app")
    insightface_common = types.ModuleType("insightface.app.common")
    insightface_common.Face = type("Face", (), {})

    stubs = {
        "opennsfw2": types.SimpleNamespace(
            Preprocessing=types.SimpleNamespace(YAHOO="YAHOO"),
            preprocess_image=preprocess_image,
            make_open_nsfw_model=lambda: types.SimpleNamespace(
                predict=lambda _views: [(1.0, 0.0)]
            ),
        ),
        "insightface": insightface,
        "insightface.app": insightface_app,
        "insightface.app.common": insightface_common,
    }
    for name, stub in stubs.items():
        if name not in sys.modules:
            sys.modules[name] = stub
            added.append(name)
    # opennsfw2 must be ours even if the real package is installed, so that
    # no model is ever built.
    sys.modules["opennsfw2"] = stubs["opennsfw2"]
    added.append("opennsfw2")

    sys.modules.pop("modules.predicter", None)
    return importlib.import_module("modules.predicter")


def _bgr_blue():
    """A pure-blue frame as OpenCV produces it: (B, G, R) = (255, 0, 0)."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    return frame


class PredictFrameChannelOrderTest(unittest.TestCase):
    def setUp(self):
        self._seen = []
        self._added = []
        self._color_correction = modules.globals.color_correction
        self.addCleanup(self._restore)
        self._predicter = _load_predicter(self._seen, self._added)

    def _restore(self):
        modules.globals.color_correction = self._color_correction
        sys.modules.pop("modules.predicter", None)
        for name in self._added:
            sys.modules.pop(name, None)

    def _channels_reaching_the_model(self):
        self._predicter.predict_frame(_bgr_blue())
        self.assertEqual(len(self._seen), 1)
        return [int(channel) for channel in self._seen[0][0, 0]]

    def test_blue_frame_stays_blue_with_color_correction_off(self):
        """The default. Without the conversion the model is told blue is red."""
        modules.globals.color_correction = False
        self.assertEqual(self._channels_reaching_the_model(), [0, 0, 255])

    def test_blue_frame_stays_blue_with_color_correction_on(self):
        modules.globals.color_correction = True
        self.assertEqual(self._channels_reaching_the_model(), [0, 0, 255])


if __name__ == "__main__":
    unittest.main()
