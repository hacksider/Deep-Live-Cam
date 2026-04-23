import sys
import types
import unittest

import numpy as np


common_module = types.ModuleType("insightface.app.common")
common_module.Face = object
app_module = types.ModuleType("insightface.app")
app_module.common = common_module
insightface_module = types.ModuleType("insightface")
insightface_module.app = app_module
sys.modules.setdefault("insightface", insightface_module)
sys.modules.setdefault("insightface.app", app_module)
sys.modules.setdefault("insightface.app.common", common_module)

from modules.processors.frame import face_masking


class FaceMaskingBadInputTest(unittest.TestCase):
    def test_create_face_mask_returns_empty_mask_for_invalid_frame(self):
        face = types.SimpleNamespace(landmark_2d_106=np.zeros((106, 2)))

        self.assertEqual(face_masking.create_face_mask(face, None).shape, (0, 0))
        self.assertEqual(face_masking.create_face_mask(face, object()).shape, (0, 0))
        self.assertEqual(
            face_masking.create_face_mask(face, types.SimpleNamespace(shape=(10,))).shape,
            (0, 0),
        )

    def test_create_face_mask_returns_frame_sized_empty_mask_without_landmarks(self):
        frame = np.zeros((4, 5, 3), dtype=np.uint8)

        mask = face_masking.create_face_mask(types.SimpleNamespace(landmark_2d_106=None), frame)

        self.assertEqual(mask.shape, (4, 5))
        self.assertFalse(mask.any())

    def test_lower_mouth_mask_returns_empty_defaults_for_bad_inputs(self):
        mask, cutout, box, polygon = face_masking.create_lower_mouth_mask(None, None)

        self.assertEqual(mask.shape, (0, 0))
        self.assertIsNone(cutout)
        self.assertEqual(box, (0, 0, 0, 0))
        self.assertIsNone(polygon)

    def test_eyes_mask_returns_empty_defaults_for_bad_inputs(self):
        frame = np.zeros((4, 5, 3), dtype=np.uint8)

        mask, cutout, box, polygon = face_masking.create_eyes_mask(
            types.SimpleNamespace(landmark_2d_106=None), frame
        )

        self.assertEqual(mask.shape, (4, 5))
        self.assertFalse(mask.any())
        self.assertIsNone(cutout)
        self.assertEqual(box, (0, 0, 0, 0))
        self.assertIsNone(polygon)


if __name__ == "__main__":
    unittest.main()
