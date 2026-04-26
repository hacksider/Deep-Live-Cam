import importlib
import sys
import types
import unittest

import numpy as np


class FaceAnalyserInvalidFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        insightface = types.ModuleType("insightface")
        insightface_app = types.ModuleType("insightface.app")
        insightface_common = types.ModuleType("insightface.app.common")
        insightface_common.Face = object
        insightface_app.common = insightface_common
        insightface.app = insightface_app
        sys.modules.setdefault("insightface", insightface)
        sys.modules.setdefault("insightface.app", insightface_app)
        sys.modules.setdefault("insightface.app.common", insightface_common)

        cluster_analysis = types.ModuleType("modules.cluster_analysis")
        cluster_analysis.find_cluster_centroids = lambda *args, **kwargs: []
        cluster_analysis.find_closest_centroid = lambda *args, **kwargs: None
        sys.modules.setdefault("modules.cluster_analysis", cluster_analysis)

        cls.face_analyser = importlib.import_module("modules.face_analyser")

    def test_get_one_face_returns_none_for_none_frame(self):
        self.assertIsNone(self.face_analyser.get_one_face(None))

    def test_get_many_faces_returns_empty_list_for_object_without_shape(self):
        self.assertEqual(self.face_analyser.get_many_faces(object()), [])

    def test_invalid_1d_frame_does_not_initialize_insightface(self):
        original = self.face_analyser.get_face_analyser
        try:
            self.face_analyser.get_face_analyser = lambda: self.fail("InsightFace should not be initialized")

            self.assertIsNone(self.face_analyser.get_one_face(np.array([1, 2, 3])))
            self.assertEqual(self.face_analyser.get_many_faces(np.array([1, 2, 3])), [])
        finally:
            self.face_analyser.get_face_analyser = original


if __name__ == "__main__":
    unittest.main()
