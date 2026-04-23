from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock


class FakeFrame:
    def __getitem__(self, key):
        return ("crop", key)


def _load_face_analyser_module(source_target_map):
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.imread = lambda path: FakeFrame()
    fake_cv2.imdecode = lambda *args, **kwargs: None
    fake_cv2.imencode = lambda *args, **kwargs: (
        True,
        types.SimpleNamespace(tofile=lambda *_args: None),
    )

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.fromfile = lambda *args, **kwargs: b""
    fake_numpy.uint8 = int
    fake_insightface = types.ModuleType("insightface")
    fake_insightface.app = types.SimpleNamespace(FaceAnalysis=object)

    fake_globals = types.ModuleType("modules.globals")
    fake_globals.execution_providers = []
    fake_globals.source_target_map = source_target_map

    fake_cluster = types.ModuleType("modules.cluster_analysis")
    fake_cluster.find_cluster_centroids = lambda *args, **kwargs: []
    fake_cluster.find_closest_centroid = lambda *args, **kwargs: (0, 0)

    fake_utilities = types.ModuleType("modules.utilities")
    fake_utilities.get_temp_directory_path = lambda *args, **kwargs: ""
    fake_utilities.create_temp = lambda *args, **kwargs: None
    fake_utilities.extract_frames = lambda *args, **kwargs: None
    fake_utilities.clean_temp = lambda *args, **kwargs: None
    fake_utilities.get_temp_frame_paths = lambda *args, **kwargs: []

    fake_typing = types.ModuleType("modules.typing")
    fake_typing.Frame = object
    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda items, *args, **kwargs: items

    sys.modules.pop("modules.face_analyser", None)
    with mock.patch.dict(
        sys.modules,
        {
            "cv2": fake_cv2,
            "insightface": fake_insightface,
            "numpy": fake_numpy,
            "modules.globals": fake_globals,
            "modules.cluster_analysis": fake_cluster,
            "modules.utilities": fake_utilities,
            "modules.typing": fake_typing,
            "tqdm": fake_tqdm,
        },
        clear=False,
    ):
        face_analyser = importlib.import_module("modules.face_analyser")
        face_analyser.modules.globals = fake_globals
        return face_analyser


class DefaultTargetFaceTests(unittest.TestCase):
    def test_default_target_face_skips_cluster_with_no_detected_faces(self) -> None:
        source_target_map = [
            {
                "id": 0,
                "target_faces_in_frame": [
                    {"frame": 0, "faces": [], "location": "frame-0.png"},
                    {"frame": 1, "faces": [], "location": "frame-1.png"},
                ],
            }
        ]

        face_analyser = _load_face_analyser_module(source_target_map)

        face_analyser.default_target_face()

        self.assertNotIn("target", source_target_map[0])

    def test_default_target_face_keeps_processing_later_clusters(self) -> None:
        lower_score_face = {"det_score": 0.4, "bbox": [0, 0, 10, 10]}
        higher_score_face = {"det_score": 0.9, "bbox": [1, 2, 11, 12]}
        source_target_map = [
            {
                "id": 0,
                "target_faces_in_frame": [
                    {"frame": 0, "faces": [], "location": "empty.png"},
                ],
            },
            {
                "id": 1,
                "target_faces_in_frame": [
                    {
                        "frame": 1,
                        "faces": [lower_score_face],
                        "location": "lower.png",
                    },
                    {
                        "frame": 2,
                        "faces": [higher_score_face],
                        "location": "higher.png",
                    },
                ],
            },
        ]

        face_analyser = _load_face_analyser_module(source_target_map)

        face_analyser.default_target_face()

        self.assertNotIn("target", source_target_map[0])
        self.assertEqual(source_target_map[1]["target"]["face"], higher_score_face)
        self.assertEqual(
            source_target_map[1]["target"]["cv2"],
            ("crop", (slice(2, 12, None), slice(1, 11, None))),
        )


if __name__ == "__main__":
    unittest.main()
