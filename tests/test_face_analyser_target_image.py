from __future__ import annotations

import importlib
import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock


def _load_face_analyser_module(target_frame=None, many_faces=None):
    calls = {"get_many_faces": []}

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.imread = lambda path: target_frame
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
    fake_globals.source_target_map = [{"id": 99}]
    fake_globals.target_path = "target.png"

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

        def fake_get_many_faces(frame):
            calls["get_many_faces"].append(frame)
            return many_faces

        face_analyser.get_many_faces = fake_get_many_faces
        return face_analyser, fake_globals, calls


class TargetImageFaceAnalysisTests(unittest.TestCase):
    def test_target_image_read_failure_clears_map_without_analyser_call(self) -> None:
        face_analyser, fake_globals, calls = _load_face_analyser_module(
            target_frame=None,
            many_faces=[],
        )

        output = StringIO()
        with redirect_stdout(output):
            result = face_analyser.get_unique_faces_from_target_image()

        self.assertEqual(result, [])
        self.assertEqual(fake_globals.source_target_map, [])
        self.assertEqual(calls["get_many_faces"], [])
        self.assertIn("Could not read target image: target.png", output.getvalue())

    def test_target_image_none_faces_leaves_map_empty(self) -> None:
        face_analyser, fake_globals, calls = _load_face_analyser_module(
            target_frame=object(),
            many_faces=None,
        )

        output = StringIO()
        with redirect_stdout(output):
            result = face_analyser.get_unique_faces_from_target_image()

        self.assertEqual(result, [])
        self.assertEqual(fake_globals.source_target_map, [])
        self.assertEqual(len(calls["get_many_faces"]), 1)
        self.assertIn("No faces detected in target image: target.png", output.getvalue())


if __name__ == "__main__":
    unittest.main()
