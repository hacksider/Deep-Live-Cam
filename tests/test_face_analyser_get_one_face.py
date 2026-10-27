import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from test_core_map_faces_fallback import _isolated_module_import


@contextmanager
def _patched_face_analyser_import():
    stubs = {
        "insightface": types.SimpleNamespace(app=types.SimpleNamespace(FaceAnalysis=object)),
        "cv2": types.SimpleNamespace(
            IMREAD_COLOR=1,
            imread=lambda *_args, **_kwargs: None,
            imdecode=lambda *_args, **_kwargs: None,
            imencode=lambda *_args, **_kwargs: (
                True,
                types.SimpleNamespace(tofile=lambda *_a, **_k: None),
            ),
        ),
        "numpy": types.SimpleNamespace(uint8=object, fromfile=lambda *_args, **_kwargs: b""),
        "tqdm": types.SimpleNamespace(tqdm=lambda iterable, **_kwargs: iterable),
        "modules.typing": types.SimpleNamespace(Frame=object),
        "modules.cluster_analysis": types.SimpleNamespace(
            find_cluster_centroids=lambda *args, **kwargs: [],
            find_closest_centroid=lambda *args, **kwargs: (0, None),
        ),
        "modules.utilities": types.SimpleNamespace(
            get_temp_directory_path=lambda path: path,
            create_temp=lambda path: None,
            extract_frames=lambda path: None,
            clean_temp=lambda path: None,
            get_temp_frame_paths=lambda path: [],
        ),
    }
    with _isolated_module_import(stubs, "modules.face_analyser") as analyser:
        yield analyser


class Face:
    def __init__(self, left):
        self.bbox = [left, 0, 10, 10]


class GetOneFaceTests(unittest.TestCase):
    def setUp(self):
        scope = _patched_face_analyser_import()
        self.face_analyser = scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)

    def test_uses_supplied_detected_faces_without_reanalysing_frame(self):
        face_analyser = self.face_analyser
        right = Face(20)
        left = Face(5)

        with patch.object(
            face_analyser,
            "_analyse_faces",
            side_effect=AssertionError("should not analyse"),
        ):
            self.assertIs(face_analyser.get_one_face("frame", [right, left]), left)

    def test_supplied_empty_detected_faces_returns_none(self):
        face_analyser = self.face_analyser

        with patch.object(
            face_analyser,
            "_analyse_faces",
            side_effect=AssertionError("should not analyse"),
        ):
            self.assertIsNone(face_analyser.get_one_face("frame", []))

    def test_without_supplied_faces_preserves_existing_detection_path(self):
        face_analyser = self.face_analyser
        right = Face(30)
        left = Face(3)

        with patch.object(face_analyser, "_is_dml", return_value=False), patch.object(
            face_analyser,
            "_analyse_faces",
            return_value=[right, left],
        ) as analyse:
            self.assertIs(face_analyser.get_one_face("frame"), left)

        analyse.assert_called_once_with("frame")


class ImportIsolationTests(unittest.TestCase):
    def test_restores_analyser_and_utilities_references(self):
        package = importlib.import_module("modules")
        original_analyser = types.ModuleType("modules.face_analyser")
        original_utilities = types.ModuleType("modules.utilities")
        with patch.dict(sys.modules, {
            "modules.face_analyser": original_analyser,
            "modules.utilities": original_utilities,
        }), patch.object(package, "face_analyser", original_analyser, create=True), patch.object(
            package, "utilities", original_utilities, create=True
        ):
            with _patched_face_analyser_import() as analyser:
                self.assertIs(package.face_analyser, analyser)
                self.assertIsNot(package.utilities, original_utilities)
            self.assertIs(sys.modules["modules.face_analyser"], original_analyser)
            self.assertIs(sys.modules["modules.utilities"], original_utilities)
            self.assertIs(package.face_analyser, original_analyser)
            self.assertIs(package.utilities, original_utilities)


if __name__ == "__main__":
    unittest.main()
