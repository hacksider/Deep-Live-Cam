import importlib
import sys
import types
import unittest
from unittest.mock import patch


def _install_import_stubs():
    sys.modules.setdefault(
        "insightface",
        types.SimpleNamespace(app=types.SimpleNamespace(FaceAnalysis=object)),
    )
    sys.modules.setdefault(
        "cv2",
        types.SimpleNamespace(
            IMREAD_COLOR=1,
            imread=lambda *_args, **_kwargs: None,
            imdecode=lambda *_args, **_kwargs: None,
            imencode=lambda *_args, **_kwargs: (
                True,
                types.SimpleNamespace(tofile=lambda *_a, **_k: None),
            ),
        ),
    )
    sys.modules.setdefault(
        "numpy",
        types.SimpleNamespace(uint8=object, fromfile=lambda *_args, **_kwargs: b""),
    )
    sys.modules.setdefault(
        "tqdm",
        types.SimpleNamespace(tqdm=lambda iterable, **_kwargs: iterable),
    )
    sys.modules["modules.typing"] = types.SimpleNamespace(Frame=object)
    sys.modules["modules.cluster_analysis"] = types.SimpleNamespace(
        find_cluster_centroids=lambda *args, **kwargs: [],
        find_closest_centroid=lambda *args, **kwargs: (0, None),
    )
    sys.modules["modules.utilities"] = types.SimpleNamespace(
        get_temp_directory_path=lambda path: path,
        create_temp=lambda path: None,
        extract_frames=lambda path: None,
        clean_temp=lambda path: None,
        get_temp_frame_paths=lambda path: [],
    )


def _load_face_analyser():
    _install_import_stubs()
    sys.modules.pop("modules.face_analyser", None)
    return importlib.import_module("modules.face_analyser")


class Face:
    def __init__(self, left):
        self.bbox = [left, 0, 10, 10]


class GetOneFaceTests(unittest.TestCase):
    def test_uses_supplied_detected_faces_without_reanalysing_frame(self):
        face_analyser = _load_face_analyser()
        right = Face(20)
        left = Face(5)

        with patch.object(
            face_analyser,
            "_analyse_faces",
            side_effect=AssertionError("should not analyse"),
        ):
            self.assertIs(face_analyser.get_one_face("frame", [right, left]), left)

    def test_supplied_empty_detected_faces_returns_none(self):
        face_analyser = _load_face_analyser()

        with patch.object(
            face_analyser,
            "_analyse_faces",
            side_effect=AssertionError("should not analyse"),
        ):
            self.assertIsNone(face_analyser.get_one_face("frame", []))

    def test_without_supplied_faces_preserves_existing_detection_path(self):
        face_analyser = _load_face_analyser()
        right = Face(30)
        left = Face(3)

        with patch.object(face_analyser, "_is_dml", return_value=False), patch.object(
            face_analyser,
            "_analyse_faces",
            return_value=[right, left],
        ) as analyse:
            self.assertIs(face_analyser.get_one_face("frame"), left)

        analyse.assert_called_once_with("frame")


if __name__ == "__main__":
    unittest.main()
