import importlib
import sys
import types
import unittest
from unittest.mock import patch


def _install_import_stubs():
    sys.modules.setdefault(
        "cv2",
        types.SimpleNamespace(
            IMREAD_COLOR=1,
            imread=lambda *_args, **_kwargs: None,
            imdecode=lambda *_args, **_kwargs: None,
            imencode=lambda *_args, **_kwargs: (True, types.SimpleNamespace(tofile=lambda *_a, **_k: None)),
        ),
    )
    sys.modules.setdefault("insightface", types.SimpleNamespace())
    sys.modules.setdefault(
        "modules.globals",
        types.SimpleNamespace(
            execution_providers=[],
            opacity=1.0,
            mouth_mask=False,
            many_faces=False,
            dml_lock=types.SimpleNamespace(
                __enter__=lambda self: self,
                __exit__=lambda self, exc_type, exc, tb: False,
            ),
        ),
    )
    sys.modules.setdefault(
        "modules.processors.frame.core",
        types.SimpleNamespace(process_video=lambda *_args, **_kwargs: None),
    )
    sys.modules.setdefault(
        "modules.core",
        types.SimpleNamespace(update_status=lambda *_args, **_kwargs: None),
    )
    sys.modules.setdefault(
        "modules.face_analyser",
        types.SimpleNamespace(
            get_one_face=lambda *_args, **_kwargs: None,
            get_many_faces=lambda *_args, **_kwargs: [],
            default_source_face=lambda *_args, **_kwargs: None,
        ),
    )
    sys.modules.setdefault("modules.typing", types.SimpleNamespace(Face=object, Frame=object))
    sys.modules.setdefault(
        "modules.utilities",
        types.SimpleNamespace(
            conditional_download=lambda *_args, **_kwargs: None,
            is_image=lambda *_args, **_kwargs: False,
            is_video=lambda *_args, **_kwargs: False,
        ),
    )
    sys.modules.setdefault(
        "modules.cluster_analysis",
        types.SimpleNamespace(find_closest_centroid=lambda *_args, **_kwargs: None),
    )
    sys.modules.setdefault(
        "modules.gpu_processing",
        types.SimpleNamespace(
            gpu_gaussian_blur=lambda image, *_args, **_kwargs: image,
            gpu_sharpen=lambda image, *_args, **_kwargs: image,
            gpu_add_weighted=lambda *_args, **_kwargs: None,
            gpu_resize=lambda image, *_args, **_kwargs: image,
            gpu_cvt_color=lambda image, *_args, **_kwargs: image,
        ),
    )


def _load_face_swapper():
    _install_import_stubs()
    sys.modules.pop("modules.processors.frame.face_swapper", None)
    return importlib.import_module("modules.processors.frame.face_swapper")


class FaceSwapperInvalidFrameTests(unittest.TestCase):
    def test_swap_face_returns_none_frame_without_loading_model(self):
        face_swapper = _load_face_swapper()

        with patch.object(
            face_swapper,
            "get_face_swapper",
            side_effect=AssertionError("should not load model for invalid frame"),
        ):
            self.assertIsNone(face_swapper.swap_face(object(), object(), None))

    def test_swap_face_returns_object_without_array_shape(self):
        face_swapper = _load_face_swapper()
        invalid_frame = object()

        with patch.object(
            face_swapper,
            "get_face_swapper",
            side_effect=AssertionError("should not load model for invalid frame"),
        ):
            self.assertIs(face_swapper.swap_face(object(), object(), invalid_frame), invalid_frame)

    def test_swap_face_returns_one_dimensional_frame_without_copying(self):
        face_swapper = _load_face_swapper()
        invalid_frame = types.SimpleNamespace(
            shape=(3,),
            dtype=object(),
            copy=lambda: (_ for _ in ()).throw(AssertionError("should not copy invalid frame")),
        )

        with patch.object(
            face_swapper,
            "get_face_swapper",
            side_effect=AssertionError("should not load model for invalid frame"),
        ):
            self.assertIs(face_swapper.swap_face(object(), object(), invalid_frame), invalid_frame)


if __name__ == "__main__":
    unittest.main()
