import importlib
import sys
import types
import unittest


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
    sys.modules["modules.processors.frame._onnx_enhancer"] = types.SimpleNamespace(
        build_provider_config=lambda providers=None: list(
            providers
            if providers is not None
            else sys.modules["modules.globals"].execution_providers
        ),
    )


def _load_face_analyser():
    _install_import_stubs()
    sys.modules.pop("modules.face_analyser", None)
    return importlib.import_module("modules.face_analyser")


class FaceAnalyserProviderConfigTests(unittest.TestCase):
    def test_uses_cpu_provider_for_directml_face_analysis(self):
        face_analyser = _load_face_analyser()
        face_analyser.modules.globals.execution_providers = ["DmlExecutionProvider"]

        self.assertEqual(
            face_analyser.build_face_analyser_provider_config(),
            ["CPUExecutionProvider"],
        )

    def test_preserves_existing_provider_config_for_non_directml(self):
        face_analyser = _load_face_analyser()
        face_analyser.modules.globals.execution_providers = ["CUDAExecutionProvider"]

        self.assertEqual(
            face_analyser.build_face_analyser_provider_config(),
            ["CUDAExecutionProvider"],
        )


if __name__ == "__main__":
    unittest.main()
