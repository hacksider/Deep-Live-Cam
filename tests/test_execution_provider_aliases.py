import importlib
import sys
import types
import unittest


def _install_import_stubs():
    sys.modules["torch"] = types.SimpleNamespace()
    sys.modules["tensorflow"] = types.SimpleNamespace()
    sys.modules["onnxruntime"] = types.SimpleNamespace(
        get_available_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    sys.modules["modules.ui"] = types.SimpleNamespace()
    sys.modules["modules.processors.frame.core"] = types.SimpleNamespace(
        get_frame_processors_modules=lambda *_args, **_kwargs: [],
        process_video_in_memory=lambda *_args, **_kwargs: None,
    )
    sys.modules["modules.utilities"] = types.SimpleNamespace(
        has_image_extension=lambda *_args, **_kwargs: False,
        is_image=lambda *_args, **_kwargs: False,
        is_video=lambda *_args, **_kwargs: False,
        detect_fps=lambda *_args, **_kwargs: 0,
        create_video=lambda *_args, **_kwargs: None,
        extract_frames=lambda *_args, **_kwargs: None,
        get_temp_frame_paths=lambda *_args, **_kwargs: [],
        restore_audio=lambda *_args, **_kwargs: None,
        create_temp=lambda *_args, **_kwargs: None,
        move_temp=lambda *_args, **_kwargs: None,
        clean_temp=lambda *_args, **_kwargs: None,
        normalize_output_path=lambda *_args, **_kwargs: None,
    )


def _load_core():
    _install_import_stubs()
    sys.modules.pop("modules.core", None)
    return importlib.import_module("modules.core")


class ExecutionProviderAliasTests(unittest.TestCase):
    def test_directml_alias_decodes_to_dml_provider(self):
        core = _load_core()

        self.assertEqual(
            core.decode_execution_providers(["directml"]),
            ["DmlExecutionProvider"],
        )

    def test_directml_is_suggested_when_dml_is_available(self):
        core = _load_core()

        self.assertIn("dml", core.suggest_execution_providers())
        self.assertIn("directml", core.suggest_execution_providers())


if __name__ == "__main__":
    unittest.main()
