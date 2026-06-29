from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest import mock


def _load_core_module(available_providers: list[str]):
    repo_root = Path(__file__).resolve().parent.parent

    fake_modules = types.ModuleType("modules")
    fake_modules.__path__ = [str(repo_root / "modules")]

    fake_onnxruntime = types.ModuleType("onnxruntime")
    fake_onnxruntime.get_available_providers = lambda: available_providers

    fake_tensorflow = types.ModuleType("tensorflow")
    fake_tensorflow.config = types.SimpleNamespace(
        experimental=types.SimpleNamespace(
            list_physical_devices=lambda *_: [],
            set_memory_growth=lambda *_: None,
        )
    )

    fake_metadata = types.ModuleType("modules.metadata")
    fake_metadata.name = "Deep-Live-Cam"
    fake_metadata.version = "test"

    fake_ui = types.ModuleType("modules.ui")

    fake_processors = types.ModuleType("modules.processors")
    fake_processors.__path__ = [str(repo_root / "modules" / "processors")]
    fake_frame = types.ModuleType("modules.processors.frame")
    fake_frame.__path__ = [str(repo_root / "modules" / "processors" / "frame")]
    fake_frame_core = types.ModuleType("modules.processors.frame.core")
    fake_frame_core.get_frame_processors_modules = lambda *_: []
    fake_frame_core.process_video_in_memory = lambda *_, **__: None

    fake_utilities = types.ModuleType("modules.utilities")
    fake_utilities.has_image_extension = lambda *_: False
    fake_utilities.is_image = lambda *_: False
    fake_utilities.is_video = lambda *_: False
    fake_utilities.detect_fps = lambda *_: 0
    fake_utilities.create_video = lambda *_, **__: None
    fake_utilities.extract_frames = lambda *_, **__: None
    fake_utilities.get_temp_frame_paths = lambda *_: []
    fake_utilities.restore_audio = lambda *_, **__: None
    fake_utilities.create_temp = lambda *_, **__: None
    fake_utilities.move_temp = lambda *_, **__: None
    fake_utilities.clean_temp = lambda *_, **__: None
    fake_utilities.normalize_output_path = lambda source, target, output: output

    fake_globals = types.ModuleType("modules.globals")
    fake_globals.execution_providers = []
    fake_globals.fp_ui = {
        "face_enhancer": False,
        "face_enhancer_gpen256": False,
        "face_enhancer_gpen512": False,
    }

    fake_modules.globals = fake_globals
    fake_modules.metadata = fake_metadata
    fake_modules.ui = fake_ui

    for module_name in [
        "modules",
        "modules.core",
        "modules.globals",
        "modules.metadata",
        "modules.ui",
        "modules.processors",
        "modules.processors.frame",
        "modules.processors.frame.core",
        "modules.utilities",
    ]:
        sys.modules.pop(module_name, None)
    with mock.patch.dict(
        sys.modules,
        {
            "modules": fake_modules,
            "onnxruntime": fake_onnxruntime,
            "tensorflow": fake_tensorflow,
            "modules.globals": fake_globals,
            "modules.metadata": fake_metadata,
            "modules.ui": fake_ui,
            "modules.processors": fake_processors,
            "modules.processors.frame": fake_frame,
            "modules.processors.frame.core": fake_frame_core,
            "modules.utilities": fake_utilities,
        },
    ):
        return importlib.import_module("modules.core"), fake_globals


def test_default_execution_threads_follow_cuda_default_provider() -> None:
    core, globals_module = _load_core_module(
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    with (
        mock.patch.object(sys, "argv", ["run.py"]),
        mock.patch("os.cpu_count", return_value=12),
    ):
        core.parse_args()

    assert globals_module.execution_providers == ["CUDAExecutionProvider"]
    assert globals_module.execution_threads == 2


def test_default_execution_threads_follow_cpu_provider() -> None:
    core, globals_module = _load_core_module(["CPUExecutionProvider"])

    with (
        mock.patch.object(sys, "argv", ["run.py"]),
        mock.patch("os.cpu_count", return_value=12),
    ):
        core.parse_args()

    assert globals_module.execution_providers == ["CPUExecutionProvider"]
    assert globals_module.execution_threads == 10
