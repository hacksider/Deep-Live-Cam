import importlib
import sys
import types
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch


@contextmanager
def _isolated_module_import(stubs, module_name):
    # Importlib writes child modules onto parent packages independently of
    # sys.modules. Restore both, including globals changed by each scenario.
    package_state = [
        (module, dict(vars(module)))
        for name, module in list(sys.modules.items())
        if (name == "modules" or name.startswith("modules."))
        and isinstance(module, types.ModuleType)
        and hasattr(module, "__path__")
    ]
    child_names = {name.rsplit(".", 1)[-1] for name in (*stubs, module_name)}
    try:
        with ExitStack() as stack:
            globals_module = sys.modules.get("modules.globals")
            if globals_module is not None:
                stack.enter_context(patch.dict(vars(globals_module)))
            stack.enter_context(patch.dict(sys.modules, stubs, clear=False))
            for name, stub in stubs.items():
                parent_name, separator, child_name = name.rpartition(".")
                parent = sys.modules.get(parent_name) if separator else None
                if parent is not None:
                    setattr(parent, child_name, stub)
            sys.modules.pop(module_name, None)
            yield importlib.import_module(module_name)
    finally:
        missing = object()
        for package, original in package_state:
            for name in set(vars(package)) | set(original):
                current = vars(package).get(name, missing)
                previous = original.get(name, missing)
                if current is previous:
                    continue
                if (
                    isinstance(current, types.ModuleType)
                    or isinstance(previous, types.ModuleType)
                    or name in child_names
                ):
                    if previous is missing:
                        delattr(package, name)
                    else:
                        setattr(package, name, previous)


@contextmanager
def _patched_core_import_stubs(calls, pipe_result=False):
    class Processor:
        NAME = "test_processor"

        def pre_start(self):
            return True

        def pre_check(self):
            return True

        def process_image(self, *_args, **_kwargs):
            raise AssertionError("image path should not be used")

        def process_video(self, source_path, frame_paths):
            calls.append(("process_video", source_path, tuple(frame_paths)))

    stubs = {
        "cv2": types.SimpleNamespace(
            IMREAD_COLOR=1,
            imdecode=lambda *_args, **_kwargs: None,
            imencode=lambda *_args, **_kwargs: (
                True,
                types.SimpleNamespace(tofile=lambda *_a, **_k: None),
            ),
        ),
        "numpy": types.SimpleNamespace(uint8=object, fromfile=lambda *_args, **_kwargs: b""),
        "torch": types.SimpleNamespace(
            cuda=types.SimpleNamespace(empty_cache=lambda: None)
        ),
        "onnxruntime": types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"]
        ),
        "tensorflow": types.SimpleNamespace(),
        "modules.metadata": types.SimpleNamespace(name="Deep-Live-Cam", version="test"),
        "modules.ui": types.SimpleNamespace(
            check_and_ignore_nsfw=lambda *_args, **_kwargs: False,
            update_status=lambda *_args, **_kwargs: None,
            init=lambda *_args, **_kwargs: types.SimpleNamespace(mainloop=lambda: None),
        ),
        "modules.processors.frame.core": types.SimpleNamespace(
            get_frame_processors_modules=lambda _names: [Processor()],
            process_video_in_memory=lambda *_args, **_kwargs: calls.append(("pipe",))
            or pipe_result,
        ),
        "modules.utilities": types.SimpleNamespace(
            has_image_extension=lambda _path: False,
            is_image=lambda _path: False,
            is_video=lambda _path: True,
            detect_fps=lambda _path: 24.0,
            create_video=lambda target_path, fps: calls.append(
                ("create_video", target_path, fps)
            )
            or True,
            extract_frames=lambda target_path: calls.append(
                ("extract_frames", target_path)
            ),
            get_temp_frame_paths=lambda target_path: [f"{target_path}/0001.png"],
            restore_audio=lambda *_args, **_kwargs: calls.append(("restore_audio",)),
            create_temp=lambda target_path: calls.append(("create_temp", target_path)),
            move_temp=lambda target_path, output_path: calls.append(
                ("move_temp", target_path, output_path)
            ),
            clean_temp=lambda target_path: calls.append(("clean_temp", target_path)),
            normalize_output_path=lambda _source, _target, output: output,
        ),
    }
    with _isolated_module_import(stubs, "modules.core") as core:
        yield core


def _configure_video_run(core, *, map_faces):
    core.modules.globals.source_path = "source.jpg"
    core.modules.globals.target_path = "target.mp4"
    core.modules.globals.output_path = "output.mp4"
    core.modules.globals.frame_processors = ["face_swapper"]
    core.modules.globals.headless = True
    core.modules.globals.keep_fps = False
    core.modules.globals.keep_audio = False
    core.modules.globals.keep_frames = False
    core.modules.globals.map_faces = map_faces
    core.modules.globals.nsfw_filter = False
    core.modules.globals.execution_threads = 1
    core.modules.globals.execution_providers = ["CPUExecutionProvider"]
    core.modules.globals.max_memory = None


class MapFacesFallbackTests(unittest.TestCase):
    def test_import_stubs_restore_module_and_parent_references_on_exception(self):
        package = importlib.import_module("modules")
        original_core = types.ModuleType("modules.core")
        original_ui = types.ModuleType("modules.ui")
        with patch.dict(sys.modules, {
            "modules.core": original_core, "modules.ui": original_ui,
        }), patch.object(package, "core", original_core, create=True), patch.object(
            package, "ui", original_ui, create=True
        ):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                with _patched_core_import_stubs([]) as core:
                    self.assertIs(package.core, core)
                    self.assertIsNot(package.ui, original_ui)
                    raise RuntimeError("test failure")
            self.assertIs(sys.modules["modules.core"], original_core)
            self.assertIs(sys.modules["modules.ui"], original_ui)
            self.assertIs(package.core, original_core)
            self.assertIs(package.ui, original_ui)

    def test_map_faces_disk_fallback_extracts_frames_before_processing(self):
        calls = []
        with _patched_core_import_stubs(calls, pipe_result=False) as core:
            _configure_video_run(core, map_faces=True)

            with patch.object(core.os.path, "isfile", return_value=True):
                core.start()

        self.assertNotIn(("pipe",), calls)
        self.assertIn(("create_temp", "target.mp4"), calls)
        self.assertIn(("extract_frames", "target.mp4"), calls)
        self.assertIn(("process_video", "source.jpg", ("target.mp4/0001.png",)), calls)
        self.assertIn(("create_video", "target.mp4", 30.0), calls)
        self.assertIn(("move_temp", "target.mp4", "output.mp4"), calls)

        step_indices = {}
        for index, call in enumerate(calls):
            step_indices.setdefault(call[0], index)

        self.assertLess(step_indices["create_temp"], step_indices["extract_frames"])
        self.assertLess(step_indices["extract_frames"], step_indices["process_video"])
        self.assertLess(step_indices["process_video"], step_indices["create_video"])
        self.assertLess(step_indices["create_video"], step_indices["move_temp"])

    def test_non_map_faces_pipe_success_does_not_extract_frames(self):
        calls = []
        with _patched_core_import_stubs(calls, pipe_result=True) as core:
            _configure_video_run(core, map_faces=False)

            with patch.object(core.os.path, "isfile", return_value=True):
                core.start()

        self.assertIn(("pipe",), calls)
        self.assertNotIn(("extract_frames", "target.mp4"), calls)
        self.assertNotIn(("process_video", "source.jpg", ("target.mp4/0001.png",)), calls)


if __name__ == "__main__":
    unittest.main()
