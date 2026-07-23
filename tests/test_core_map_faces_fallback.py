import importlib
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch


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
    with patch.dict(sys.modules, stubs, clear=False):
        sys.modules.pop("modules.core", None)
        yield importlib.import_module("modules.core")
        sys.modules.pop("modules.core", None)


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
