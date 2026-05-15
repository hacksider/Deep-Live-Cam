import importlib
import sys
import types
import unittest
from unittest.mock import patch


def _install_core_import_stubs(calls):
    sys.modules.setdefault("torch", types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)))
    sys.modules["onnxruntime"] = types.SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"]
    )
    sys.modules.setdefault("tensorflow", types.SimpleNamespace())
    sys.modules["modules.metadata"] = types.SimpleNamespace(name="Deep-Live-Cam", version="test")
    sys.modules["modules.ui"] = types.SimpleNamespace(
        check_and_ignore_nsfw=lambda *_args, **_kwargs: False,
        update_status=lambda *_args, **_kwargs: None,
        init=lambda *_args, **_kwargs: types.SimpleNamespace(mainloop=lambda: None),
    )

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

    sys.modules["modules.processors.frame.core"] = types.SimpleNamespace(
        get_frame_processors_modules=lambda _names: [Processor()],
        process_video_in_memory=lambda *_args, **_kwargs: calls.append(("pipe",)) or False,
    )
    sys.modules["modules.utilities"] = types.SimpleNamespace(
        has_image_extension=lambda _path: False,
        is_image=lambda _path: False,
        is_video=lambda _path: True,
        detect_fps=lambda _path: 24.0,
        create_video=lambda target_path, fps: calls.append(("create_video", target_path, fps)) or True,
        extract_frames=lambda target_path: calls.append(("extract_frames", target_path)),
        get_temp_frame_paths=lambda target_path: [f"{target_path}/0001.png"],
        restore_audio=lambda *_args, **_kwargs: calls.append(("restore_audio",)),
        create_temp=lambda target_path: calls.append(("create_temp", target_path)),
        move_temp=lambda target_path, output_path: calls.append(("move_temp", target_path, output_path)),
        clean_temp=lambda target_path: calls.append(("clean_temp", target_path)),
        normalize_output_path=lambda _source, _target, output: output,
    )


def _load_core(calls):
    _install_core_import_stubs(calls)
    sys.modules.pop("modules.core", None)
    return importlib.import_module("modules.core")


class MapFacesFallbackTests(unittest.TestCase):
    def test_map_faces_disk_fallback_extracts_frames_before_processing(self):
        calls = []
        core = _load_core(calls)

        core.modules.globals.source_path = "source.jpg"
        core.modules.globals.target_path = "target.mp4"
        core.modules.globals.output_path = "output.mp4"
        core.modules.globals.frame_processors = ["face_swapper"]
        core.modules.globals.headless = True
        core.modules.globals.keep_fps = False
        core.modules.globals.keep_audio = False
        core.modules.globals.keep_frames = False
        core.modules.globals.map_faces = True
        core.modules.globals.nsfw_filter = False
        core.modules.globals.execution_threads = 1
        core.modules.globals.execution_providers = ["CPUExecutionProvider"]
        core.modules.globals.max_memory = None

        with patch.object(core.os.path, "isfile", return_value=True):
            core.start()

        self.assertNotIn(("pipe",), calls)
        self.assertIn(("create_temp", "target.mp4"), calls)
        self.assertIn(("extract_frames", "target.mp4"), calls)
        self.assertIn(("process_video", "source.jpg", ("target.mp4/0001.png",)), calls)

        extract_index = calls.index(("extract_frames", "target.mp4"))
        process_index = calls.index(("process_video", "source.jpg", ("target.mp4/0001.png",)))
        self.assertLess(extract_index, process_index)


if __name__ == "__main__":
    unittest.main()
