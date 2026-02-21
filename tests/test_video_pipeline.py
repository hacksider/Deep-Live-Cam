"""Tests for video pipeline performance improvements (issues #9, #10)."""

import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest


class TestJPEGIntermediateFrames:
    """Issue #9: Verify JPEG format is used for intermediate video frames."""

    def test_extract_frames_uses_jpg_extension(self):
        """extract_frames should write frames as .jpg files."""
        from modules import utilities

        with patch.object(utilities, "run_ffmpeg") as mock_ffmpeg, \
             patch.object(utilities, "get_temp_directory_path", return_value="/tmp/test"):
            utilities.extract_frames("/fake/video.mp4")
            args = mock_ffmpeg.call_args[0][0]
            # Find the output path argument (last positional arg to ffmpeg)
            output_pattern = [a for a in args if "%04d" in a][0]
            assert output_pattern.endswith(".jpg"), f"Expected .jpg output, got {output_pattern}"
            # Verify JPEG quality flag is present
            assert "-qscale:v" in args, "Missing -qscale:v flag for JPEG quality"

    def test_get_temp_frame_paths_globs_jpg(self):
        """get_temp_frame_paths should glob for .jpg files."""
        from modules import utilities

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for ext in [".jpg", ".bmp", ".png"]:
                open(os.path.join(tmpdir, f"0001{ext}"), "w").close()

            with patch.object(utilities, "get_temp_directory_path", return_value=tmpdir):
                paths = utilities.get_temp_frame_paths("/fake/video.mp4")
                assert len(paths) == 1
                assert paths[0].endswith(".jpg")

    def test_create_video_reads_jpg_frames(self):
        """create_video should read .jpg frames as input."""
        from modules import utilities
        import modules.globals

        modules.globals.video_encoder = "libx264"
        modules.globals.video_quality = 18
        modules.globals.execution_providers = ["CPUExecutionProvider"]

        with patch.object(utilities, "run_ffmpeg") as mock_ffmpeg, \
             patch.object(utilities, "get_temp_output_path", return_value="/tmp/test/output.mp4"), \
             patch.object(utilities, "get_temp_directory_path", return_value="/tmp/test"):
            utilities.create_video("/fake/video.mp4", fps=30.0)
            args = mock_ffmpeg.call_args[0][0]
            input_pattern = [a for a in args if isinstance(a, str) and "%04d" in a][0]
            assert input_pattern.endswith(".jpg"), f"Expected .jpg input, got {input_pattern}"

    def test_face_swapper_writes_jpeg_quality(self):
        """face_swapper process_frames should write with JPEG quality 95."""
        import cv2
        from unittest.mock import ANY

        with patch("cv2.imread") as mock_read, \
             patch("cv2.imwrite") as mock_write, \
             patch("modules.processors.frame.face_swapper.get_one_face") as mock_face, \
             patch("modules.processors.frame.face_swapper.process_frame") as mock_proc, \
             patch("modules.processors.frame.face_swapper.update_status"):
            import numpy as np
            fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_read.return_value = fake_frame
            mock_face.return_value = MagicMock()
            mock_proc.return_value = fake_frame

            from modules.processors.frame.face_swapper import process_frames
            # Set up globals for simple mode
            import modules.globals
            modules.globals.map_faces = False
            modules.globals.source_path = "/fake/source.jpg"

            process_frames("/fake/source.jpg", ["/tmp/0001.jpg"], progress=None)

            if mock_write.called:
                call_args = mock_write.call_args
                # Check JPEG quality params are passed
                assert call_args[0][0] == "/tmp/0001.jpg"
                assert [cv2.IMWRITE_JPEG_QUALITY, 95] == call_args[0][2]


class TestProcessPoolExecutor:
    """Issue #10: Verify ProcessPoolExecutor is used for video batch mode."""

    def test_core_imports_process_pool_executor(self):
        """core module should have ProcessPoolExecutor available."""
        from modules.processors.frame import core
        assert hasattr(core, "ProcessPoolExecutor")

    def test_multi_process_frame_uses_process_pool(self):
        """multi_process_frame should instantiate ProcessPoolExecutor."""
        import modules.processors.frame.core as core
        import modules.globals

        modules.globals.execution_threads = 1

        # Patch at module level to intercept the class
        original_ppe = core.ProcessPoolExecutor
        calls = []

        class FakeProcessPoolExecutor:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))
                self._executor = original_ppe(max_workers=1)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self._executor.shutdown(wait=True)
                return False

            def submit(self, fn, *args, **kwargs):
                from concurrent.futures import Future
                f = Future()
                f.set_result(None)
                return f

        core.ProcessPoolExecutor = FakeProcessPoolExecutor
        try:
            core.multi_process_frame("/fake/src.jpg", ["/tmp/0001.jpg"], lambda *a: None, progress=None)
            assert len(calls) == 1, "ProcessPoolExecutor should be instantiated once"
        finally:
            core.ProcessPoolExecutor = original_ppe

    def test_multi_process_frame_live_exists_and_uses_threads(self):
        """multi_process_frame_live should exist and use ThreadPoolExecutor."""
        from modules.processors.frame import core
        assert hasattr(core, "multi_process_frame_live"), "multi_process_frame_live must exist for live mode"

        import modules.globals
        modules.globals.execution_threads = 1

        original_tpe = core.ThreadPoolExecutor
        calls = []

        class FakeThreadPoolExecutor:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, fn, *args, **kwargs):
                from concurrent.futures import Future
                f = Future()
                f.set_result(None)
                return f

        core.ThreadPoolExecutor = FakeThreadPoolExecutor
        try:
            core.multi_process_frame_live("/fake/src.jpg", ["/tmp/0001.jpg"], lambda *a: None, progress=None)
            assert len(calls) == 1, "ThreadPoolExecutor should be instantiated once"
        finally:
            core.ThreadPoolExecutor = original_tpe

    def test_process_video_delegates_to_multi_process_frame(self):
        """process_video should call multi_process_frame (ProcessPoolExecutor path)."""
        from modules.processors.frame import core

        called_with = []
        original = core.multi_process_frame
        core.multi_process_frame = lambda *args, **kwargs: called_with.append(args)

        try:
            core.process_video("/src.jpg", ["/tmp/f1.jpg"], lambda *a: None)
            assert len(called_with) == 1
        finally:
            core.multi_process_frame = original
