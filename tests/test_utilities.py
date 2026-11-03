from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest import mock


def _load_utilities_module():
    fake_globals = types.ModuleType("modules.globals")
    fake_globals.execution_threads = 0
    fake_globals.log_level = "error"
    fake_globals.execution_providers = []
    fake_globals.video_encoder = "libx264"
    fake_globals.video_quality = 23

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.IMREAD_COLOR = 1
    fake_cv2.imdecode = lambda *args, **kwargs: None
    fake_cv2.imencode = lambda *args, **kwargs: (True, types.SimpleNamespace(tofile=lambda *_: None))

    fake_numpy = types.ModuleType("numpy")
    fake_numpy.uint8 = int
    fake_numpy.fromfile = lambda *args, **kwargs: b""

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, *args, **kwargs: iterable

    with mock.patch.dict(
        sys.modules,
        {
            "modules.globals": fake_globals,
            "cv2": fake_cv2,
            "numpy": fake_numpy,
            "tqdm": fake_tqdm,
        },
        clear=False,
    ):
        sys.modules.pop("modules", None)
        sys.modules.pop("modules.utilities", None)
        return importlib.import_module("modules.utilities")


class DetectFpsTests(unittest.TestCase):
    def test_detect_fps_returns_fraction_result(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(utilities.subprocess, "check_output", return_value="30000/1001\n") as check:
            self.assertAlmostEqual(utilities.detect_fps("demo.mp4"), 30000 / 1001)

        check.assert_called_once_with(mock.ANY, encoding="utf-8")

    def test_detect_fps_falls_back_on_command_failure(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(
            utilities.subprocess,
            "check_output",
            side_effect=utilities.subprocess.CalledProcessError(1, ["ffprobe"]),
        ):
            self.assertEqual(utilities.detect_fps("demo.mp4"), 30.0)

    def test_detect_fps_falls_back_on_malformed_output(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(utilities.subprocess, "check_output", return_value="not-a-fraction"):
            self.assertEqual(utilities.detect_fps("demo.mp4"), 30.0)

    def test_detect_fps_falls_back_on_zero_denominator(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(utilities.subprocess, "check_output", return_value="30/0"):
            self.assertEqual(utilities.detect_fps("demo.mp4"), 30.0)


def _fake_ffprobe(dimensions: bytes, side_data: bytes = b"", rotate_tag: bytes = b""):
    """Answer each ffprobe call according to the entries it asked for."""

    def run(command, *args, **kwargs):
        entry = command[command.index("-show_entries") + 1]
        if entry == "stream=width,height":
            return dimensions
        if entry == "stream_side_data=rotation":
            return side_data
        return rotate_tag

    return run


class GetVideoDimensionsTests(unittest.TestCase):
    def test_unrotated_video_keeps_probed_dimensions(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(
            utilities.subprocess, "check_output", side_effect=_fake_ffprobe(b"1920x1080\n")
        ):
            self.assertEqual(utilities.get_video_dimensions("demo.mp4"), (1920, 1080))

    def test_quarter_turn_swaps_dimensions(self) -> None:
        # A portrait phone recording: a 1920x1080 codec stream plus a
        # display-matrix rotation of 90 is displayed, and decoded, as 1080x1920.
        utilities = _load_utilities_module()

        for rotation in (b"90\n", b"270\n", b"-90.000000\n"):
            with self.subTest(rotation=rotation):
                with mock.patch.object(
                    utilities.subprocess,
                    "check_output",
                    side_effect=_fake_ffprobe(b"1920x1080\n", side_data=rotation),
                ):
                    self.assertEqual(
                        utilities.get_video_dimensions("demo.mp4"), (1080, 1920)
                    )

    def test_half_turn_keeps_dimensions(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(
            utilities.subprocess,
            "check_output",
            side_effect=_fake_ffprobe(b"1920x1080\n", side_data=b"180\n"),
        ):
            self.assertEqual(utilities.get_video_dimensions("demo.mp4"), (1920, 1080))

    def test_legacy_rotate_tag_is_honoured(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(
            utilities.subprocess,
            "check_output",
            side_effect=_fake_ffprobe(b"1920x1080\n", rotate_tag=b"90\n"),
        ):
            self.assertEqual(utilities.get_video_dimensions("demo.mp4"), (1080, 1920))

    def test_trailing_side_data_field_is_tolerated(self) -> None:
        # Some ffprobe builds print an extra empty field for the side-data
        # subsection that rotated streams carry, e.g. "1920x1080x".
        utilities = _load_utilities_module()

        with mock.patch.object(
            utilities.subprocess,
            "check_output",
            side_effect=_fake_ffprobe(b"1920x1080x\n", side_data=b"90\n"),
        ):
            self.assertEqual(utilities.get_video_dimensions("demo.mp4"), (1080, 1920))


class GetVideoRotationTests(unittest.TestCase):
    def test_missing_rotation_reads_as_zero(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(
            utilities.subprocess, "check_output", side_effect=_fake_ffprobe(b"1920x1080\n")
        ):
            self.assertEqual(utilities.get_video_rotation("demo.mp4"), 0)

    def test_probe_failure_reads_as_zero(self) -> None:
        utilities = _load_utilities_module()

        with mock.patch.object(
            utilities.subprocess,
            "check_output",
            side_effect=utilities.subprocess.CalledProcessError(1, ["ffprobe"]),
        ):
            self.assertEqual(utilities.get_video_rotation("demo.mp4"), 0)


if __name__ == "__main__":
    unittest.main()
