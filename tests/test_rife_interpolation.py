"""Tests for the RIFE frame interpolation module."""
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

import modules.globals
from modules.rife_interpolation import (
    _binary_name,
    _build_command,
    _count_frames,
    _get_backend,
    find_binary,
    find_model_dir,
    has_native_binding,
    pre_check,
    interpolate_frames,
    RIFE_DIR,
    DEFAULT_MODEL,
)


# ---------------------------------------------------------------------------
# Binary name
# ---------------------------------------------------------------------------


class TestBinaryName:
    def test_linux_binary_name(self):
        with patch.object(sys, "platform", "linux"):
            assert _binary_name() == "rife-ncnn-vulkan"

    def test_darwin_binary_name(self):
        with patch.object(sys, "platform", "darwin"):
            assert _binary_name() == "rife-ncnn-vulkan"

    def test_windows_binary_name(self):
        with patch.object(sys, "platform", "win32"):
            assert _binary_name() == "rife-ncnn-vulkan.exe"


# ---------------------------------------------------------------------------
# Binary detection
# ---------------------------------------------------------------------------


class TestFindBinary:
    def test_finds_on_path(self):
        with patch("modules.rife_interpolation.shutil.which", return_value="/usr/bin/rife-ncnn-vulkan"):
            assert find_binary() == "/usr/bin/rife-ncnn-vulkan"

    def test_finds_in_models_dir(self, tmp_path):
        binary_path = tmp_path / "rife-ncnn-vulkan"
        binary_path.touch()
        binary_path.chmod(0o755)

        with patch("modules.rife_interpolation.shutil.which", return_value=None), \
             patch("modules.rife_interpolation.RIFE_DIR", str(tmp_path)):
            result = find_binary()
            assert result == str(binary_path)

    def test_returns_none_when_not_found(self):
        with patch("modules.rife_interpolation.shutil.which", return_value=None), \
             patch("modules.rife_interpolation.RIFE_DIR", "/nonexistent/path"):
            assert find_binary() is None


# ---------------------------------------------------------------------------
# Model dir detection
# ---------------------------------------------------------------------------


class TestFindModelDir:
    def test_finds_in_rife_dir(self, tmp_path):
        model_dir = tmp_path / "rife-v4.25-lite"
        model_dir.mkdir()

        with patch("modules.rife_interpolation.RIFE_DIR", str(tmp_path)):
            assert find_model_dir("rife-v4.25-lite") == str(model_dir)

    def test_finds_in_models_dir(self, tmp_path):
        model_dir = tmp_path / "rife-v4.25"
        model_dir.mkdir()

        with patch("modules.rife_interpolation.RIFE_DIR", "/nonexistent"), \
             patch("modules.rife_interpolation.MODELS_DIR", str(tmp_path)):
            assert find_model_dir("rife-v4.25") == str(model_dir)

    def test_returns_none_when_not_found(self):
        with patch("modules.rife_interpolation.RIFE_DIR", "/nonexistent"), \
             patch("modules.rife_interpolation.MODELS_DIR", "/also_nonexistent"):
            assert find_model_dir("rife-v4.25") is None


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


class TestBackendDetection:
    def test_native_preferred_over_cli(self):
        with patch("modules.rife_interpolation.has_native_binding", return_value=True), \
             patch("modules.rife_interpolation.find_binary", return_value="/usr/bin/rife"):
            assert _get_backend() == "native"

    def test_cli_fallback_when_no_native(self):
        with patch("modules.rife_interpolation.has_native_binding", return_value=False), \
             patch("modules.rife_interpolation.find_binary", return_value="/usr/bin/rife"):
            assert _get_backend() == "cli"

    def test_none_when_nothing_available(self):
        with patch("modules.rife_interpolation.has_native_binding", return_value=False), \
             patch("modules.rife_interpolation.find_binary", return_value=None):
            assert _get_backend() == "none"

    def test_has_native_binding_returns_false_without_package(self):
        # The package is not installed in test env
        assert has_native_binding() is False


# ---------------------------------------------------------------------------
# Pre-check
# ---------------------------------------------------------------------------


class TestPreCheck:
    def test_passes_when_disabled(self):
        modules.globals.rife_enabled = False
        assert pre_check() is True

    def test_fails_when_no_backend(self):
        modules.globals.rife_enabled = True
        with patch("modules.rife_interpolation._get_backend", return_value="none"):
            assert pre_check() is False
        modules.globals.rife_enabled = False

    def test_passes_with_native_backend(self):
        modules.globals.rife_enabled = True
        with patch("modules.rife_interpolation._get_backend", return_value="native"):
            assert pre_check() is True
        modules.globals.rife_enabled = False

    def test_passes_with_cli_backend_and_model(self):
        modules.globals.rife_enabled = True
        modules.globals.rife_model = "rife-v4.25-lite"
        with patch("modules.rife_interpolation._get_backend", return_value="cli"), \
             patch("modules.rife_interpolation.find_model_dir", return_value="/models/rife-v4.25-lite"):
            assert pre_check() is True
        modules.globals.rife_enabled = False

    def test_fails_with_cli_backend_no_model(self):
        modules.globals.rife_enabled = True
        modules.globals.rife_model = "rife-v4.25-lite"
        with patch("modules.rife_interpolation._get_backend", return_value="cli"), \
             patch("modules.rife_interpolation.find_model_dir", return_value=None):
            assert pre_check() is False
        modules.globals.rife_enabled = False


# ---------------------------------------------------------------------------
# Frame counting
# ---------------------------------------------------------------------------


class TestCountFrames:
    def test_counts_jpg_files(self, tmp_path):
        for i in range(5):
            (tmp_path / f"{i:04d}.jpg").touch()
        assert _count_frames(str(tmp_path)) == 5

    def test_counts_png_files(self, tmp_path):
        for i in range(3):
            (tmp_path / f"{i:04d}.png").touch()
        assert _count_frames(str(tmp_path)) == 3

    def test_counts_mixed_formats(self, tmp_path):
        (tmp_path / "0001.jpg").touch()
        (tmp_path / "0002.png").touch()
        (tmp_path / "0003.jpeg").touch()
        assert _count_frames(str(tmp_path)) == 3

    def test_empty_directory(self, tmp_path):
        assert _count_frames(str(tmp_path)) == 0


# ---------------------------------------------------------------------------
# CLI command building
# ---------------------------------------------------------------------------


class TestBuildCommand:
    def test_basic_command(self):
        cmd = _build_command(
            binary="/usr/bin/rife-ncnn-vulkan",
            input_dir="/tmp/frames",
            output_dir="/tmp/output",
            model_dir="/models/rife-v4.25-lite",
            input_frame_count=100,
            multiplier=2,
        )
        assert cmd[0] == "/usr/bin/rife-ncnn-vulkan"
        assert "-i" in cmd
        assert cmd[cmd.index("-i") + 1] == "/tmp/frames"
        assert "-o" in cmd
        assert cmd[cmd.index("-o") + 1] == "/tmp/output"
        assert "-m" in cmd
        assert cmd[cmd.index("-m") + 1] == "/models/rife-v4.25-lite"
        assert "-n" in cmd
        assert cmd[cmd.index("-n") + 1] == "200"  # 100 * 2
        assert "-f" in cmd
        assert cmd[cmd.index("-f") + 1] == "%04d.jpg"

    def test_4x_multiplier(self):
        cmd = _build_command(
            binary="rife-ncnn-vulkan",
            input_dir="/tmp/in",
            output_dir="/tmp/out",
            model_dir="/models/rife-v4.25",
            input_frame_count=50,
            multiplier=4,
        )
        assert cmd[cmd.index("-n") + 1] == "200"  # 50 * 4


# ---------------------------------------------------------------------------
# interpolate_frames dispatch
# ---------------------------------------------------------------------------


class TestInterpolateFramesDispatch:
    def test_dispatches_to_native(self, tmp_path):
        # Create dummy frames
        for i in range(3):
            (tmp_path / f"{i+1:04d}.jpg").touch()

        with patch("modules.rife_interpolation._get_backend", return_value="native"), \
             patch("modules.rife_interpolation._interpolate_native", return_value=5) as mock_native:
            result = interpolate_frames(str(tmp_path))
            assert result == 5
            mock_native.assert_called_once_with(str(tmp_path))

    def test_dispatches_to_cli(self, tmp_path):
        for i in range(3):
            (tmp_path / f"{i+1:04d}.jpg").touch()

        with patch("modules.rife_interpolation._get_backend", return_value="cli"), \
             patch("modules.rife_interpolation._interpolate_cli", return_value=5) as mock_cli:
            result = interpolate_frames(str(tmp_path))
            assert result == 5
            mock_cli.assert_called_once_with(str(tmp_path))

    def test_returns_none_when_no_backend(self, tmp_path):
        with patch("modules.rife_interpolation._get_backend", return_value="none"):
            result = interpolate_frames(str(tmp_path))
            assert result is None


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------


class TestGlobals:
    def test_rife_globals_exist(self):
        assert hasattr(modules.globals, "rife_enabled")
        assert hasattr(modules.globals, "rife_model")
        assert hasattr(modules.globals, "rife_multiplier")

    def test_rife_defaults(self):
        assert isinstance(modules.globals.rife_enabled, bool)
        assert isinstance(modules.globals.rife_model, str)
        assert isinstance(modules.globals.rife_multiplier, int)
        assert modules.globals.rife_model in ("rife-v4.25", "rife-v4.25-lite")
        assert modules.globals.rife_multiplier in (2, 4)
