import os
import cv2
import numpy as np
import pytest

import modules.globals
from modules.processors.frame.core import load_frame_processor_module, ALLOWED_PROCESSORS
import modules.processors.frame.background_swapper as bg_swapper


def test_background_swapper_registration():
    assert "background_swapper" in ALLOWED_PROCESSORS
    module = load_frame_processor_module("background_swapper")
    assert module is not None
    assert hasattr(module, "pre_check")
    assert hasattr(module, "pre_start")
    assert hasattr(module, "process_frame")
    assert hasattr(module, "process_image")
    assert hasattr(module, "process_video")


def test_swap_background_noop_when_not_configured():
    # If no background path is set and blur is 0, frame should return untouched
    modules.globals.background_path = None
    modules.globals.background_blur = 0.0

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = (120, 50, 20)

    result = bg_swapper.swap_background(frame)
    assert np.array_equal(result, frame)


def test_swap_background_blur(tmp_path):
    modules.globals.background_path = None
    modules.globals.background_blur = 50.0

    # Create synthetic frame with sharp edges
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:50, :50] = (255, 255, 255)

    result = bg_swapper.swap_background(frame)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    # Background blur should modify the pixel values around edges/background
    assert not np.array_equal(result, frame)


def test_swap_background_custom_image(tmp_path):
    # Create a custom background image
    bg_file = str(tmp_path / "custom_bg.png")
    bg_img = np.full((120, 120, 3), (0, 255, 0), dtype=np.uint8)
    cv2.imwrite(bg_file, bg_img)

    modules.globals.background_path = bg_file
    modules.globals.background_blur = 0.0
    modules.globals.background_feather = 5.0

    frame = np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8)

    result = bg_swapper.swap_background(frame)
    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    # Blended output should contain green from background in background areas
    assert not np.array_equal(result, frame)


def test_process_frame_and_process_frame_v2(tmp_path):
    bg_file = str(tmp_path / "bg.png")
    bg_img = np.full((50, 50, 3), (100, 100, 100), dtype=np.uint8)
    cv2.imwrite(bg_file, bg_img)

    modules.globals.background_path = bg_file
    frame = np.full((50, 50, 3), (200, 200, 200), dtype=np.uint8)

    res1 = bg_swapper.process_frame(None, frame)
    res2 = bg_swapper.process_frame_v2(frame)

    assert res1.shape == frame.shape
    assert res2.shape == frame.shape
    assert np.array_equal(res1, res2)


def test_process_image(tmp_path):
    target_path = str(tmp_path / "target.png")
    out_path = str(tmp_path / "out.png")
    bg_path = str(tmp_path / "bg.png")

    target_img = np.full((60, 60, 3), (50, 50, 50), dtype=np.uint8)
    bg_img = np.full((60, 60, 3), (200, 100, 50), dtype=np.uint8)
    cv2.imwrite(target_path, target_img)
    cv2.imwrite(bg_path, bg_img)

    modules.globals.background_path = bg_path
    bg_swapper.process_image(None, target_path, out_path)

    assert os.path.exists(out_path)
    loaded_out = cv2.imread(out_path)
    assert loaded_out is not None
    assert loaded_out.shape == target_img.shape
