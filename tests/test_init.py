"""Tests for modules/__init__.py — imwrite_unicode correctness."""
import os
import tempfile
import numpy as np
import pytest


def _make_frame(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_imwrite_unicode_with_extension(tmp_path):
    from modules import imwrite_unicode

    path = str(tmp_path / "frame.png")
    result = imwrite_unicode(path, _make_frame())
    assert result is True
    assert os.path.exists(path)


def test_imwrite_unicode_without_extension(tmp_path):
    """When path has no extension, .png should be used and file written."""
    from modules import imwrite_unicode

    # os.path.splitext gives no ext for a path like "frame" (no dot)
    path = str(tmp_path / "frame")
    result = imwrite_unicode(path, _make_frame())
    assert result is True
    assert os.path.exists(path)


def test_imwrite_unicode_with_jpg_extension(tmp_path):
    from modules import imwrite_unicode

    path = str(tmp_path / "frame.jpg")
    result = imwrite_unicode(path, _make_frame())
    assert result is True
    assert os.path.exists(path)
