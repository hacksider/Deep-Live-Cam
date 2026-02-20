"""Tests for modules/processors/frame/face_enhancer.py."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def _blank_frame(h=64, w=64):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_process_frame_v2_is_callable():
    """process_frame_v2 must exist and be callable after Wave 5 fix."""
    from modules.processors.frame import face_enhancer
    assert callable(face_enhancer.process_frame_v2)


def test_process_frame_v2_returns_frame_when_no_face():
    """process_frame_v2 must return a frame even when no face is detected."""
    frame = _blank_frame()
    with patch("modules.processors.frame.face_enhancer.get_one_face", return_value=None):
        with patch("modules.processors.frame.face_enhancer.get_face_enhancer",
                   side_effect=RuntimeError("Model not loaded")):
            from modules.processors.frame import face_enhancer
            result = face_enhancer.process_frame_v2(frame)
            assert result is not None
            assert isinstance(result, np.ndarray)
            assert result.shape == frame.shape


def test_enhance_face_handles_runtime_error():
    """enhance_face must return the input frame when get_face_enhancer raises RuntimeError."""
    from modules.processors.frame import face_enhancer

    frame = _blank_frame()
    with patch("modules.processors.frame.face_enhancer.get_face_enhancer",
               side_effect=RuntimeError("Model load failed")):
        result = face_enhancer.enhance_face(frame)
        assert np.array_equal(result, frame)
