"""TDD tests for GPEN-BFR face enhancer processors (256 and 512)."""

import importlib
from unittest.mock import patch

import numpy as np
import pytest

# Required interface methods for all frame processors
REQUIRED_METHODS = ["pre_check", "pre_start", "process_frame", "process_image", "process_video"]


@pytest.fixture(params=["face_enhancer_gpen256", "face_enhancer_gpen512"])
def gpen_module(request):
    """Load the GPEN processor module by name."""
    return importlib.import_module(f"modules.processors.frame.{request.param}")


@pytest.fixture
def mock_no_face(gpen_module):
    """Patch get_one_face to return None in the given gpen module."""
    target = f"modules.processors.frame.{gpen_module.__name__.split('.')[-1]}.get_one_face"
    with patch(target, return_value=None):
        yield


class TestModuleInterface:
    """Verify both GPEN modules implement the frame processor interface."""

    def test_has_required_methods(self, gpen_module):
        for method in REQUIRED_METHODS:
            assert hasattr(gpen_module, method), f"Missing method: {method}"
            assert callable(getattr(gpen_module, method))

    def test_has_process_frame_v2(self, gpen_module):
        assert hasattr(gpen_module, "process_frame_v2")
        assert callable(gpen_module.process_frame_v2)

    def test_has_name(self, gpen_module):
        assert hasattr(gpen_module, "NAME")
        assert isinstance(gpen_module.NAME, str)
        assert "GPEN" in gpen_module.NAME


class TestNoFacePassthrough:
    """When no face is detected, the processor must return the original frame."""

    def test_process_frame_no_face_returns_original(self, gpen_module, mock_no_face):
        """A blank frame (no face) should pass through unchanged."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = gpen_module.process_frame(None, frame)
        assert result is not None
        np.testing.assert_array_equal(result, frame)

    def test_process_frame_v2_no_face_returns_original(self, gpen_module, mock_no_face):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = gpen_module.process_frame_v2(frame)
        assert result is not None
        np.testing.assert_array_equal(result, frame)


class TestOutputProperties:
    """Verify output frame has correct shape and dtype."""

    def test_output_shape_matches_input(self, gpen_module, mock_no_face):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = gpen_module.process_frame(None, frame)
        assert result.shape == frame.shape

    def test_output_dtype_is_uint8(self, gpen_module, mock_no_face):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = gpen_module.process_frame(None, frame)
        assert result.dtype == np.uint8


class TestGpen256Specific:
    """Tests specific to the 256 variant."""

    def test_input_size_is_256(self):
        mod = importlib.import_module("modules.processors.frame.face_enhancer_gpen256")
        assert mod.INPUT_SIZE == 256

    def test_name_contains_256(self):
        mod = importlib.import_module("modules.processors.frame.face_enhancer_gpen256")
        assert "256" in mod.NAME


class TestGpen512Specific:
    """Tests specific to the 512 variant."""

    def test_input_size_is_512(self):
        mod = importlib.import_module("modules.processors.frame.face_enhancer_gpen512")
        assert mod.INPUT_SIZE == 512

    def test_name_contains_512(self):
        mod = importlib.import_module("modules.processors.frame.face_enhancer_gpen512")
        assert "512" in mod.NAME


class TestOnnxEnhancerShared:
    """Tests for the shared ONNX enhancer utilities."""

    def test_import_shared_module(self):
        mod = importlib.import_module("modules.processors.frame._onnx_enhancer")
        assert hasattr(mod, "create_onnx_session")
        assert hasattr(mod, "enhance_face_onnx")

    def test_preprocess_face_shape(self):
        from modules.processors.frame._onnx_enhancer import preprocess_face
        face_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        blob = preprocess_face(face_img, 256)
        assert blob.shape == (1, 3, 256, 256)
        assert blob.dtype == np.float32

    def test_postprocess_face_shape(self):
        from modules.processors.frame._onnx_enhancer import postprocess_face
        output = np.random.randn(1, 3, 256, 256).astype(np.float32)
        img = postprocess_face(output)
        assert img.shape == (256, 256, 3)
        assert img.dtype == np.uint8
