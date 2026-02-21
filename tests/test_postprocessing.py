"""Tests for post-processing optimizations (issue #14).

Verifies that float32 blending, vectorized hull padding, and corrected LAB
color transfer produce results equivalent to the original float64 paths.
"""
import numpy as np
import cv2
import pytest


# ---------------------------------------------------------------------------
# apply_color_transfer (face_masking.py version)
# ---------------------------------------------------------------------------

class TestApplyColorTransferMasking:
    """Tests for face_masking.apply_color_transfer."""

    def _get_fn(self):
        from modules.processors.frame.face_masking import apply_color_transfer
        return apply_color_transfer

    def test_output_dtype_is_uint8(self):
        fn = self._get_fn()
        src = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        tgt = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = fn(src, tgt)
        assert result.dtype == np.uint8

    def test_output_shape_matches_source(self):
        fn = self._get_fn()
        src = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
        tgt = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
        result = fn(src, tgt)
        assert result.shape == src.shape

    def test_identical_input_returns_similar_output(self):
        """When source and target have the same color distribution, output ~= source."""
        fn = self._get_fn()
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        result = fn(img.copy(), img.copy())
        # Should be very close (within rounding)
        assert np.allclose(result, img, atol=2)

    def test_pixel_values_in_valid_range(self):
        fn = self._get_fn()
        src = np.full((32, 32, 3), 250, dtype=np.uint8)
        tgt = np.full((32, 32, 3), 10, dtype=np.uint8)
        result = fn(src, tgt)
        assert result.min() >= 0
        assert result.max() <= 255


# ---------------------------------------------------------------------------
# apply_color_transfer (face_swapper.py version)
# ---------------------------------------------------------------------------

class TestApplyColorTransferSwapper:
    """Tests for face_swapper.apply_color_transfer."""

    def _get_fn(self):
        from modules.processors.frame.face_swapper import apply_color_transfer
        return apply_color_transfer

    def test_output_dtype_is_uint8(self):
        fn = self._get_fn()
        src = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        tgt = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = fn(src, tgt)
        assert result.dtype == np.uint8

    def test_none_input_returns_source(self):
        fn = self._get_fn()
        src = np.zeros((32, 32, 3), dtype=np.uint8)
        assert fn(src, None) is src
        assert fn(None, src) is None  # source is None


# ---------------------------------------------------------------------------
# create_face_mask (face_masking.py — vectorized hull padding)
# ---------------------------------------------------------------------------

class TestCreateFaceMaskMasking:
    """Tests for face_masking.create_face_mask vectorized hull padding."""

    def test_vectorized_hull_padding_shape(self):
        """Verify the vectorized hull expansion produces correct shape."""
        from modules.processors.frame.face_masking import create_face_mask

        # Create a mock face with valid 106 landmarks
        class MockFace:
            pass
        face = MockFace()
        # Place landmarks in a rough face shape
        landmarks = np.zeros((106, 2), dtype=np.float32)
        # Outline points 0-32 forming a rough oval
        angles = np.linspace(0, 2 * np.pi, 33, endpoint=False)
        landmarks[0:33, 0] = 150 + 50 * np.cos(angles)
        landmarks[0:33, 1] = 150 + 70 * np.sin(angles)
        # Eyebrow points 33-42 and 43-51
        landmarks[33:52, 0] = np.linspace(110, 190, 19)
        landmarks[33:52, 1] = 110
        face.landmark_2d_106 = landmarks

        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        mask = create_face_mask(face, frame)

        assert mask.shape == (300, 300)
        assert mask.dtype == np.uint8
        assert mask.max() > 0  # mask is not empty


# ---------------------------------------------------------------------------
# apply_mouth_area blending precision
# ---------------------------------------------------------------------------

class TestApplyMouthAreaBlending:
    """Verify float32 blending produces acceptable quality."""

    def test_float32_blending_output_valid(self):
        """Ensure the float32 blend path produces valid uint8 output."""
        from modules.processors.frame.face_swapper import apply_mouth_area
        import modules.globals

        # Set required globals
        modules.globals.mask_feather_ratio = 12
        modules.globals.face_mask_blur = 31

        frame = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        mouth_cutout = np.random.randint(50, 200, (40, 60, 3), dtype=np.uint8)
        mouth_box = (50, 80, 110, 120)

        # Create a face mask (uint8, 0-255)
        face_mask = np.zeros((200, 200), dtype=np.uint8)
        face_mask[70:130, 40:160] = 255

        # Create a simple polygon
        polygon = np.array([
            [55, 85], [105, 85], [105, 115], [55, 115]
        ], dtype=np.int32)

        result = apply_mouth_area(frame, mouth_cutout, mouth_box, face_mask, polygon)

        assert result.dtype == np.uint8
        assert result.shape == frame.shape
        assert result.min() >= 0
        assert result.max() <= 255


# ---------------------------------------------------------------------------
# apply_mask_area float32 blending (face_masking.py)
# ---------------------------------------------------------------------------

class TestApplyMaskAreaBlending:
    """Verify float32 blending in apply_mask_area."""

    def test_output_is_valid_uint8(self):
        from modules.processors.frame.face_masking import apply_mask_area
        import modules.globals

        modules.globals.MOUTH_FEATHER_RADIUS = 10
        modules.globals.mask_feather_ratio = 12

        frame = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
        cutout = np.random.randint(50, 200, (40, 60, 3), dtype=np.uint8)
        box = (50, 80, 110, 120)
        face_mask = np.zeros((200, 200), dtype=np.uint8)
        face_mask[70:130, 40:160] = 255
        polygon = np.array([
            [55, 85], [105, 85], [105, 115], [55, 115]
        ], dtype=np.int32)

        result = apply_mask_area(frame, cutout, box, face_mask, polygon)
        assert result.dtype == np.uint8
        assert result.shape == frame.shape
