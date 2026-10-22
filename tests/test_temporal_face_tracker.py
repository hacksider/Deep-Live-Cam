"""Tests for modules.temporal_face_tracker on synthetic, exactly known motion."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from insightface.app.common import Face  # noqa: E402

from modules.temporal_face_tracker import TemporalFaceTracker  # noqa: E402


def _scene(width, height, seed=0):
    """Smooth random texture: plenty of corners for LK to lock on to."""
    rng = np.random.RandomState(seed)
    noise = rng.randint(0, 256, (height, width, 3)).astype(np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


def _face(width, height, size):
    cx, cy = width / 2, height / 2
    kps = np.float32([[-0.2, -0.1], [0.2, -0.1], [0, 0.05], [-0.15, 0.25], [0.15, 0.25]])
    return Face(
        bbox=np.float32([cx - size / 2, cy - size * 0.6, cx + size / 2, cy + size * 0.6]),
        kps=kps * size + np.float32([cx, cy]),
        det_score=np.float32(0.9),
    )


def _similarity(angle, scale, shift, center):
    matrix = cv2.getRotationMatrix2D(tuple(map(float, center)), angle, scale)
    matrix[:, 2] += shift
    return matrix


def _warp(frame, matrix):
    height, width = frame.shape[:2]
    return cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)


def _apply(matrix, points):
    return points @ matrix[:, :2].T + matrix[:, 2]


class TrackingTest(unittest.TestCase):
    def test_follows_translation_at_common_resolutions(self):
        # 640x480 tracks at full resolution; 1080p downscales the face crop.
        for width, height, size in ((640, 480, 160), (1920, 1080, 480)):
            with self.subTest(resolution=(width, height)):
                frame = _scene(width, height)
                face = _face(width, height, size)
                tracker = TemporalFaceTracker()
                self.assertTrue(tracker.reset(frame, face))

                moved = tracker.update(_warp(frame, np.float32([[1, 0, 6], [0, 1, -4]])))

                self.assertIsNotNone(moved)
                np.testing.assert_allclose(moved.kps, face.kps + [6, -4], atol=0.3)
                np.testing.assert_allclose(moved.bbox, face.bbox + [6, -4, 6, -4], atol=0.3)

    def test_follows_rotation_and_scale_across_frames(self):
        frame = _scene(640, 480)
        face = _face(640, 480, 160)
        center = (face.bbox[:2] + face.bbox[2:]) / 2
        tracker = TemporalFaceTracker()
        self.assertTrue(tracker.reset(frame, face))

        for step in range(1, 4):
            matrix = _similarity(3 * step, 1 + 0.03 * step, (4 * step, 2 * step), center)
            moved = tracker.update(_warp(frame, matrix))
            self.assertIsNotNone(moved, f"lost the face at step {step}")
            np.testing.assert_allclose(moved.kps, _apply(matrix, face.kps), atol=0.5)

        # The bbox follows the center and size but stays axis-aligned.
        size = (face.bbox[2:] - face.bbox[:2]) * 1.09
        np.testing.assert_allclose(moved.bbox[2:] - moved.bbox[:2], size, atol=0.5)
        np.testing.assert_allclose(
            (moved.bbox[:2] + moved.bbox[2:]) / 2, _apply(matrix, center[None])[0], atol=0.5
        )

    def test_moves_landmarks_and_keeps_the_detection_intact(self):
        frame = _scene(640, 480)
        face = _face(640, 480, 160)
        grid = np.stack(np.meshgrid(np.linspace(-0.3, 0.3, 11), np.linspace(-0.3, 0.4, 10)), -1)
        face.landmark_2d_106 = grid.reshape(-1, 2)[:106].astype(np.float32) * 160 + [320, 240]
        face.embedding = np.ones(512, dtype=np.float32)
        original = {key: np.array(value) for key, value in face.items()}
        tracker = TemporalFaceTracker()
        self.assertTrue(tracker.reset(frame, face))

        moved = tracker.update(_warp(frame, np.float32([[1, 0, 5], [0, 1, 3]])))

        self.assertIsInstance(moved, Face)
        np.testing.assert_allclose(moved.landmark_2d_106, face.landmark_2d_106 + [5, 3], atol=0.3)
        self.assertIs(moved.embedding, face.embedding)
        self.assertEqual(moved.det_score, face.det_score)
        for key, value in original.items():
            np.testing.assert_array_equal(face[key], value, f"reset() input {key} was mutated")


class GiveUpTest(unittest.TestCase):
    """When motion cannot be estimated, update() must return None, never a
    guess: the caller then falls back to the last detection."""

    def test_featureless_face_is_not_tracked(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        tracker = TemporalFaceTracker()
        self.assertFalse(tracker.reset(frame, _face(640, 480, 160)))
        self.assertIsNone(tracker.update(frame))

    def test_scene_change_stops_tracking(self):
        tracker = TemporalFaceTracker()
        self.assertTrue(tracker.reset(_scene(640, 480, seed=0), _face(640, 480, 160)))
        self.assertIsNone(tracker.update(_scene(640, 480, seed=1)))
        self.assertFalse(tracker.active)
        # Inactive until the next detection, even if the old scene comes back.
        self.assertIsNone(tracker.update(_scene(640, 480, seed=0)))

    def test_frame_size_change_stops_tracking(self):
        tracker = TemporalFaceTracker()
        self.assertTrue(tracker.reset(_scene(640, 480), _face(640, 480, 160)))
        self.assertIsNone(tracker.update(_scene(1280, 720)))

    def test_rejects_missing_tiny_and_off_frame_faces(self):
        frame = _scene(640, 480)
        tiny = _face(640, 480, 10)
        off_frame = _face(640, 480, 160)
        off_frame.bbox = off_frame.bbox + [700, 0, 700, 0]
        tracker = TemporalFaceTracker()
        for face in (None, tiny, off_frame):
            with self.subTest(face=None if face is None else face.bbox):
                self.assertFalse(tracker.reset(frame, face))
                self.assertIsNone(tracker.update(frame))


if __name__ == "__main__":
    unittest.main()
