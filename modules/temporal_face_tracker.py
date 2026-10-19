"""Optical-flow face tracking between detector runs in the live preview.

The live loop only runs the face detector every few frames. On the frames in
between it reused the last detection unchanged, so the swapped face was pasted
where the head *was* and visibly lagged behind it. ``TemporalFaceTracker``
carries that detection along with the head instead: it follows feature points
with pyramidal Lucas-Kanade flow, fits a similarity transform with RANSAC and
applies it to the detection's bbox, keypoints and 106-point landmarks.

All work happens on a crop around the face that is downscaled to a fixed size,
so it costs about a millisecond per frame whatever the camera resolution.
Whenever an estimate looks unreliable, ``update`` returns ``None`` and the
caller falls back to the last detection.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Optional, Tuple

import cv2
import numpy as np

# Rough size of the face after downscaling. Larger costs more and tracks no
# better.
WORK_SIZE = 160
# Context kept around the bbox, as a fraction of its size. It bounds how far
# the head can move in one frame and still be tracked.
CROP_MARGIN = 0.5
MAX_POINTS = 60
MIN_POINTS = 8
MIN_FACE_SIZE = 16.0
MIN_TRACKED_RATIO = 0.5
MIN_INLIER_RATIO = 0.6
MAX_LK_ERROR = 20.0
# Bounds on the motion between two consecutive frames.
MAX_STEP_SCALE = 1.25
MAX_STEP_ROTATION = 30.0
MAX_STEP_TRANSLATION = 0.35
# Bound on the total scale change since the last detection.
MAX_TOTAL_SCALE = 2.0

_LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)

# (x0, y0, x1, y1, factor): a crop in frame pixels and the integer factor it
# is downscaled by. Its size is a multiple of the factor, which keeps
# INTER_AREA on its fast path.
Window = Tuple[int, int, int, int, int]


class TemporalFaceTracker:
    """Track one face from a detection until the next detection.

    ``reset(frame, face)`` registers a fresh detection; ``update(frame)``
    returns a copy of that face moved into ``frame``, or ``None`` when the
    motion cannot be estimated reliably. After a failure the tracker stays
    inactive until the next ``reset``. The face passed to ``reset`` is never
    mutated. Not thread-safe: the processing worker owns its tracker.
    """

    def __init__(self) -> None:
        self.clear()

    @property
    def active(self) -> bool:
        return self._reference is not None

    def clear(self) -> None:
        self._reference: Any = None
        self._reference_bbox: Optional[np.ndarray] = None
        # Similarity transform from the reference detection to the last frame.
        self._matrix: Optional[np.ndarray] = None
        self._frame_shape: Optional[Tuple[int, ...]] = None
        self._window: Optional[Window] = None
        self._previous: Optional[np.ndarray] = None
        # Tracked points in frame coordinates, shape (N, 2).
        self._points: Optional[np.ndarray] = None

    def reset(self, frame: np.ndarray, face: Any) -> bool:
        """Start tracking ``face`` as detected in ``frame``.

        Returns False, leaving the tracker inactive, when there is no face or
        it is too small, off-frame or too featureless to track.
        """
        self.clear()
        bbox = _as_points(_get(face, "bbox"), 4)
        if bbox is None or len(bbox) != 1 or not _usable_bbox(bbox[0], frame.shape):
            return False
        bbox = bbox[0]

        window = _window(bbox, frame.shape)
        if window is None:
            return False
        gray = _crop(frame, window)
        points = _add_features(gray, window, bbox, _as_points(_get(face, "kps"), 2))
        if len(points) < MIN_POINTS:
            return False

        self._reference = face
        self._reference_bbox = bbox
        self._matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        self._frame_shape = frame.shape
        self._window = window
        self._previous = gray
        self._points = points
        return True

    def update(self, frame: np.ndarray) -> Any:
        """Return the tracked face for ``frame``, or None if unreliable."""
        if self._reference is None:
            return None
        if frame.shape != self._frame_shape:
            return self._fail()

        window = self._window
        current = _crop(frame, window)
        old_small = _to_window(self._points, window)
        try:
            new_small, status, error = cv2.calcOpticalFlowPyrLK(
                self._previous, current, old_small.reshape(-1, 1, 2), None,
                winSize=(15, 15), maxLevel=3, criteria=_LK_CRITERIA,
            )
        except cv2.error:
            return self._fail()
        if new_small is None:
            return self._fail()

        new_small = new_small.reshape(-1, 2)
        ok = (status.reshape(-1) == 1) & (error.reshape(-1) <= MAX_LK_ERROR)
        ok &= np.isfinite(new_small).all(axis=1)
        tracked = int(ok.sum())
        if tracked < MIN_POINTS or tracked < MIN_TRACKED_RATIO * len(ok):
            return self._fail()

        old = self._points[ok]
        new = _from_window(new_small[ok], window)
        # One pixel of the downscaled crop is ``factor`` frame pixels.
        step, inliers = cv2.estimateAffinePartial2D(
            old, new, method=cv2.RANSAC, ransacReprojThreshold=1.5 * window[4],
            maxIters=200, confidence=0.99, refineIters=10,
        )
        if step is None or inliers is None or not np.isfinite(step).all():
            return self._fail()
        inliers = inliers.reshape(-1).astype(bool)
        if inliers.sum() < MIN_POINTS or inliers.mean() < MIN_INLIER_RATIO:
            return self._fail()
        if not self._plausible_step(step):
            return self._fail()

        matrix = _compose(step, self._matrix)
        face = self._moved_face(matrix)
        if face is None:
            return self._fail()
        bbox = _get(face, "bbox")

        # Keep the crop while the face stays well inside it, so this frame's
        # crop can be reused; otherwise re-center it on the face.
        if _covers(window, bbox, frame.shape):
            gray = current
        else:
            window = _window(bbox, frame.shape)
            if window is None:
                return self._fail()
            gray = _crop(frame, window)
        points = _inside(new[inliers], bbox)
        if len(points) < MAX_POINTS // 2:
            points = _add_features(gray, window, bbox, points)
        if len(points) < MIN_POINTS:
            return self._fail()

        self._matrix = matrix
        self._window = window
        self._previous = gray
        self._points = points
        return face

    def _plausible_step(self, step: np.ndarray) -> bool:
        scale = math.hypot(step[0, 0], step[1, 0])
        if not 1 / MAX_STEP_SCALE <= scale <= MAX_STEP_SCALE:
            return False
        if abs(math.degrees(math.atan2(step[1, 0], step[0, 0]))) > MAX_STEP_ROTATION:
            return False
        bbox = _apply(self._matrix, self._reference_bbox.reshape(2, 2))
        center = bbox.mean(axis=0)
        shift = np.linalg.norm(_apply(step, center[None])[0] - center)
        return shift <= MAX_STEP_TRANSLATION * np.linalg.norm(bbox[1] - bbox[0])

    def _moved_face(self, matrix: np.ndarray) -> Any:
        """Copy of the reference face with its geometry mapped by ``matrix``."""
        scale = math.hypot(matrix[0, 0], matrix[1, 0])
        if not 1 / MAX_TOTAL_SCALE <= scale <= MAX_TOTAL_SCALE:
            return None
        # Move the bbox center and scale its size, but keep it axis-aligned and
        # with the detector's aspect: a rotated, re-bounded box would grow with
        # head roll, which a detector box does not.
        reference = self._reference_bbox
        center = _apply(matrix, ((reference[:2] + reference[2:]) / 2)[None])[0]
        half = (reference[2:] - reference[:2]) * scale / 2
        bbox = np.concatenate([center - half, center + half]).astype(np.float32)
        if not _usable_bbox(bbox, self._frame_shape):
            return None

        face = _copy_face(self._reference)
        _set(face, "bbox", bbox)
        for name in ("kps", "landmark_2d_106"):
            points = _as_points(_get(self._reference, name), 2)
            if points is not None:
                _set(face, name, _apply(matrix, points).astype(np.float32))
        return face

    def _fail(self) -> None:
        self.clear()
        return None


def _get(face: Any, name: str) -> Any:
    if isinstance(face, dict):
        return face.get(name)
    return getattr(face, name, None)


def _set(face: Any, name: str, value: np.ndarray) -> None:
    if isinstance(face, dict):
        face[name] = value
    else:
        setattr(face, name, value)


def _copy_face(face: Any) -> Any:
    # insightface's Face is a dict whose __getattr__ returns None for unknown
    # names, which breaks copy.copy(). Rebuild it from its items instead.
    if isinstance(face, dict):
        return type(face)(dict(face))
    return copy.copy(face)


def _as_points(value: Any, width: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        points = np.asarray(value, dtype=np.float64).reshape(-1, width)
    except (TypeError, ValueError):
        return None
    if len(points) == 0 or not np.isfinite(points).all():
        return None
    return points


def _usable_bbox(bbox: np.ndarray, frame_shape: Tuple[int, ...]) -> bool:
    """Large enough, and centered inside the frame."""
    if len(bbox) != 4 or not np.isfinite(bbox).all():
        return False
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if min(width, height) < MIN_FACE_SIZE:
        return False
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return 0 <= cx < frame_shape[1] and 0 <= cy < frame_shape[0]


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def _compose(step: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """``step`` applied after ``matrix``."""
    return np.hstack([step[:, :2] @ matrix[:, :2], _apply(step, matrix[:, 2][None]).T])


def _window(bbox: np.ndarray, frame_shape: Tuple[int, ...]) -> Optional[Window]:
    """Crop around ``bbox`` that brings the face down to about WORK_SIZE."""
    height, width = frame_shape[:2]
    size = bbox[2:] - bbox[:2]
    factor = max(1, round(max(size) / WORK_SIZE))
    x0 = max(0, int(bbox[0] - CROP_MARGIN * size[0]))
    y0 = max(0, int(bbox[1] - CROP_MARGIN * size[1]))
    x1 = min(width, int(math.ceil(bbox[2] + CROP_MARGIN * size[0])))
    y1 = min(height, int(math.ceil(bbox[3] + CROP_MARGIN * size[1])))
    x1 -= (x1 - x0) % factor
    y1 -= (y1 - y0) % factor
    if min(x1 - x0, y1 - y0) < MIN_FACE_SIZE * factor:
        return None
    return x0, y0, x1, y1, factor


def _covers(window: Window, bbox: np.ndarray, frame_shape: Tuple[int, ...]) -> bool:
    """Whether ``window`` still holds ``bbox`` with half the usual margin."""
    fresh = _window(bbox, frame_shape)
    if fresh is None or fresh[4] != window[4]:
        return False
    # Halfway between the bbox and a crop freshly centered on it, but never
    # past the frame edge, where the fresh crop is clipped too.
    low = np.maximum((np.array(fresh[:2]) + bbox[:2]) / 2, fresh[:2])
    high = np.minimum((np.array(fresh[2:4]) + bbox[2:]) / 2, fresh[2:4])
    return bool((window[:2] <= low).all() and (window[2:4] >= high).all())


def _crop(frame: np.ndarray, window: Window) -> np.ndarray:
    """Downscaled grayscale crop of ``window``."""
    x0, y0, x1, y1, factor = window
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if factor > 1:
        size = ((x1 - x0) // factor, (y1 - y0) // factor)
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return gray


def _to_window(points: np.ndarray, window: Window) -> np.ndarray:
    return ((points - window[:2]) / window[4]).astype(np.float32)


def _from_window(points: np.ndarray, window: Window) -> np.ndarray:
    return points.astype(np.float64) * window[4] + window[:2]


def _inside(points: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    keep = (points >= bbox[:2]).all(axis=1) & (points <= bbox[2:]).all(axis=1)
    return points[keep]


def _add_features(
    gray: np.ndarray, window: Window, bbox: np.ndarray, points: Optional[np.ndarray]
) -> np.ndarray:
    """Top ``points`` up with corners found inside the face."""
    points = np.empty((0, 2)) if points is None else _inside(points, bbox)
    # Look a little inside the bbox so background corners, which do not move
    # with the head, are not picked up.
    inset = (bbox[2:] - bbox[:2]) * 0.08
    (x0, y0), (x1, y1) = _to_window(
        np.array([bbox[:2] + inset, bbox[2:] - inset]), window
    ).astype(int)
    mask = np.zeros_like(gray)
    mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=MAX_POINTS, qualityLevel=0.01, minDistance=4,
        mask=mask, blockSize=5,
    )
    if corners is None:
        return points
    corners = _from_window(corners.reshape(-1, 2), window)
    return np.concatenate([points, corners[: max(0, MAX_POINTS - len(points))]])
