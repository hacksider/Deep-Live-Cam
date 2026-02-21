"""Tests for modules/processors/frame/core.process_frames_io."""

import os
import numpy as np
import cv2
import pytest


def test_process_frames_io_reads_processes_writes(tmp_path):
    from modules.processors.frame.core import process_frames_io

    # Create a test frame
    frame = np.full((64, 64, 3), 100, dtype=np.uint8)
    path = str(tmp_path / "frame.jpg")
    cv2.imwrite(path, frame)

    # Process: invert colours
    def invert(f):
        return 255 - f

    process_frames_io([path], process_fn=invert)

    result = cv2.imread(path)
    assert result is not None
    # Inverted 100 → 155 (JPEG compression may shift a few values)
    assert abs(int(result[0, 0, 0]) - 155) < 10


def test_process_frames_io_skips_missing(tmp_path):
    from modules.processors.frame.core import process_frames_io

    missing = str(tmp_path / "nope.jpg")

    class FakeProgress:
        def __init__(self):
            self.count = 0
        def update(self, n):
            self.count += n

    prog = FakeProgress()
    process_frames_io([missing], process_fn=lambda f: f, progress=prog)
    assert prog.count == 1


def test_process_frames_io_handles_none_return(tmp_path):
    from modules.processors.frame.core import process_frames_io

    frame = np.full((32, 32, 3), 50, dtype=np.uint8)
    path = str(tmp_path / "frame.jpg")
    cv2.imwrite(path, frame)

    # process_fn returns None → should write original frame
    process_frames_io([path], process_fn=lambda f: None)

    result = cv2.imread(path)
    assert result is not None
    assert abs(int(result[0, 0, 0]) - 50) < 5
