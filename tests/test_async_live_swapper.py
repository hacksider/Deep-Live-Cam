"""Tests for AsyncLiveSwapper with a stand-in swap model.

The fake model returns its input crop and can be held on a gate, so the tests
control exactly when inferences finish.
"""
import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from insightface.app.common import Face  # noqa: E402
from insightface.utils import face_align  # noqa: E402

import modules.globals  # noqa: E402
import modules.processors.frame.face_swapper as face_swapper  # noqa: E402

KPS = np.float32([[130, 110], [190, 110], [160, 140], [135, 170], [185, 170]])


class _FakeSession:
    """Returns the input crop unchanged; holds each run until ``gate`` is set
    and raises while ``fail`` is set."""

    def __init__(self):
        self.gate = threading.Event()
        self.gate.set()
        self.started = threading.Event()
        self.fail = False
        self.inputs = []

    def run(self, _output_names, inputs):
        blob = inputs["target"]
        self.inputs.append(int(round(blob[0, 0, 64, 64] * 255)))
        self.started.set()
        self.gate.wait(5)
        if self.fail:
            raise RuntimeError("inference failed")
        return [blob]


class _FakeSwapper:
    input_size = (128, 128)
    input_mean = 0.0
    input_std = 255.0
    input_names = ["target", "source"]
    output_names = ["output"]
    emap = np.eye(512, dtype=np.float32)

    def __init__(self):
        self.session = _FakeSession()


def _frame(value):
    """A uniform frame: the crop the model sees tells which frame it came from."""
    return np.full((320, 320, 3), value, dtype=np.uint8)


def _source():
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    return Face(embedding=embedding)


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class AsyncLiveSwapperTest(unittest.TestCase):
    def setUp(self):
        self.model = _FakeSwapper()
        self.pasted = []
        self.loaded_on = []
        modules.globals.execution_providers = ["CPUExecutionProvider"]

        def fake_paste(frame, bgr_fake, M, _target_face):
            self.pasted.append((int(bgr_fake[64, 64, 0]), M))
            return frame

        def fake_get_face_swapper():
            self.loaded_on.append(threading.current_thread().name)
            return self.model

        for name, value in (
            ("FACE_SWAPPER", self.model),
            ("get_face_swapper", fake_get_face_swapper),
            ("paste_swapped_face", fake_paste),
        ):
            patcher = patch.object(face_swapper, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.swapper = face_swapper.AsyncLiveSwapper()
        self.addCleanup(self.swapper.close)
        self.target = Face(kps=KPS.copy())

    def _swap(self, value, source):
        return self.swapper.swap(source, self.target, _frame(value))

    def test_pastes_latest_result_with_current_alignment(self):
        source = _source()
        # Nothing to paste before the first inference finishes.
        self._swap(10, source)
        self.assertEqual(self.pasted, [])
        self.assertTrue(_wait_for(lambda: self.swapper._result is not None))

        self.target.kps = KPS + [7, -3]
        self._swap(20, source)

        value, M = self.pasted[-1]
        self.assertEqual(value, 10, "pasted the wrong inference result")
        _, expected = face_align.norm_crop2(_frame(20), self.target.kps, 128)
        np.testing.assert_allclose(M, expected, atol=1e-5)

    def test_never_waits_for_the_model_and_runs_only_the_newest_crop(self):
        source = _source()
        self.model.session.gate.clear()
        self._swap(10, source)
        self.assertTrue(self.model.session.started.wait(5))

        # The model is busy: these must return at once, and only the last
        # one may be run next.
        started = time.perf_counter()
        for value in (20, 30, 40):
            self._swap(value, source)
        self.assertLess(time.perf_counter() - started, 0.5)

        self.model.session.gate.set()
        self.assertTrue(_wait_for(lambda: len(self.model.session.inputs) == 2))
        time.sleep(0.1)
        self.assertEqual(self.model.session.inputs, [10, 40])

    def test_result_for_another_source_is_not_pasted(self):
        self._swap(10, _source())
        self.assertTrue(_wait_for(lambda: self.swapper._result is not None))

        self._swap(20, _source())

        self.assertEqual(self.pasted, [])

    def test_clear_drops_the_result_and_all_pending_work(self):
        source = _source()
        self.model.session.gate.clear()
        self._swap(10, source)
        self.assertTrue(self.model.session.started.wait(5))
        self._swap(20, source)  # waits in the queue

        self.swapper.clear()
        self.model.session.gate.set()
        time.sleep(0.2)

        self.assertIsNone(self.swapper._result, "the inference in flight was kept")
        self.assertEqual(self.model.session.inputs, [10], "the queued crop still ran")
        self._swap(30, source)
        self.assertEqual(self.pasted, [])

    def test_failed_inference_drops_the_last_result(self):
        source = _source()
        self._swap(10, source)
        self.assertTrue(_wait_for(lambda: self.swapper._result is not None))

        self.model.session.fail = True
        self._swap(20, source)
        self.assertTrue(_wait_for(lambda: self.swapper._result is None))

        self.pasted.clear()
        self._swap(30, source)
        self.assertEqual(self.pasted, [], "kept pasting a result after inference failed")

    def test_result_is_not_pasted_on_a_different_face(self):
        source = _source()
        self._swap(10, source)
        self.assertTrue(_wait_for(lambda: self.swapper._result is not None))

        # The detector switched to someone else, far away in the frame.
        self.target.kps = KPS + [0, 150]
        self._swap(20, source)

        self.assertEqual(self.pasted, [])

    def test_result_is_not_pasted_on_a_face_of_another_size(self):
        source = _source()
        self._swap(10, source)
        self.assertTrue(_wait_for(lambda: self.swapper._result is not None))

        # Someone nearer the camera, in about the same place.
        center = KPS[:2].mean(axis=0)
        self.target.kps = (KPS - center) * 1.8 + center
        self._swap(20, source)

        self.assertEqual(self.pasted, [])

    def test_swap_never_loads_the_model_itself(self):
        with patch.object(face_swapper, "FACE_SWAPPER", None):
            frame = _frame(10)
            self.assertIs(self.swapper.swap(_source(), self.target, frame), frame)
        self.assertTrue(_wait_for(lambda: self.loaded_on))
        self.assertEqual(set(self.loaded_on), {"DLC.ASYNC-SWAP"})

    def test_close_waits_for_the_inference_in_progress(self):
        self.model.session.gate.clear()
        self._swap(10, _source())
        self.assertTrue(self.model.session.started.wait(5))
        threading.Timer(0.3, self.model.session.gate.set).start()

        started = time.perf_counter()
        self.swapper.close()

        self.assertGreaterEqual(time.perf_counter() - started, 0.25)
        self.assertFalse(self.swapper._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
