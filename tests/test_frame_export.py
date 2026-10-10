import io
import subprocess
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import modules.globals as state
import modules.face_analyser as analyser
from modules.processors.frame import core as frame_core


class _Progress:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def set_postfix(self, _values):
        pass

    def update(self, _count):
        pass


class _WriterInput:
    def __init__(self, *, broken=False):
        self.frames = []
        self.closed = False
        self.broken = broken

    def write(self, data):
        if self.broken:
            raise BrokenPipeError("test encoder closed")
        self.frames.append(data)

    def close(self):
        self.closed = True


class _Process:
    def __init__(self, *, stdin=None, stdout=None, stderr=None):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = None
        self.exit_code = 0
        self.kill_count = 0
        self.wait_timeouts = []
        self.finalization_timeout = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.kill_count += 1

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.finalization_timeout and self.kill_count == 0:
            raise subprocess.TimeoutExpired("test ffmpeg", timeout)
        self.returncode = -9 if self.kill_count else self.exit_code
        return self.returncode


def _pipe_boundaries(monkeypatch, frame_ids, *, broken=False):
    raw = b"".join(np.full((2, 2, 3), value, dtype=np.uint8).tobytes()
                   for value in frame_ids)
    reader = _Process(stdout=io.BytesIO(raw), stderr=io.BytesIO())
    sink = _WriterInput(broken=broken)
    writer = _Process(stdin=sink, stderr=io.BytesIO())
    sink.processes = [reader, writer]
    processes = iter([reader, writer])
    monkeypatch.setattr(frame_core.subprocess, "Popen", lambda *_a, **_k: next(processes))
    monkeypatch.setattr(frame_core, "tqdm", lambda **_kwargs: _Progress())
    monkeypatch.setattr(frame_core.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(state, "many_faces", False)
    monkeypatch.setattr(state, "mouth_mask", False)
    return sink


def _run_pipe(processors):
    return frame_core._run_pipe_pipeline(
        "input.mp4", "temp.mp4", 30.0, object(), processors,
        2, 2, 12, 3, "libx264", ["-crf", "18"],
    )


def test_export_uses_current_frame_geometry_across_motion_and_scene_cut(monkeypatch):
    sink = _pipe_boundaries(monkeypatch, [10, 20, 240])
    detected = []
    swapped = []

    def detect(frame):
        frame_id = int(frame[0, 0, 0])
        detected.append(frame_id)
        return SimpleNamespace(frame_id=frame_id)

    class Processor:
        NAME = "DLC.FACE-SWAPPER"

        def process_frame(self, _source, frame, *, target_face):
            frame_id = int(frame[0, 0, 0])
            swapped.append((frame_id, target_face.frame_id))
            # Real paste-back mutates the processed buffer. Detection for the
            # next frame must have its own buffer, not this one.
            frame[:] = 0
            return frame

    monkeypatch.setattr(frame_core, "detect_one_face_fast", detect)
    assert _run_pipe([Processor()])
    assert detected == [10, 20, 240]
    assert swapped == [(10, 10), (20, 20), (240, 240)]
    assert len(sink.frames) == 3
    assert sink.closed


@pytest.mark.parametrize("failure", ["processor", "encoder"])
def test_export_joins_lookahead_detector_on_failure(monkeypatch, failure):
    sink = _pipe_boundaries(monkeypatch, [10, 20], broken=failure == "encoder")
    next_detection_started = threading.Event()
    allow_finish = threading.Event()
    finished = threading.Event()
    release_timer = None

    def detect(frame):
        if int(frame[0, 0, 0]) == 20:
            next_detection_started.set()
            assert allow_finish.wait(timeout=2)
            finished.set()
        return object()

    class Processor:
        NAME = "DLC.FACE-SWAPPER"

        def process_frame(self, _source, frame, *, target_face):
            nonlocal release_timer
            assert next_detection_started.wait(timeout=2)
            release_timer = threading.Timer(0.02, allow_finish.set)
            release_timer.start()
            if failure == "processor":
                raise RuntimeError("test processor failed")
            return frame

    monkeypatch.setattr(frame_core, "detect_one_face_fast", detect)
    try:
        assert not _run_pipe([Processor()])
        assert finished.is_set()
        for process in sink.processes:
            assert process.kill_count == 1
            assert process.returncode == -9
            assert process.wait_timeouts == [5]
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    assert pipe.closed
    finally:
        allow_finish.set()
        if release_timer is not None:
            release_timer.join()


def test_export_preserves_explicit_no_face_detection(monkeypatch):
    sink = _pipe_boundaries(monkeypatch, [10, 20, 240])
    detected = []
    received = []

    def detect(frame):
        detected.append(int(frame[0, 0, 0]))
        return None

    class Processor:
        NAME = "DLC.FACE-SWAPPER"

        def process_frame(self, _source, frame, *, target_face):
            raise AssertionError("no-face frames must not trigger swap/redetection")

        def apply_post_processing(self, frame, bboxes):
            received.append(bboxes)
            return frame

    monkeypatch.setattr(frame_core, "detect_one_face_fast", detect)
    assert _run_pipe([Processor()])
    assert detected == [10, 20, 240]
    assert received == [[], [], []]
    assert len(sink.frames) == 3


def test_export_shares_current_detection_with_enhancer_including_empty_list(monkeypatch):
    _pipe_boundaries(monkeypatch, [10, 20, 240])
    received = []

    def detect(frame):
        frame_id = int(frame[0, 0, 0])
        return None if frame_id == 20 else SimpleNamespace(frame_id=frame_id)

    class Enhancer:
        NAME = "DLC.FACE-ENHANCER"

        def process_frame(self, _source, frame, *, detected_faces):
            received.append([face.frame_id for face in detected_faces])
            return frame

    monkeypatch.setattr(frame_core, "detect_one_face_fast", detect)
    assert _run_pipe([Enhancer()])
    assert received == [[10], [], [240]]


def test_export_internal_processor_type_error_is_not_retried(monkeypatch):
    _pipe_boundaries(monkeypatch, [10, 20])
    attempts = []

    class Processor:
        NAME = "DLC.FACE-SWAPPER"

        def process_frame(self, _source, frame, *, target_face=None):
            attempts.append(target_face)
            raise TypeError("test internal processor error")

    monkeypatch.setattr(frame_core, "detect_one_face_fast", lambda _frame: object())
    assert not _run_pipe([Processor()])
    assert len(attempts) == 1


def test_export_success_closes_and_reaps_all_ffmpeg_resources(monkeypatch):
    sink = _pipe_boundaries(monkeypatch, [10, 20, 240])
    monkeypatch.setattr(frame_core, "detect_one_face_fast", lambda _frame: None)
    assert _run_pipe([])
    for process in sink.processes:
        assert process.returncode == 0
        assert process.kill_count == 0
        assert process.wait_timeouts[-1] == 5
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                assert pipe.closed


def test_export_decoder_failure_cannot_report_success_for_partial_frames(monkeypatch):
    sink = _pipe_boundaries(monkeypatch, [10, 20, 240])
    sink.processes[0].exit_code = 1
    monkeypatch.setattr(frame_core, "detect_one_face_fast", lambda _frame: None)
    assert not _run_pipe([])
    assert len(sink.frames) == 3
    assert sink.processes[0].returncode == 1
    assert sink.processes[0].stdout.closed


def test_export_encoder_start_failure_cleans_already_started_reader(monkeypatch):
    sink = _pipe_boundaries(monkeypatch, [10, 20, 240])
    reader = sink.processes[0]

    def start(_command, **kwargs):
        if "stdout" in kwargs:
            return reader
        raise OSError("test encoder could not start")

    monkeypatch.setattr(frame_core.subprocess, "Popen", start)
    assert not _run_pipe([])
    assert reader.kill_count == 1
    assert reader.returncode == -9
    assert reader.stdout.closed
    assert reader.stderr.closed


def test_export_finalization_timeout_forces_bounded_cleanup(monkeypatch):
    sink = _pipe_boundaries(monkeypatch, [10, 20, 240])
    sink.processes[1].finalization_timeout = True
    monkeypatch.setattr(frame_core, "detect_one_face_fast", lambda _frame: None)
    assert not _run_pipe([])
    assert sink.processes[1].wait_timeouts == [30, 5]
    for process in sink.processes:
        assert process.kill_count == 1
        assert process.returncode == -9
        for pipe in (process.stdin, process.stdout, process.stderr):
            if pipe is not None:
                assert pipe.closed


def test_export_fast_detector_preserves_face_selection_without_recognition(monkeypatch):
    _pipe_boundaries(monkeypatch, [10, 20, 240])
    enhanced_faces = []

    def unexpected(*_args, **_kwargs):
        raise AssertionError("target recognition/106-point model is unnecessary")

    def detect(_frame, **_kwargs):
        return (
            np.array([[40, 0, 50, 10, 0.99], [4, 0, 14, 10, 0.98]]),
            np.array([np.full((5, 2), 34), np.full((5, 2), 17)]),
        )

    face_analysis = SimpleNamespace(
        det_model=SimpleNamespace(detect=detect),
        models={"recognition": SimpleNamespace(get=unexpected),
                "landmark_2d_106": SimpleNamespace(get=unexpected)},
    )
    monkeypatch.setattr(analyser, "get_face_analyser", lambda: face_analysis)
    monkeypatch.setattr(frame_core, "detect_one_face_fast", analyser.detect_one_face_fast)
    monkeypatch.setattr(frame_core, "ensure_landmarks", unexpected)

    class Enhancer:
        NAME = "DLC.FACE-ENHANCER-GPEN512"

        def process_frame(self, _source, frame, *, detected_faces):
            face = detected_faces[0]
            assert face.bbox[0] == 4
            assert np.array_equal(face.kps, np.full((5, 2), 17))
            assert face.normed_embedding is None
            enhanced_faces.append(face)
            return frame

    assert _run_pipe([Enhancer()])
    assert len(enhanced_faces) == 3


def test_export_adds_mouth_landmarks_to_the_same_current_frame(monkeypatch):
    _pipe_boundaries(monkeypatch, [10, 20, 240])
    monkeypatch.setattr(state, "mouth_mask", True)
    landmark_frames = []

    def detect(frame):
        frame_id = int(frame[0, 0, 0])
        return None if frame_id == 20 else SimpleNamespace(frame_id=frame_id)

    def landmarks(frame, faces):
        landmark_frames.append((int(frame[0, 0, 0]), faces[0].frame_id))
        faces[0].landmark_2d_106 = np.zeros((106, 2))

    class Swapper:
        NAME = "DLC.FACE-SWAPPER"

        def process_frame(self, _source, frame, *, target_face):
            assert target_face.landmark_2d_106.shape == (106, 2)
            return frame

        def apply_post_processing(self, frame, _bboxes):
            return frame

    monkeypatch.setattr(frame_core, "detect_one_face_fast", detect)
    monkeypatch.setattr(frame_core, "ensure_landmarks", landmarks)
    assert _run_pipe([Swapper()])
    assert landmark_frames == [(10, 10), (240, 240)]
