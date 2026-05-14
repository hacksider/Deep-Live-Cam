import importlib
import io
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


def _load_core():
    sys.modules["modules.face_analyser"] = types.SimpleNamespace(
        get_one_face=lambda *_args, **_kwargs: None,
    )
    sys.modules.pop("modules.processors.frame.core", None)
    return importlib.import_module("modules.processors.frame.core")


class _DummyTqdm:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_postfix(self, *_args, **_kwargs):
        pass

    def update(self, *_args, **_kwargs):
        pass


class _FakeStdout:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read(self, _size):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _FakeStdin:
    def __init__(self):
        self.writes = []
        self.closed = False

    def write(self, data):
        self.writes.append(data)

    def close(self):
        self.closed = True


class _FakeProc:
    def __init__(self, *, stdout_chunks=(), returncode=0, stderr=b""):
        self.stdout = _FakeStdout(stdout_chunks)
        self.stdin = _FakeStdin()
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.killed = False

    def wait(self):
        return self.returncode

    def kill(self):
        self.killed = True


class PipePipelineIntegrityTests(unittest.TestCase):
    def _run_pipeline(self, *, reader_returncode=0, writer_returncode=0, frame_chunks=1, total_frames=1):
        core = _load_core()
        frame_size = 3
        frame = np.array([1, 2, 3], dtype=np.uint8).tobytes()
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        output_path = os.path.join(tmp_dir.name, "out.mp4")
        with open(output_path, "wb") as f:
            f.write(b"video")

        calls = []

        def fake_popen(_cmd, stdout=None, stdin=None, stderr=None):
            if stdout is not None:
                proc = _FakeProc(
                    stdout_chunks=[frame] * frame_chunks + [b""],
                    returncode=reader_returncode,
                    stderr=b"decoder failed",
                )
            else:
                proc = _FakeProc(returncode=writer_returncode, stderr=b"encoder failed")
            calls.append(proc)
            return proc

        processor = types.SimpleNamespace(process_frame=lambda _source, frame, **_kwargs: frame)
        core.modules.globals.execution_providers = []
        core.modules.globals.execution_threads = 1
        core.modules.globals.many_faces = True

        with patch.object(core, "subprocess", types.SimpleNamespace(Popen=fake_popen, PIPE=object())), patch.object(core, "tqdm", _DummyTqdm):
            result = core._run_pipe_pipeline(
                "input.mp4",
                output_path,
                30.0,
                None,
                [processor],
                1,
                1,
                frame_size,
                total_frames,
                "libx264",
                [],
            )
        return result, calls

    def test_decoder_failure_is_not_reported_as_success(self):
        result, _calls = self._run_pipeline(
            reader_returncode=1,
            writer_returncode=0,
            frame_chunks=1,
            total_frames=1,
        )

        self.assertFalse(result)

    def test_truncated_frame_count_falls_back(self):
        result, _calls = self._run_pipeline(
            reader_returncode=0,
            writer_returncode=0,
            frame_chunks=1,
            total_frames=2,
        )

        self.assertFalse(result)

    def test_complete_pipeline_still_succeeds(self):
        result, _calls = self._run_pipeline(
            reader_returncode=0,
            writer_returncode=0,
            frame_chunks=2,
            total_frames=2,
        )

        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
