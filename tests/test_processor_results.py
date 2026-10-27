from types import SimpleNamespace

import numpy as np
import pytest

import modules.globals as state
from modules.processors.frame import (
    _onnx_enhancer,
    face_enhancer as gfpgan,
    face_enhancer_gpen256 as gpen256,
    face_enhancer_gpen512 as gpen512,
    face_swapper as swapper,
)


PROCESSORS = [swapper, gfpgan, gpen256, gpen512]
ENHANCERS = [gfpgan, gpen256, gpen512]


def _face(dx=0):
    return SimpleNamespace(
        normed_embedding=np.ones(512, dtype=np.float32),
        bbox=np.array([5 + dx, 5, 55 + dx, 55]),
        kps=np.array([[20, 23], [43, 23], [32, 35], [23, 47], [41, 47]],
                     dtype=np.float32) + np.array([dx, 0], dtype=np.float32),
    )


@pytest.fixture
def images(monkeypatch):
    source = np.full((64, 64, 3), 12, dtype=np.uint8)
    target = np.full((64, 64, 3), 42, dtype=np.uint8)
    for name, value in {
        "headless": True, "map_faces": False, "many_faces": False,
        "opacity": 1.0, "sharpness": 0.0, "enable_interpolation": False,
        "mouth_mask": False, "poisson_blend": False,
    }.items():
        monkeypatch.setattr(state, name, value)
    return source, target


def _image_boundaries(monkeypatch, processor, images, *, target_has_face=False):
    source, target = images
    monkeypatch.setattr(
        processor, "imread_unicode",
        lambda path: source if path == "source.jpg" else target,
    )
    face = _face()
    if processor is swapper:
        monkeypatch.setattr(processor, "get_one_face",
                            lambda image: face if image is source or target_has_face else None)
    elif processor is gfpgan:
        monkeypatch.setattr(processor, "get_many_faces",
                            lambda _image: [face] if target_has_face else [])
    else:
        monkeypatch.setattr(processor, "get_one_face",
                            lambda _image: face if target_has_face else None)


def _model_boundaries(monkeypatch, processor):
    if processor is swapper:
        model = SimpleNamespace(
            input_size=(128, 128),
            get=lambda *_args, **_kwargs: (
                np.full((128, 128, 3), 128, dtype=np.uint8),
                np.array([[2, 0, 0], [0, 2, 0]], dtype=np.float32),
            ),
        )
        monkeypatch.setattr(swapper, "get_face_swapper", lambda: model)
    else:
        session = SimpleNamespace(get_inputs=lambda: [
            SimpleNamespace(name="input", shape=[1, 3, 8, 8]),
        ])
        getter = "get_face_enhancer" if processor is gfpgan else "get_enhancer"
        monkeypatch.setattr(processor, getter, lambda: session)
        monkeypatch.setattr(_onnx_enhancer, "run_inference",
                            lambda _session, _name, blob: blob.copy())


@pytest.mark.parametrize("processor", PROCESSORS, ids=lambda p: p.NAME)
@pytest.mark.parametrize("write_succeeds", [False, True])
def test_image_result_requires_successful_write(monkeypatch, processor, write_succeeds, images):
    _image_boundaries(monkeypatch, processor, images, target_has_face=True)
    _model_boundaries(monkeypatch, processor)
    writes = []

    def write(path, frame):
        writes.append((path, frame))
        return write_succeeds

    monkeypatch.setattr(processor, "imwrite_unicode", write)
    assert processor.process_image("source.jpg", "target.jpg", "result.png") is write_succeeds
    assert len(writes) == 1
    assert writes[0][0] == "result.png"


@pytest.mark.parametrize("processor", PROCESSORS, ids=lambda p: p.NAME)
def test_image_without_target_face_cannot_write_original_as_success(monkeypatch, processor, images):
    _image_boundaries(monkeypatch, processor, images)
    writes = []
    monkeypatch.setattr(processor, "imwrite_unicode", lambda *_args: writes.append(1))
    assert processor.process_image("source.jpg", "target.jpg", "result.png") is False
    assert writes == []


@pytest.mark.parametrize("many_faces", [False, True])
def test_mapped_image_without_valid_pairs_is_failure(monkeypatch, many_faces, images):
    _image_boundaries(monkeypatch, swapper, images)
    writes = []
    monkeypatch.setattr(state, "map_faces", True)
    monkeypatch.setattr(state, "many_faces", many_faces)
    monkeypatch.setattr(state, "target_path", "target.jpg")
    monkeypatch.setattr(state, "source_target_map", [{"target": {"face": _face()}}])
    monkeypatch.setattr(swapper, "is_image", lambda _path: True)
    monkeypatch.setattr(swapper, "default_source_face", lambda: None)
    monkeypatch.setattr(swapper, "imwrite_unicode", lambda *_args: writes.append(1))
    assert swapper.process_image("source.jpg", "target.jpg", "result.png") is False
    assert writes == []


def test_valid_mapped_image_preserves_preselected_faces_with_staged_path(monkeypatch, images):
    _image_boundaries(monkeypatch, swapper, images)
    _model_boundaries(monkeypatch, swapper)
    source_face, target_face = _face(), _face(3)
    monkeypatch.setattr(state, "map_faces", True)
    monkeypatch.setattr(state, "target_path", "target.jpg")
    monkeypatch.setattr(state, "source_target_map", [{
        "source": {"face": source_face}, "target": {"face": target_face},
    }])
    monkeypatch.setattr(swapper, "is_image", lambda _path: True)
    model = swapper.get_face_swapper()
    model_get = model.get
    swapped = []

    def swap(frame, target, source, **kwargs):
        swapped.append((source, target))
        return model_get(frame, target, source, **kwargs)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("mapped images must use preselected identities and geometry")

    model.get = swap
    monkeypatch.setattr(swapper, "get_one_face", unexpected)
    monkeypatch.setattr(swapper, "get_many_faces", unexpected)
    monkeypatch.setattr(swapper, "imwrite_unicode", lambda *_args: True)
    assert swapper.process_image("source.jpg", "staging.png", "staging.png") is True
    assert swapped == [(source_face, target_face)]


@pytest.mark.parametrize("mapped", [False, True])
def test_explicit_zero_opacity_is_valid_noop_even_without_target_face(monkeypatch, mapped, images):
    _image_boundaries(monkeypatch, swapper, images)
    monkeypatch.setattr(state, "opacity", 0.0)
    monkeypatch.setattr(state, "map_faces", mapped)
    monkeypatch.setattr(state, "source_target_map", [])
    monkeypatch.setattr(swapper, "imwrite_unicode", lambda *_args: True)
    assert swapper.process_image("source.jpg", "target.jpg", "result.png") is True


def test_normal_video_frame_without_face_remains_accepted(monkeypatch, images):
    _image_boundaries(monkeypatch, swapper, images)
    frame = images[1]
    assert swapper.process_frame(_face(), frame) is frame


@pytest.mark.parametrize("analysis", ["no_face", "failure"])
def test_source_analysis_failure_is_not_reported_as_success(monkeypatch, analysis, images):
    _image_boundaries(monkeypatch, swapper, images)
    writes = []

    def detect(_image):
        if analysis == "failure":
            raise RuntimeError("test source detector failure")
        return None

    monkeypatch.setattr(swapper, "get_one_face", detect)
    monkeypatch.setattr(swapper, "imwrite_unicode", lambda *_args: writes.append(1))
    assert swapper.process_image("source.jpg", "target.jpg", "result.png") is False
    assert writes == []


@pytest.mark.parametrize("processor", PROCESSORS, ids=lambda p: p.NAME)
def test_image_inference_failure_does_not_write_original_as_success(monkeypatch, processor, images):
    _image_boundaries(monkeypatch, processor, images, target_has_face=True)
    writes = []

    def fail(*_args, **_kwargs):
        raise RuntimeError("test model inference failed")

    if processor is swapper:
        model = SimpleNamespace(get=fail)
        monkeypatch.setattr(swapper, "get_face_swapper", lambda: model)
    else:
        session = SimpleNamespace(get_inputs=lambda: [
            SimpleNamespace(name="input", shape=[1, 3, 8, 8]),
        ])
        getter = "get_face_enhancer" if processor is gfpgan else "get_enhancer"
        monkeypatch.setattr(processor, getter, lambda: session)
        monkeypatch.setattr(_onnx_enhancer, "run_inference", fail)
    monkeypatch.setattr(processor, "imwrite_unicode", lambda *_args: writes.append(1))
    assert processor.process_image("source.jpg", "target.jpg", "result.png") is False
    assert writes == []


@pytest.mark.parametrize("geometry", ["missing", "degenerate", "nonfinite"])
def test_image_swapper_invalid_alignment_never_writes_original_as_success(monkeypatch, geometry, images):
    _image_boundaries(monkeypatch, swapper, images, target_has_face=True)
    _model_boundaries(monkeypatch, swapper)
    if geometry == "missing":
        affine = None
    elif geometry == "degenerate":
        affine = np.zeros((2, 3), dtype=np.float32)
    else:
        affine = np.array([[1, 0, np.nan], [0, 1, 0]], dtype=np.float32)
    swapper.get_face_swapper().get = lambda *_args, **_kwargs: (
        np.full((128, 128, 3), 128, dtype=np.uint8), affine,
    )
    writes = []
    monkeypatch.setattr(swapper, "imwrite_unicode", lambda *_args: writes.append(1))
    assert swapper.process_image("source.jpg", "target.jpg", "result.png") is False
    assert writes == []


@pytest.mark.parametrize("processor", PROCESSORS, ids=lambda p: p.NAME)
def test_image_missing_result_is_failure(monkeypatch, processor, images):
    _image_boundaries(monkeypatch, processor, images)
    writes = []
    monkeypatch.setattr(processor, "process_frame", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(processor, "imwrite_unicode", lambda *_args: writes.append(1))
    assert processor.process_image("source.jpg", "target.jpg", "result.png") is False
    assert writes == []


@pytest.mark.parametrize("processor", ENHANCERS, ids=lambda p: p.NAME)
def test_explicit_empty_faces_skips_detector_and_model(monkeypatch, processor, images):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("empty face list must not run detection or load a model")

    detector = "get_many_faces" if processor is gfpgan else "get_one_face"
    getter = "get_face_enhancer" if processor is gfpgan else "get_enhancer"
    monkeypatch.setattr(processor, detector, unexpected)
    monkeypatch.setattr(processor, getter, unexpected)
    frame = images[1]
    assert processor.process_frame(None, frame, detected_faces=[]) is frame


@pytest.mark.parametrize("processor", ENHANCERS, ids=lambda p: p.NAME)
@pytest.mark.parametrize("geometry", ["missing", "invalid", "failed_affine", "degenerate"])
def test_image_failed_alignment_never_writes_original_as_success(
    monkeypatch, processor, geometry, images
):
    _image_boundaries(monkeypatch, processor, images, target_has_face=True)
    _model_boundaries(monkeypatch, processor)
    face = _face()
    if geometry == "missing":
        face.kps = None
    elif geometry == "invalid":
        face.kps[0, 0] = np.nan
    elif geometry == "failed_affine":
        monkeypatch.setattr(_onnx_enhancer.cv2, "estimateAffinePartial2D", lambda *_a, **_k: (None, None))
    else:
        monkeypatch.setattr(_onnx_enhancer.cv2, "estimateAffinePartial2D",
                            lambda *_a, **_k: (np.zeros((2, 3), dtype=np.float32), None))
    detector = "get_many_faces" if processor is gfpgan else "get_one_face"
    monkeypatch.setattr(processor, detector, lambda _frame: [face] if processor is gfpgan else face)
    writes = []
    inference_calls = []
    monkeypatch.setattr(processor, "imwrite_unicode", lambda *_args: writes.append(1))

    def unexpected(*_args, **_kwargs):
        inference_calls.append(1)
        raise AssertionError("failed alignment must not invoke the model")

    monkeypatch.setattr(_onnx_enhancer, "run_inference", unexpected)
    assert processor.process_image("source.jpg", "target.jpg", "result.png") is False
    assert writes == []
    assert inference_calls == []


def test_gfpgan_uses_fresh_pixels_and_geometry_on_consecutive_frames(monkeypatch, images):
    session = SimpleNamespace(get_inputs=lambda: [
        SimpleNamespace(name="input", shape=[1, 3, 8, 8]),
    ])
    monkeypatch.setattr(gfpgan, "get_face_enhancer", lambda: session)
    inference_inputs = []
    pastes = []
    paste_back = gfpgan._paste_back

    def inference(_session, _name, blob):
        inference_inputs.append(blob.copy())
        return blob.copy()

    def paste(frame, restored, affine, *, output_size):
        pastes.append((restored.copy(), affine.copy()))
        return paste_back(frame, restored, affine, output_size=output_size)

    monkeypatch.setattr(_onnx_enhancer, "run_inference", inference)
    monkeypatch.setattr(gfpgan, "_paste_back", paste)
    # Warm up, then check two adjacent frames with a moving face and changed
    # expression/pixels. The old every-other-frame cache froze the third frame.
    for value, dx in [(20, 0), (80, 3), (140, 6)]:
        frame = np.full((64, 64, 3), value, dtype=np.uint8)
        result = gfpgan.process_frame(None, frame, detected_faces=[_face(dx)], strict=True)
        assert result is frame

    assert len(inference_inputs) == 3
    assert len(pastes) == 3
    assert not np.array_equal(inference_inputs[1], inference_inputs[2])
    assert not np.array_equal(pastes[1][0], pastes[2][0])
    assert not np.allclose(pastes[1][1], pastes[2][1])
