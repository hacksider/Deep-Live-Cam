"""Offline model-cache regressions using small, real ONNX graphs."""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

# InsightFace imports Albumentations, whose update check would make test
# collection contact the network even though every test body is offline.
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import TensorProto, helper, numpy_helper

from modules import face_analyser, onnx_optimize
from modules.processors.frame import _onnx_enhancer


def test_model_fingerprint_works_without_file_digest_and_preserves_cache_keys(
    tmp_path, monkeypatch
):
    # Python 3.10 lacks file_digest. A multi-chunk file and an empty member
    # must retain the same content/pack digest used by already compiled caches.
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    source = tmp_path / "model.onnx"
    empty = tmp_path / "empty.onnx"
    source_bytes = bytes(range(256)) * 8193
    source.write_bytes(source_bytes)
    empty.write_bytes(b"")
    members = [
        (empty.name, hashlib.sha256(b"").hexdigest()),
        (source.name, hashlib.sha256(source_bytes).hexdigest()),
    ]
    expected = hashlib.sha256(
        json.dumps(members, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert onnx_optimize.model_fingerprint([str(source), str(empty)]) == expected


def _shape_model(path, index=2):
    graph = helper.make_graph(
        [
            helper.make_node("Shape", ["image"], ["shape"]),
            helper.make_node("Gather", ["shape", "index"], ["dimension"]),
        ],
        "shape-folding",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("dimension", TensorProto.INT64, [])],
        [numpy_helper.from_array(np.array(index, dtype=np.int64), name="index")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, path)


def _folded_dimension(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    value = next(item for item in model.graph.initializer if item.name == "dimension")
    return int(numpy_helper.to_array(value))


@pytest.fixture
def model_setup(tmp_path, monkeypatch):
    pack = tmp_path / "installation" / "models" / "buffalo_l"
    pack.mkdir(parents=True)
    source = pack / "det_10g.onnx"
    _shape_model(source)
    cache = tmp_path / "installation" / ".cache"
    monkeypatch.setattr(onnx_optimize, "CACHE_DIR", cache)
    monkeypatch.setattr(onnx_optimize, "IS_APPLE_SILICON", True)
    monkeypatch.setattr(_onnx_enhancer, "IS_APPLE_SILICON", True)
    return source, cache


def test_optimized_model_is_outside_pack_and_reused_without_reloading(
    model_setup, monkeypatch
):
    source, cache = model_setup
    source_bytes = source.read_bytes()
    optimized = onnx_optimize.optimize_for_coreml(str(source), (1, 3, 320, 320))

    assert Path(optimized).parent == cache / "onnx"
    assert list(source.parent.glob("*.onnx")) == [source]
    assert source.read_bytes() == source_bytes
    assert _folded_dimension(optimized) == 320
    monkeypatch.setattr(
        onnx, "load", Mock(side_effect=AssertionError("cache hit must not reload ONNX"))
    )
    assert onnx_optimize.optimize_for_coreml(str(source), (1, 3, 320, 320)) == optimized


def test_detection_shapes_and_rewrite_revision_have_separate_cache_entries(
    model_setup, monkeypatch
):
    source, _cache = model_setup
    small = onnx_optimize.optimize_for_coreml(str(source), (1, 3, 320, 320))
    large = onnx_optimize.optimize_for_coreml(str(source), (1, 3, 640, 640))
    assert small != large
    assert _folded_dimension(small) == 320
    assert _folded_dimension(large) == 640
    monkeypatch.setattr(
        onnx_optimize, "OPTIMIZER_REVISION", onnx_optimize.OPTIMIZER_REVISION + 1
    )
    assert onnx_optimize.optimize_for_coreml(str(source), (1, 3, 320, 320)) != small


def test_changed_model_with_preserved_size_and_timestamp_invalidates_cache(model_setup):
    source, _cache = model_setup
    original_stat = source.stat()
    old = onnx_optimize.optimize_for_coreml(str(source), (1, 3, 320, 640))
    _shape_model(source, index=3)
    assert source.stat().st_size == original_stat.st_size
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    new = onnx_optimize.optimize_for_coreml(str(source), (1, 3, 320, 640))
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert old != new
    assert _folded_dimension(old) == 320
    assert _folded_dimension(new) == 640


def test_interrupted_write_preserves_source_and_previous_complete_cache(
    model_setup, monkeypatch
):
    source, cache = model_setup
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    valid = onnx_optimize.optimize_for_coreml(str(source), (1, 3, 320, 320))
    valid_bytes = Path(valid).read_bytes()

    def interrupted_save(_model, path):
        Path(path).write_bytes(b"partial model")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(onnx, "save", interrupted_save)
    with pytest.raises(OSError, match="interrupted write"):
        onnx_optimize.optimize_for_coreml(str(source), (1, 3, 640, 640))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert Path(valid).read_bytes() == valid_bytes
    assert list((cache / "onnx").iterdir()) == [Path(valid)]
    assert onnx_optimize.optimize_for_coreml(str(source), (1, 3, 320, 320)) == valid


def test_shape_rewrite_matches_real_cpu_inference(model_setup):
    source, _cache = model_setup
    optimized = onnx_optimize.optimize_for_coreml(str(source), (1, 3, 3, 4))
    image = np.arange(36, dtype=np.float32).reshape(1, 3, 3, 4)
    original_session = onnxruntime.InferenceSession(
        str(source), providers=["CPUExecutionProvider"]
    )
    optimized_session = onnxruntime.InferenceSession(
        optimized, providers=["CPUExecutionProvider"]
    )
    np.testing.assert_array_equal(
        original_session.run(None, {"image": image})[0],
        optimized_session.run(None, {"image": image})[0],
    )


def test_reflect_pad_rewrite_matches_real_cpu_inference(model_setup):
    source, _cache = model_setup
    graph = helper.make_graph(
        [helper.make_node("Pad", ["image", "pads"], ["padded"], mode="reflect")],
        "reflect-padding",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 3, 4])],
        [helper.make_tensor_value_info("padded", TensorProto.FLOAT, [1, 1, 5, 8])],
        [
            numpy_helper.from_array(
                np.array([0, 0, 1, 2, 0, 0, 1, 2], dtype=np.int64), name="pads"
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, source)
    optimized = onnx_optimize.optimize_for_coreml(str(source))
    image = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4)
    original_session = onnxruntime.InferenceSession(
        str(source), providers=["CPUExecutionProvider"]
    )
    optimized_session = onnxruntime.InferenceSession(
        optimized, providers=["CPUExecutionProvider"]
    )
    np.testing.assert_array_equal(
        original_session.run(None, {"image": image})[0],
        optimized_session.run(None, {"image": image})[0],
    )


def test_compiled_cache_tracks_model_runtime_macos_and_provider_options(
    model_setup, monkeypatch
):
    source, cache = model_setup
    options = {
        "ModelFormat": "MLProgram",
        "MLComputeUnits": "ALL",
        "AllowLowPrecisionAccumulationOnGPU": 1,
    }
    monkeypatch.setattr(
        _onnx_enhancer.platform, "mac_ver", lambda: ("27.0", (), "arm64")
    )
    first = _onnx_enhancer.configure_coreml_cache(options, [str(source)])
    assert all(isinstance(value, str) for value in first.values())
    assert Path(first["ModelCacheDirectory"]).parent == cache / "coreml"
    assert _onnx_enhancer.configure_coreml_cache(first, [str(source)]) == first

    device = _onnx_enhancer.configure_coreml_cache(
        {**options, "MLComputeUnits": "CPUAndGPU"}, [str(source)]
    )
    assert device["ModelCacheDirectory"] != first["ModelCacheDirectory"]
    monkeypatch.setattr(onnxruntime, "__version__", "different-runtime")
    runtime = _onnx_enhancer.configure_coreml_cache(options, [str(source)])
    assert runtime["ModelCacheDirectory"] != first["ModelCacheDirectory"]
    monkeypatch.setattr(
        _onnx_enhancer.platform, "mac_ver", lambda: ("27.1", (), "arm64")
    )
    macos = _onnx_enhancer.configure_coreml_cache(options, [str(source)])
    assert macos["ModelCacheDirectory"] != runtime["ModelCacheDirectory"]
    _shape_model(source, index=3)
    changed = _onnx_enhancer.configure_coreml_cache(options, [str(source)])
    assert changed["ModelCacheDirectory"] != macos["ModelCacheDirectory"]
    assert "ModelCacheDirectory" not in options


def test_provider_cache_fingerprint_covers_all_pack_models(model_setup):
    source, _cache = model_setup
    recognition = source.parent / "w600k_r50.onnx"
    landmark = source.parent / "2d106det.onnx"
    recognition.write_bytes(b"recognition weights")
    landmark.write_bytes(b"landmark weights")
    paths = [str(source), str(recognition), str(landmark)]
    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    initial = _onnx_enhancer.build_provider_config(providers, model_paths=paths)
    assert initial[1] == "CPUExecutionProvider"
    assert (
        _onnx_enhancer.build_provider_config(
            providers, model_paths=list(reversed(paths))
        )
        == initial
    )
    landmark.write_bytes(b"different landmark weights")
    updated = _onnx_enhancer.build_provider_config(providers, model_paths=paths)
    assert updated[0][1]["ModelCacheDirectory"] != initial[0][1]["ModelCacheDirectory"]


def test_enhancer_session_caches_the_final_rewritten_model(model_setup, monkeypatch):
    source, _cache = model_setup
    model = onnx.load(source)
    dimensions = model.graph.input[0].type.tensor_type.shape.dim
    dimensions[2].dim_value = 4
    dimensions[3].dim_value = 5
    onnx.save(model, source)
    monkeypatch.setattr(
        _onnx_enhancer.modules.globals,
        "execution_providers",
        [
            (
                "CoreMLExecutionProvider",
                {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
            )
        ],
    )
    session = Mock()
    factory = Mock(return_value=session)
    monkeypatch.setattr(onnxruntime, "InferenceSession", factory)

    assert _onnx_enhancer.create_onnx_session(str(source)) is session
    final_path = factory.call_args.args[0]
    assert final_path != str(source)
    assert _folded_dimension(final_path) == 4
    options = factory.call_args.kwargs["providers"][0][1]
    assert options == _onnx_enhancer.configure_coreml_cache(
        {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}, [final_path]
    )
    assert options["ModelCacheDirectory"] != _onnx_enhancer.configure_coreml_cache(
        {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}, [str(source)]
    )["ModelCacheDirectory"]


def test_provider_defaults_preserve_explicit_coreml_tuple_options(model_setup):
    source, _cache = model_setup
    explicit = {
        "ModelFormat": "MLProgram",
        "MLComputeUnits": "CPUAndGPU",
        "SpecializationStrategy": "Default",
        "EnableOnSubgraphs": "0",
    }
    original = dict(explicit)
    configured = _onnx_enhancer.build_provider_config(
        [("CoreMLExecutionProvider", explicit)],
        model_paths=[str(source)],
        coreml_options={
            "SpecializationStrategy": "FastPrediction",
            "EnableOnSubgraphs": "1",
        },
    )
    assert configured[0][1] == _onnx_enhancer.configure_coreml_cache(
        original, [str(source)]
    )
    assert explicit == original


def test_detection_override_preserves_options_and_recomputes_cache_identity(
    model_setup, monkeypatch
):
    source, _cache = model_setup
    original = _onnx_enhancer.build_provider_config(
        ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        model_paths=[str(source)],
        coreml_options={
            "SpecializationStrategy": "FastPrediction",
            "EnableOnSubgraphs": "1",
        },
    )
    initial_options = dict(original[0][1])
    session = Mock()
    session_factory = Mock(return_value=session)
    monkeypatch.setattr(onnxruntime, "InferenceSession", session_factory)
    analyser = SimpleNamespace(
        det_model=SimpleNamespace(model_file=str(source), session=None)
    )

    face_analyser._optimize_det_model(analyser, original, (320, 320))

    assert analyser.det_model.session is session
    resulting = session_factory.call_args.kwargs["providers"]
    options = resulting[0][1]
    assert options["MLComputeUnits"] == "CPUAndGPU"
    assert options["SpecializationStrategy"] == "FastPrediction"
    assert options["EnableOnSubgraphs"] == "1"
    assert options["AllowLowPrecisionAccumulationOnGPU"] == "1"
    assert options["ModelCacheDirectory"] != initial_options["ModelCacheDirectory"]
    assert original[0][1] == initial_options
    assert resulting[1] == "CPUExecutionProvider"
