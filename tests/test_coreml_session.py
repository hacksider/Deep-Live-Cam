"""Tests for modules.coreml_session — CoreML session wrapper."""
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _NodeArg
# ---------------------------------------------------------------------------

def test_nodearg_attributes():
    from modules.coreml_session import _NodeArg
    node = _NodeArg("target", [1, 3, 128, 128], "tensor(float)")
    assert node.name == "target"
    assert node.shape == [1, 3, 128, 128]
    assert node.type == "tensor(float)"


def test_nodearg_default_type():
    from modules.coreml_session import _NodeArg
    node = _NodeArg("x", [1])
    assert node.type == "tensor(float)"


# ---------------------------------------------------------------------------
# CoreMLSessionWrapper.load — platform guard
# ---------------------------------------------------------------------------

def test_load_returns_none_on_non_darwin():
    from modules.coreml_session import CoreMLSessionWrapper
    with patch("modules.coreml_session.sys") as mock_sys:
        mock_sys.platform = "linux"
        result = CoreMLSessionWrapper.load("/fake/path.mlpackage")
    assert result is None


def test_load_returns_none_when_coremltools_missing():
    """If coremltools is not installed, load() returns None gracefully."""
    from modules.coreml_session import CoreMLSessionWrapper
    with patch("modules.coreml_session.sys") as mock_sys:
        mock_sys.platform = "darwin"
        with patch.dict(sys.modules, {"coremltools": None}):
            # Force ImportError on `import coremltools`
            with patch("builtins.__import__", side_effect=_import_raiser("coremltools")):
                result = CoreMLSessionWrapper.load("/fake/path.mlpackage")
    assert result is None


def _import_raiser(blocked_name):
    """Return an __import__ replacement that raises ImportError for *blocked_name*."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
    def _import(name, *args, **kwargs):
        if name == blocked_name:
            raise ImportError(f"No module named '{blocked_name}'")
        return real_import(name, *args, **kwargs)
    return _import


# ---------------------------------------------------------------------------
# CoreMLSessionWrapper — interface contract
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_coreml_wrapper():
    """Build a CoreMLSessionWrapper with a mocked MLModel."""
    from modules.coreml_session import CoreMLSessionWrapper, _NodeArg

    mock_mlmodel = MagicMock()
    # predict returns a dict matching output spec
    mock_mlmodel.predict.return_value = {
        "output": np.zeros((1, 3, 128, 128), dtype=np.float32),
    }

    input_specs = [
        _NodeArg("target", [1, 3, 128, 128]),
        _NodeArg("source", [1, 512]),
    ]
    output_specs = [_NodeArg("output", [1, 3, 128, 128])]
    return CoreMLSessionWrapper(mock_mlmodel, input_specs, output_specs)


def test_get_inputs_returns_list(mock_coreml_wrapper):
    inputs = mock_coreml_wrapper.get_inputs()
    assert len(inputs) == 2
    assert inputs[0].name == "target"
    assert inputs[1].name == "source"


def test_get_outputs_returns_list(mock_coreml_wrapper):
    outputs = mock_coreml_wrapper.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "output"


def _patch_ct_import(func):
    """Decorator to stub `import coremltools as ct` inside run()."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ct_mock = MagicMock()
        with patch.dict(sys.modules, {"coremltools": ct_mock}):
            return func(*args, **kwargs)
    return wrapper


@_patch_ct_import
def test_run_returns_numpy_arrays(mock_coreml_wrapper):
    feed = {
        "target": np.zeros((1, 3, 128, 128), dtype=np.float32),
        "source": np.zeros((1, 512), dtype=np.float32),
    }
    results = mock_coreml_wrapper.run(None, feed)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], np.ndarray)


@_patch_ct_import
def test_run_with_output_names(mock_coreml_wrapper):
    feed = {
        "target": np.zeros((1, 3, 128, 128), dtype=np.float32),
        "source": np.zeros((1, 512), dtype=np.float32),
    }
    results = mock_coreml_wrapper.run(["output"], feed)
    assert len(results) == 1


@_patch_ct_import
def test_run_casts_inputs_to_float32(mock_coreml_wrapper):
    """Ensure float64 inputs are cast to float32 before passing to CoreML."""
    feed = {
        "target": np.zeros((1, 3, 128, 128), dtype=np.float64),
        "source": np.zeros((1, 512), dtype=np.float64),
    }
    mock_coreml_wrapper.run(None, feed)

    call_args = mock_coreml_wrapper._model.predict.call_args[0][0]
    assert call_args["target"].dtype == np.float32
    assert call_args["source"].dtype == np.float32


# ---------------------------------------------------------------------------
# Conversion script — unit tests
# ---------------------------------------------------------------------------

def test_convert_missing_model(tmp_path):
    """convert() returns 1 when the ONNX model doesn't exist."""
    from scripts.convert_to_coreml import convert
    assert convert(str(tmp_path)) == 1


def test_convert_already_exists(tmp_path):
    """convert() returns 0 when the .mlpackage already exists."""
    # Create both the ONNX file and the mlpackage dir
    (tmp_path / "inswapper_128_fp16.onnx").touch()
    (tmp_path / "inswapper_128.mlpackage").mkdir()
    from scripts.convert_to_coreml import convert
    assert convert(str(tmp_path)) == 0
