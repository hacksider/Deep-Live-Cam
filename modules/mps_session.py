"""Drop-in replacement for onnxruntime.InferenceSession that runs inference
on Apple MPS (Metal Performance Shaders) via PyTorch for significant speedup
on Apple Silicon Macs.

Usage: pass an MPSSession instance anywhere an onnxruntime session is expected.
It exposes the same .run(), .get_inputs(), .get_outputs(), and .get_providers()
interface used by insightface's INSwapper.
"""

import numpy as np
import platform

_MPS_AVAILABLE = False
_torch = None
_convert = None

if platform.system() == "Darwin" and platform.machine() == "arm64":
    try:
        import torch as _torch
        from onnx2torch import convert as _convert
        import onnx as _onnx
        if _torch.backends.mps.is_available():
            _MPS_AVAILABLE = True
    except ImportError:
        pass


class _FakeIO:
    """Mimics onnxruntime input/output metadata."""
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class MPSSession:
    """PyTorch MPS-backed inference session compatible with insightface."""

    def __init__(self, model_path, providers=None):
        if not _MPS_AVAILABLE:
            raise RuntimeError("MPS not available")

        self._model = _convert(model_path)
        self._model.eval()
        self._model.to("mps")
        self._providers = ["MPSExecutionProvider"]
        self._provider_options = [{}]

        # Discover input/output metadata from the ONNX file
        onnx_model = _onnx.load(model_path)
        self._inputs = []
        for inp in onnx_model.graph.input:
            shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
            self._inputs.append(_FakeIO(inp.name, shape))
        self._outputs = []
        self._output_names = []
        for out in onnx_model.graph.output:
            shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
            self._outputs.append(_FakeIO(out.name, shape))
            self._output_names.append(out.name)

        # Warmup run
        dummy_inputs = {}
        for inp in self._inputs:
            s = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            dummy_inputs[inp.name] = np.random.randn(*s).astype(np.float32)
        self.run([o.name for o in self._outputs], dummy_inputs)

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def get_providers(self):
        return self._providers

    def run(self, output_names, input_feed, run_options=None):
        tensors = []
        for inp in self._inputs:
            arr = input_feed[inp.name]
            t = _torch.from_numpy(arr).to("mps")
            tensors.append(t)

        with _torch.no_grad():
            out = self._model(*tensors)
            _torch.mps.synchronize()

        # Build name-to-result mapping
        if isinstance(out, _torch.Tensor):
            all_outputs = {self._output_names[0]: out.cpu().numpy()}
        else:
            all_outputs = {
                name: o.cpu().numpy()
                for name, o in zip(self._output_names, out)
            }

        # Return outputs in the order requested by output_names
        if output_names is None:
            return list(all_outputs.values())
        return [all_outputs[name] for name in output_names]


def is_mps_available():
    return _MPS_AVAILABLE
