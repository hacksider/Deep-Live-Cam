"""CoreML session wrapper with ONNX Runtime InferenceSession-compatible interface.

Allows swapping face_swapper.session with a CoreML-backed implementation that
exposes the same .get_inputs() / .run() API insightface expects.

Only available on macOS. Import guard ensures this module is a no-op on other platforms.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np


class _NodeArg:
    """Minimal stand-in for onnxruntime.NodeArg."""

    def __init__(self, name: str, shape: List, type_str: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = type_str


class CoreMLSessionWrapper:
    """Wraps a coremltools MLModel to look like an onnxruntime.InferenceSession.

    Usage::

        from modules.coreml_session import CoreMLSessionWrapper
        wrapper = CoreMLSessionWrapper.load("models/inswapper_128.mlpackage")
        if wrapper:
            face_swapper.session = wrapper
    """

    def __init__(self, mlmodel: Any, input_specs: List[_NodeArg], output_specs: List[_NodeArg]) -> None:
        self._model = mlmodel
        self._inputs = input_specs
        self._outputs = output_specs

    @classmethod
    def load(cls, mlpackage_path: str) -> Optional["CoreMLSessionWrapper"]:
        """Load an .mlpackage and return a wrapper, or None on failure."""
        if sys.platform != "darwin":
            return None
        try:
            import coremltools as ct
        except ImportError:
            return None

        try:
            mlmodel = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)
            spec = mlmodel.get_spec()

            input_specs = [
                _NodeArg(
                    inp.name,
                    list(inp.type.multiArrayType.shape),
                )
                for inp in spec.description.input
            ]
            output_specs = [
                _NodeArg(
                    out.name,
                    list(out.type.multiArrayType.shape),
                )
                for out in spec.description.output
            ]
            return cls(mlmodel, input_specs, output_specs)
        except Exception as exc:
            print(f"CoreMLSessionWrapper: failed to load {mlpackage_path}: {exc}")
            return None

    def get_inputs(self) -> List[_NodeArg]:
        return self._inputs

    def get_outputs(self) -> List[_NodeArg]:
        return self._outputs

    def run(self, output_names: Optional[List[str]], input_feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Run inference, returning outputs as a list of numpy arrays."""
        import coremltools as ct

        # coremltools expects inputs as dict of {name: numpy_array}
        # Cast to float32 — CoreML FP16 model handles precision internally
        coreml_input = {k: v.astype(np.float32) for k, v in input_feed.items()}
        predictions = self._model.predict(coreml_input)

        # Return outputs in spec order (matching ONNX Runtime behaviour)
        if output_names:
            return [np.array(predictions[name]) for name in output_names]
        return [np.array(predictions[spec.name]) for spec in self._outputs]
