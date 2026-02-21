"""MLX-native inswapper forward pass for Apple Silicon.

Loads weights directly from inswapper_128_fp16.onnx and runs inference
using MLX (native unified memory, no ONNX Runtime overhead).

Architecture: StyleGAN2-like encoder-decoder with 6 AdaIN style blocks.
  Encoder:      4 conv layers (stride-2 downsampling at conv3/conv4)
  Style blocks: 6 × (conv + instance_norm + AdaIN) × 2 with residual
  Decoder:      2× bilinear upsample + 4 conv layers

Weight format: ONNX NCHW → MLX NHWC (transposed on load).
Only available on macOS. Raises ImportError on other platforms.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional

import numpy as np

if sys.platform != "darwin":
    raise ImportError("mlx_inswapper is macOS-only")

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reflect_pad(x: mx.array, pad_h: int, pad_w: int) -> mx.array:
    """Reflect-pad (N, H, W, C) by *pad_h* rows and *pad_w* cols on each side."""
    if pad_h > 0:
        top = x[:, 1:pad_h + 1, :, :][:, ::-1, :, :]
        bot = x[:, -(pad_h + 1):-1, :, :][:, ::-1, :, :]
        x = mx.concatenate([top, x, bot], axis=1)
    if pad_w > 0:
        left = x[:, :, 1:pad_w + 1, :][:, :, ::-1, :]
        right = x[:, :, -(pad_w + 1):-1, :][:, :, ::-1, :]
        x = mx.concatenate([left, x, right], axis=2)
    return x


def _instance_norm(x: mx.array, eps: float = 1.192e-7) -> mx.array:
    """Instance norm over spatial dims (H=1, W=2) for (N, H, W, C) input.

    Matches the ONNX graph's cast-to-float32 precision pattern.
    """
    x_f = x.astype(mx.float32)
    mean = mx.mean(x_f, axis=(1, 2), keepdims=True)
    centered = x_f - mean
    var = mx.mean(centered * centered, axis=(1, 2), keepdims=True)
    normed = centered * mx.rsqrt(var + eps)
    return normed.astype(x.dtype)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLXInswapper:
    """Inswapper-128 forward pass implemented natively in MLX.

    Instantiate via :meth:`from_onnx`, then call :meth:`forward`.
    """

    def __init__(self, weights: Dict[str, mx.array]) -> None:
        self._w = weights
        # Build upsampler once (bilinear 2×, align_corners=False ≈ pytorch_half_pixel)
        self._upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_onnx(cls, onnx_path: str) -> "MLXInswapper":
        """Load weights from an inswapper_128_fp16.onnx file."""
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exc:
            raise ImportError("onnx is required to load MLXInswapper") from exc

        model = onnx.load(onnx_path)
        raw: Dict[str, np.ndarray] = {
            t.name: numpy_helper.to_array(t) for t in model.graph.initializer
        }

        def _conv(name: str) -> mx.array:
            # ONNX (out, in, kH, kW) → MLX (out, kH, kW, in)
            return mx.array(raw[name].astype(np.float16).transpose(0, 2, 3, 1))

        def _vec(name: str) -> mx.array:
            return mx.array(raw[name].astype(np.float16))

        w: Dict[str, mx.array] = {}

        # Encoder
        w["enc1_w"] = _conv("onnx::Conv_833")
        w["enc1_b"] = _vec("onnx::Conv_834")
        w["enc2_w"] = _conv("onnx::Conv_836")
        w["enc2_b"] = _vec("onnx::Conv_837")
        w["enc3_w"] = _conv("onnx::Conv_839")
        w["enc3_b"] = _vec("onnx::Conv_840")
        w["enc4_w"] = _conv("onnx::Conv_842")
        w["enc4_b"] = _vec("onnx::Conv_843")

        # Style blocks (6 blocks, 2 conv+style layers each)
        for i in range(6):
            for c in (1, 2):
                w[f"s{i}c{c}_w"] = _conv(f"styles.{i}.conv{c}.1.weight")
                w[f"s{i}c{c}_b"] = _vec(f"styles.{i}.conv{c}.1.bias")
                w[f"s{i}f{c}_w"] = _vec(f"styles.{i}.style{c}.linear.weight")
                w[f"s{i}f{c}_b"] = _vec(f"styles.{i}.style{c}.linear.bias")

        # Decoder
        w["dec1_w"] = _conv("onnx::Conv_845")
        w["dec1_b"] = _vec("onnx::Conv_846")
        w["dec2_w"] = _conv("onnx::Conv_848")
        w["dec2_b"] = _vec("onnx::Conv_849")
        w["dec3_w"] = _conv("onnx::Conv_851")
        w["dec3_b"] = _vec("onnx::Conv_852")
        w["dec4_w"] = _conv("up0.1.weight")
        w["dec4_b"] = _vec("up0.1.bias")

        mx.eval(w)
        return cls(w)

    # ------------------------------------------------------------------
    # Forward pass helpers
    # ------------------------------------------------------------------

    def _conv2d(self, x: mx.array, wk: str, bk: str,
                stride: int = 1, padding: int = 0) -> mx.array:
        y = mx.conv2d(x, self._w[wk], stride=stride, padding=padding)
        return y + self._w[bk][None, None, None, :]

    def _style_block(self, x: mx.array, source: mx.array, i: int) -> mx.array:
        """One AdaIN style residual block."""
        residual = x
        for c in (1, 2):
            # Reflect pad 1 + conv
            y = _reflect_pad(x, 1, 1)
            y = self._conv2d(y, f"s{i}c{c}_w", f"s{i}c{c}_b")

            # Instance norm
            y = _instance_norm(y)

            # AdaIN: linear(source) → scale + bias (split 2048 → 1024 + 1024)
            style = source @ self._w[f"s{i}f{c}_w"].T + self._w[f"s{i}f{c}_b"]
            scale = style[:, :1024][:, None, None, :]   # (N, 1, 1, 1024)
            bias  = style[:, 1024:][:, None, None, :]   # (N, 1, 1, 1024)
            y = scale * y + bias

            # ReLU after conv1; conv2 feeds directly into residual add
            if c == 1:
                y = nn.relu(y)
            x = y

        return residual + x

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(self, target: mx.array, source: mx.array) -> mx.array:
        """Run the inswapper forward pass.

        Args:
            target: ``(N, H, W, 3)`` float16 in ``[-1, 1]``
                    (the normalised blob InsightFace produces, converted to NHWC).
            source: ``(N, 512)`` float16 post-emap face embedding.

        Returns:
            ``(N, H, W, 3)`` float32 in ``[0, 1]``.
        """
        x = target.astype(mx.float16)
        s = source.astype(mx.float16)

        # --- Encoder ---
        x = _reflect_pad(x, 3, 3)                                           # [N, 134, 134, 3]
        x = nn.leaky_relu(self._conv2d(x, "enc1_w", "enc1_b"), 0.2)        # [N, 128, 128, 128]
        x = nn.leaky_relu(self._conv2d(x, "enc2_w", "enc2_b", padding=1), 0.2)          # [N, 128, 128, 256]
        x = nn.leaky_relu(self._conv2d(x, "enc3_w", "enc3_b", stride=2, padding=1), 0.2)  # [N, 64, 64, 512]
        x = nn.leaky_relu(self._conv2d(x, "enc4_w", "enc4_b", stride=2, padding=1), 0.2)  # [N, 32, 32, 1024]

        # --- Style blocks ---
        for i in range(6):
            x = self._style_block(x, s, i)

        # --- Decoder ---
        x = self._upsample(x)                                               # [N, 64, 64, 1024]
        x = nn.leaky_relu(self._conv2d(x, "dec1_w", "dec1_b", padding=1), 0.2)  # [N, 64, 64, 512]
        x = self._upsample(x)                                               # [N, 128, 128, 512]
        x = nn.leaky_relu(self._conv2d(x, "dec2_w", "dec2_b", padding=1), 0.2)  # [N, 128, 128, 256]
        x = nn.leaky_relu(self._conv2d(x, "dec3_w", "dec3_b", padding=1), 0.2)  # [N, 128, 128, 128]
        x = _reflect_pad(x, 3, 3)                                           # [N, 134, 134, 128]
        x = mx.tanh(self._conv2d(x, "dec4_w", "dec4_b").astype(mx.float32))     # [N, 128, 128, 3]
        return (x + 1.0) / 2.0


# ---------------------------------------------------------------------------
# ONNX Runtime-compatible session wrapper
# ---------------------------------------------------------------------------

class _NodeArg:
    """Minimal stand-in for onnxruntime.NodeArg."""
    def __init__(self, name: str, shape: List, type_str: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = type_str


class MLXSessionWrapper:
    """Wraps :class:`MLXInswapper` with the same ``get_inputs()`` / ``run()`` interface
    as ``onnxruntime.InferenceSession``.

    Drop-in replacement for ``face_swapper.session``::

        from modules.mlx_inswapper import MLXSessionWrapper
        wrapper = MLXSessionWrapper.load("models/inswapper_128_fp16.onnx")
        if wrapper:
            face_swapper.session = wrapper

    Input convention (NCHW, matching InsightFace):
        - ``"target"``: ``(1, 3, 128, 128)`` float32 blob in ``[-1, 1]``
        - ``"source"``: ``(1, 512)`` float32 post-emap face embedding

    Output:
        - ``[0]``: ``(1, 3, 128, 128)`` float32 in ``[0, 1]``
    """

    def __init__(self, model: MLXInswapper) -> None:
        self._model = model
        self._inputs = [
            _NodeArg("target", [1, 3, 128, 128]),
            _NodeArg("source", [1, 512]),
        ]
        self._outputs = [_NodeArg("output", [1, 3, 128, 128])]

    @classmethod
    def load(cls, onnx_path: str) -> Optional["MLXSessionWrapper"]:
        """Load from an ONNX file, returning ``None`` on failure."""
        if sys.platform != "darwin":
            return None
        try:
            model = MLXInswapper.from_onnx(onnx_path)
            return cls(model)
        except Exception as exc:
            print(f"MLXSessionWrapper: failed to load {onnx_path}: {exc}")
            return None

    def get_inputs(self) -> List[_NodeArg]:
        return self._inputs

    def get_outputs(self) -> List[_NodeArg]:
        return self._outputs

    def run(self, output_names, input_feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """Run inference. Converts NCHW numpy → NHWC MLX → NCHW numpy."""
        # NCHW float32 → NHWC float16
        target_nhwc = mx.array(
            input_feed["target"].transpose(0, 2, 3, 1).astype(np.float16)
        )
        source_mx = mx.array(input_feed["source"].astype(np.float16))

        out_nhwc = self._model.forward(target_nhwc, source_mx)
        mx.eval(out_nhwc)

        # NHWC → NCHW float32
        out_nchw = np.array(out_nhwc).transpose(0, 3, 1, 2).astype(np.float32)
        return [out_nchw]
