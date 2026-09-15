"""ONNX model optimizations for CoreML execution on Apple Silicon.

CoreML's EP can leave parts of a model on CPU, introducing CPU↔ANE
round-trips.  This module folds dynamic ``Shape`` → ``Gather`` chains into
constants so the whole graph stays on the ANE:

**Shape/Gather constant folding** — Dynamic ``Shape`` → ``Gather`` chains
(e.g. for FPN upsample target sizes in RetinaFace) force ops onto CPU even
when the input dimensions are known at load time.  We run ONNX shape
inference with the known input size and replace these chains with constants.
Float32-noise-level differences only (max ~6e-6).

The result is cached on disk with a ``_coreml`` suffix so the rewrite cost is
paid only once per model.
"""

import os
import platform

import numpy as np

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"


def optimize_for_coreml(model_path: str, input_shape: tuple = None) -> str:
    """Return path to a CoreML-optimized ONNX model.

    Applies Shape/Gather constant folding (when an input shape is known) and
    caches the result next to the original model (with ``_coreml`` suffix).

    Args:
        model_path: Path to the original ONNX model.
        input_shape: Optional fixed input shape (e.g. ``(1, 3, 640, 640)``).
            When provided, enables Shape/Gather constant folding.

    Returns the optimized path, or the original path if no optimization
    applies or we're not on Apple Silicon.
    """
    if not IS_APPLE_SILICON:
        return model_path

    base, ext = os.path.splitext(model_path)
    optimized_path = f"{base}_coreml{ext}"
    if os.path.exists(optimized_path):
        if os.path.getmtime(optimized_path) >= os.path.getmtime(model_path):
            return optimized_path

    import onnx
    from onnx import numpy_helper

    model = onnx.load(model_path)

    if not _fold_shape_gather(model, input_shape):
        return model_path

    # Preserve insightface's emap convention: the INSwapper class reads
    # graph.initializer[-1] as the embedding map.  If the original model
    # had a (512, 512) matrix as its last initializer, keep it last.
    _preserve_emap_position(model, numpy_helper)

    onnx.save(model, optimized_path)
    return optimized_path


# ---------------------------------------------------------------------------
# Fold Shape → Gather chains into constants
# ---------------------------------------------------------------------------

def _fold_shape_gather(model, input_shape) -> bool:
    """Replace dynamic Shape→Gather chains with constants when input size is known.

    Only removes a Shape node when ALL of its consumers are Gather nodes
    that are also being folded.  This prevents breaking graphs where
    a Shape output feeds into other ops as well.
    """
    if input_shape is None:
        return False

    from onnx import numpy_helper, shape_inference

    graph = model.graph

    # Set fixed input dimensions for shape inference
    inp = graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    for i, size in enumerate(input_shape):
        if i < len(dims):
            dims[i].dim_value = size

    try:
        model_inferred = shape_inference.infer_shapes(model)
    except Exception:
        return False

    # Extract inferred shapes
    value_shapes = {}
    for vi in list(model_inferred.graph.value_info) + list(graph.input) + list(graph.output):
        shape_dims = vi.type.tensor_type.shape.dim
        shape = []
        for d in shape_dims:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(None)
        value_shapes[vi.name] = shape

    inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    # Build consumer map: output_name → list of consuming nodes
    consumers = {}
    for node in graph.node:
        for i in node.input:
            consumers.setdefault(i, []).append(node)

    # Also check graph outputs — an output name consumed by the graph
    # output list must not be removed
    graph_output_names = {o.name for o in graph.output}

    # Find Shape nodes with fully-known output
    shape_constants = {}
    for node in graph.node:
        if node.op_type == "Shape":
            inp_shape = value_shapes.get(node.input[0])
            if inp_shape and all(isinstance(d, int) for d in inp_shape):
                shape_constants[node.output[0]] = np.array(inp_shape, dtype=np.int64)

    if not shape_constants:
        return False

    # Find Gather nodes consuming Shape constants
    gather_constants = {}
    for node in graph.node:
        if node.op_type == "Gather" and node.input[0] in shape_constants:
            idx_name = node.input[1]
            if idx_name in inits:
                idx = int(inits[idx_name])
                val = int(shape_constants[node.input[0]][idx])
                gather_constants[node.output[0]] = np.array(val, dtype=np.int64)

    if not gather_constants:
        return False

    # Determine which Gather nodes to fold (always safe — we replace
    # the output with a constant initializer)
    gather_remove_ids = set()
    for node in graph.node:
        if node.op_type == "Gather" and node.output[0] in gather_constants:
            gather_remove_ids.add(id(node))

    # Determine which Shape nodes are safe to remove: only if ALL
    # consumers of the Shape output are Gather nodes being folded,
    # and the output isn't a graph output.
    shape_remove_ids = set()
    for node in graph.node:
        if node.op_type == "Shape" and node.output[0] in shape_constants:
            out_name = node.output[0]
            if out_name in graph_output_names:
                continue
            node_consumers = consumers.get(out_name, [])
            if all(id(c) in gather_remove_ids for c in node_consumers):
                shape_remove_ids.add(id(node))

    remove_ids = gather_remove_ids | shape_remove_ids

    # Add Gather output constants as initializers
    existing = {i.name for i in graph.initializer}
    for name, val in gather_constants.items():
        if name not in existing:
            graph.initializer.append(numpy_helper.from_array(val, name=name))

    new_nodes = [n for n in graph.node if id(n) not in remove_ids]
    del graph.node[:]
    graph.node.extend(new_nodes)
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preserve_emap_position(model, numpy_helper):
    """Keep the insightface emap (512×512 matrix) as the last initializer."""
    graph = model.graph
    emap_init = None
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if len(arr.shape) == 2 and arr.shape[0] == 512 and arr.shape[1] == 512:
            emap_init = init
            break

    if emap_init is not None:
        inits = [i for i in graph.initializer if i.name != emap_init.name]
        del graph.initializer[:]
        graph.initializer.extend(inits)
        graph.initializer.append(emap_init)
