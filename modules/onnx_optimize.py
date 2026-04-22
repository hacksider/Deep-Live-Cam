"""ONNX model optimizations for CoreML execution on Apple Silicon.

Each pass eliminates a different CPU↔ANE round-trip that ORT's CoreML EP
would otherwise introduce:

1. **Shape/Gather constant folding** — Dynamic ``Shape`` → ``Gather`` chains
   (e.g. for FPN upsample target sizes in RetinaFace) force ops onto CPU even
   when the input dimensions are known at load time.  We run ONNX shape
   inference with the known input size and replace these chains with constants.
   Float32-noise-level differences only (max ~6e-6).

2. **Pad(reflect) decomposition** — CoreML doesn't support ``Pad(mode=reflect)``.
   Models using reflect padding (e.g. inswapper_128) get split into many CoreML
   subgraphs with CPU fallbacks between each.  We rewrite each ``Pad(reflect)``
   as equivalent ``Slice`` + ``Concat`` ops that CoreML handles natively.
   Bit-for-bit identical output. (Fixed upstream in microsoft/onnxruntime#28073.)

3. **Split → Slice decomposition** — CoreML's EP doesn't support the ONNX
   ``Split`` op, causing partition boundaries in models with channel-wise
   splits (e.g. GFPGAN's SFT modulation). Each 2-way Split becomes two Slices.

4. **Scalar Gather widening** — ORT's CoreML EP rejects ``Gather`` nodes with
   rank-0 (scalar) indices. StyleGAN-derived models (GFPGAN) slice per-layer
   style codes using exactly this pattern. We widen each scalar index to
   ``[1]`` and squeeze the added axis on the Gather output.
   (Filed upstream as microsoft/onnxruntime#28180.)

All passes are cached on disk with a ``_coreml`` suffix so the rewrite cost
is paid only once per model.
"""

import os
import platform

import numpy as np

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"


def optimize_for_coreml(model_path: str, input_shape: tuple = None) -> str:
    """Return path to a CoreML-optimized ONNX model.

    Applies all applicable optimizations and caches the result next to
    the original model (with ``_coreml`` suffix).

    Args:
        model_path: Path to the original ONNX model.
        input_shape: Optional fixed input shape (e.g. ``(1, 3, 640, 640)``).
            When provided, enables Shape/Gather constant folding.

    Returns the optimized path, or the original path if no optimizations
    apply or we're not on Apple Silicon.
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
    changed = False

    if _fold_shape_gather(model, input_shape):
        changed = True

    # TODO(ort>=1.26): drop this pass. Fixed upstream by microsoft/onnxruntime#28073.
    if _decompose_reflect_pad(model):
        changed = True

    if _decompose_split(model):
        changed = True

    # TODO: drop this pass once microsoft/onnxruntime#28180 ships. The CoreML
    # Gather op builder rejects rank-0 (scalar) indices; we widen them to [1]
    # + Squeeze so StyleGAN-family models (GFPGAN) stay on ANE.
    if _rewrite_scalar_gather(model):
        changed = True

    if not changed:
        return model_path

    # Preserve insightface's emap convention: the INSwapper class reads
    # graph.initializer[-1] as the embedding map.  If the original model
    # had a (512, 512) matrix as its last initializer, keep it last.
    _preserve_emap_position(model, numpy_helper)

    onnx.save(model, optimized_path)
    return optimized_path


# ---------------------------------------------------------------------------
# Pass 1: Fold Shape → Gather chains into constants
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
# Pass 2: Decompose Pad(reflect) → Slice + Concat
#
# TEMPORARY: fixed upstream in microsoft/onnxruntime#28073 (merged 2026-04-20).
# Once the ORT floor is >= 1.26.0, MLProgram handles Pad(mode=reflect) natively
# via MIL tensor_operation.pad and this entire pass can be deleted.
# ---------------------------------------------------------------------------

def _decompose_reflect_pad(model) -> bool:
    """Rewrite Pad(reflect) as Slice+Concat sequences CoreML can handle."""
    from onnx import numpy_helper, helper

    graph = model.graph
    inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    reflect_pads = []
    for node in graph.node:
        if node.op_type == "Pad":
            mode = "constant"
            for attr in node.attribute:
                if attr.name == "mode":
                    mode = attr.s.decode()
            if mode == "reflect" and len(node.input) > 1 and node.input[1] in inits:
                reflect_pads.append(node)

    if not reflect_pads:
        return False

    existing_names = {i.name for i in graph.initializer}

    def ensure_const(name, value):
        if name not in existing_names:
            graph.initializer.append(
                numpy_helper.from_array(np.array(value, dtype=np.int64), name=name)
            )
            existing_names.add(name)

    ensure_const("_rp_ax2", [2])
    ensure_const("_rp_ax3", [3])

    max_pad = 0
    for node in reflect_pads:
        pads = inits[node.input[1]].tolist()
        max_pad = max(max_pad, int(pads[2]), int(pads[3]))

    for v in range(1, max_pad + 2):
        ensure_const(f"_rp_p{v}", [v])
        ensure_const(f"_rp_n{v}", [-v])

    _counter = [0]

    def uid():
        _counter[0] += 1
        return _counter[0]

    pad_ids = {id(n) for n in reflect_pads}
    pad_init_names = set()

    new_nodes = []
    for node in graph.node:
        if id(node) not in pad_ids:
            new_nodes.append(node)
            continue

        pads = inits[node.input[1]].tolist()
        h_pad, w_pad = int(pads[2]), int(pads[3])

        for inp in node.input[1:]:
            if inp in inits:
                pad_init_names.add(inp)

        current = node.input[0]

        if h_pad > 0:
            top = []
            for i in range(h_pad, 0, -1):
                name = f"_rp_t{uid()}"
                new_nodes.append(helper.make_node(
                    "Slice",
                    inputs=[current, f"_rp_p{i}", f"_rp_p{i+1}", "_rp_ax2"],
                    outputs=[name],
                ))
                top.append(name)

            bot = []
            for i in range(1, h_pad + 1):
                name = f"_rp_b{uid()}"
                new_nodes.append(helper.make_node(
                    "Slice",
                    inputs=[current, f"_rp_n{i+1}", f"_rp_n{i}", "_rp_ax2"],
                    outputs=[name],
                ))
                bot.append(name)

            h_out = f"_rp_h{uid()}"
            new_nodes.append(helper.make_node(
                "Concat", inputs=top + [current] + bot, outputs=[h_out], axis=2
            ))
            current = h_out

        if w_pad > 0:
            left = []
            for i in range(w_pad, 0, -1):
                name = f"_rp_l{uid()}"
                new_nodes.append(helper.make_node(
                    "Slice",
                    inputs=[current, f"_rp_p{i}", f"_rp_p{i+1}", "_rp_ax3"],
                    outputs=[name],
                ))
                left.append(name)

            right = []
            for i in range(1, w_pad + 1):
                name = f"_rp_r{uid()}"
                new_nodes.append(helper.make_node(
                    "Slice",
                    inputs=[current, f"_rp_n{i+1}", f"_rp_n{i}", "_rp_ax3"],
                    outputs=[name],
                ))
                right.append(name)

            new_nodes.append(helper.make_node(
                "Concat",
                inputs=left + [current] + right,
                outputs=[node.output[0]],
                axis=3,
            ))
        elif h_pad > 0:
            new_nodes.append(helper.make_node(
                "Identity", inputs=[current], outputs=[node.output[0]]
            ))

    # Remove old Pad initializers
    clean_inits = [i for i in graph.initializer if i.name not in pad_init_names]
    del graph.initializer[:]
    graph.initializer.extend(clean_inits)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return True


# ---------------------------------------------------------------------------
# Pass 3: Decompose Split → Slice pairs
# ---------------------------------------------------------------------------

def _decompose_split(model) -> bool:
    """Rewrite Split(axis=1) as Slice pairs that CoreML can handle.

    CoreML's EP doesn't support the ONNX ``Split`` op, causing partition
    boundaries in models that use channel-wise splits (e.g. GFPGAN's SFT
    modulation layers).  Each Split with two outputs becomes two Slice ops.
    """
    from onnx import numpy_helper, helper

    graph = model.graph

    splits = []
    for node in graph.node:
        if node.op_type == "Split":
            axis = 0
            split_sizes = []
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
                if attr.name == "split":
                    split_sizes = list(attr.ints)
            if axis == 1 and len(split_sizes) == 2 and len(node.output) == 2:
                splits.append((node, split_sizes))

    if not splits:
        return False

    existing = {i.name for i in graph.initializer}

    def ensure_const(name, value):
        if name not in existing:
            graph.initializer.append(
                numpy_helper.from_array(np.array(value, dtype=np.int64), name=name)
            )
            existing.add(name)

    ensure_const("_sp_ax1", [1])

    # Collect all needed boundary constants
    for _, (a, b) in splits:
        ensure_const(f"_sp_s0", [0])
        ensure_const(f"_sp_s{a}", [a])
        ensure_const(f"_sp_s{a + b}", [a + b])

    split_ids = {id(node) for node, _ in splits}
    replacements = {}
    for node, (a, b) in splits:
        slice0 = helper.make_node(
            "Slice",
            inputs=[node.input[0], "_sp_s0", f"_sp_s{a}", "_sp_ax1"],
            outputs=[node.output[0]],
        )
        slice1 = helper.make_node(
            "Slice",
            inputs=[node.input[0], f"_sp_s{a}", f"_sp_s{a + b}", "_sp_ax1"],
            outputs=[node.output[1]],
        )
        replacements[id(node)] = [slice0, slice1]

    new_nodes = []
    for node in graph.node:
        if id(node) in split_ids:
            new_nodes.extend(replacements[id(node)])
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    return True


# ---------------------------------------------------------------------------
# Pass 4: Widen scalar Gather indices to [1] + Squeeze
#
# TEMPORARY: filed upstream as microsoft/onnxruntime#28180. ORT's CoreML EP
# GatherOpBuilder::IsOpSupportedImpl rejects rank-0 (scalar) indices with
# `Gather does not support scalar 'indices'`. The builder's own comment
# describes the workaround (promote to [1], squeeze the added axis) but
# doesn't apply it. We do the same thing at the ONNX level so StyleGAN-
# family models (GFPGAN is the hot example — 16 per-layer style-code
# slices) don't split the CoreML subgraph. Once the upstream fix ships
# and the ORT floor is raised, delete this pass.
# ---------------------------------------------------------------------------

def _rewrite_scalar_gather(model) -> bool:
    """Rewrite Gather(data, scalar_idx) as Gather(data, [scalar_idx]) + Squeeze.

    Only touches Gather nodes whose index is a rank-0 int64 constant or
    initializer; everything else passes through unchanged. The rewrite
    is semantically identical — indices get an added leading axis, the
    Squeeze removes it after the gather.
    """
    from onnx import numpy_helper, helper, TensorProto

    graph = model.graph

    # Opset 13 moved Squeeze's axes from attribute to input.
    opset = next(
        (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")),
        11,
    )

    const_values = {}
    for n in graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    const_values[n.output[0]] = a.t
    init_values = {i.name: i for i in graph.initializer}

    def scalar_int64(name):
        """Return int value if `name` resolves to a rank-0 int64 constant, else None."""
        tensor = const_values.get(name) or init_values.get(name)
        if tensor is None or tensor.data_type != TensorProto.INT64:
            return None
        arr = numpy_helper.to_array(tensor)
        return int(arr) if arr.ndim == 0 else None

    rewrote = 0
    new_nodes = []
    for n in graph.node:
        if n.op_type == "Gather":
            val = scalar_int64(n.input[1])
            if val is not None:
                axis = next((a.i for a in n.attribute if a.name == "axis"), 0)
                idx_1d_name = f"{n.input[1]}_1d_{rewrote}"
                idx_const = helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[idx_1d_name],
                    value=helper.make_tensor(idx_1d_name, TensorProto.INT64, [1], [val]),
                )
                gather_out = f"{n.output[0]}_pre_squeeze_{rewrote}"
                new_gather = helper.make_node(
                    "Gather",
                    inputs=[n.input[0], idx_1d_name],
                    outputs=[gather_out],
                    name=n.name,
                    axis=axis,
                )
                if opset < 13:
                    squeeze = helper.make_node(
                        "Squeeze",
                        inputs=[gather_out],
                        outputs=[n.output[0]],
                        name=(n.name or "gather") + "_squeeze",
                        axes=[axis],
                    )
                    new_nodes.extend([idx_const, new_gather, squeeze])
                else:
                    axes_name = f"{idx_1d_name}_sq_axes"
                    axes_const = helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=[axes_name],
                        value=helper.make_tensor(axes_name, TensorProto.INT64, [1], [axis]),
                    )
                    squeeze = helper.make_node(
                        "Squeeze",
                        inputs=[gather_out, axes_name],
                        outputs=[n.output[0]],
                        name=(n.name or "gather") + "_squeeze",
                    )
                    new_nodes.extend([idx_const, axes_const, new_gather, squeeze])
                rewrote += 1
                continue
        new_nodes.append(n)

    if rewrote == 0:
        return False

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
        if not init.name.startswith("_rp_"):
            arr = numpy_helper.to_array(init)
            if len(arr.shape) == 2 and arr.shape[0] == 512 and arr.shape[1] == 512:
                emap_init = init
                break

    if emap_init is not None:
        inits = [i for i in graph.initializer if i.name != emap_init.name]
        del graph.initializer[:]
        graph.initializer.extend(inits)
        graph.initializer.append(emap_init)
