# ADR 0002: Plugin Architecture for Frame Processors

## Status
**Accepted** (Design pattern evident in codebase since 2024)

## Context

Deep-Live-Cam processes frames through multiple optional transformations:
1. Face swapping (core)
2. Face enhancement with GFPGAN (optional)
3. Mouth masking (optional)

Early design used serial composition (swap → enhance → mask). The codebase evolved to support dynamic processor loading and enabling/disabling via UI and CLI.

### Why This Matters

- **Feature Independence**: Each processor can be developed, tested, and optimized independently
- **Runtime Composition**: Users toggle processors on/off without code changes
- **Extensibility**: New processors (gaze correction, expression transfer) can be added without modifying core pipeline
- **Testing**: Mock processors in unit tests without full dependency chain

## Decision

Implement a plugin architecture where frame processors are dynamically loaded from `modules/processors/frame/`:

```
modules/processors/frame/
├── core.py                 # Plugin loader + ThreadPoolExecutor
├── face_swapper.py         # Core face swap (required)
├── face_enhancer.py        # Optional enhancement
└── face_masking.py         # Optional mouth masking
```

### Interface Contract

Each processor implements:
```python
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',       # Validate requirements (download models)
    'pre_start',       # Pre-execution setup
    'process_frame',   # Transform single frame
    'process_image',   # Transform still image
    'process_video'    # Process video batch
]
```

### Composition Strategy

- **Dynamic loading**: `importlib` loads processor modules based on CLI/UI selections
- **Sequential pipeline**: `core.py:multi_process_frame()` chains outputs (face_swapper → face_enhancer → face_masking)
- **ThreadPoolExecutor**: Parallel frame batching for throughput
- **State management**: `modules.globals` stores enabled processors and settings

## Consequences

### Positive
✓ **Independent Development**: Each processor has isolated code, tests, dependencies
✓ **Runtime Flexibility**: Enable/disable features without recompiling
✓ **Extensibility**: New processors added by creating module with interface methods
✓ **Testability**: Mock/stub processors in unit tests
✓ **Maintenance**: Bug in face_enhancer doesn't affect face_swapper
✓ **Community**: Contributors can create custom processors

### Negative
✗ **Complexity**: Plugin discovery and loading adds abstraction layer
✗ **Debugging**: Harder to trace errors across processor boundaries
✗ **Interface Brittleness**: Adding/removing interface methods requires all processors to update
✗ **Documentation**: Must maintain clear contract for processor authors
✗ **Threading**: ThreadPoolExecutor batching adds complexity; memory usage scales with batch size

### Mitigations
- Clear interface documentation in core.py
- Example processor templates for new contributions
- Batch size configuration to control memory
- Error handling in loader (graceful processor skip on import failure)

## Evidence

### Git History
- Design evident in: modules/processors/frame/core.py (used since 2024)
- UI processor toggles: ui.py frame processor checkboxes (multiple references)
- CLI args: core.py --frame-processor flag (2024+)
- Version 2.0c+ maintains architecture without major changes

### Architecture Benefits
- Face enhancer can be toggled on 1080p frames (on) or full HD (off) based on GPU
- Mouth masking independently improves realism without touching swap logic
- Custom processors (not yet public) demonstrated in internal branches

### Performance
- Sequential composition ensures output quality (later processors enhance earlier results)
- ThreadPoolExecutor with configurable workers balances memory vs throughput
- Batch processing reduces model load overhead

## Related Decisions
- [ADR 0001: ONNX/InsightFace](0001-use-onnx-and-insightface-for-face-detection-and-swap.md) (core processor uses this)
- [ADR 0005: Global Mutable State](0005-global-mutable-state-for-configuration.md) (processors access globals)
- [ADR 0008: ThreadPoolExecutor for Parallel Processing](0008-threadpoolexecutor-for-parallel-frame-processing.md)

## Future Improvements
- Async/await processor interface (replace ThreadPoolExecutor)
- Processor dependency injection (replace globals)
- Processor version negotiation (support multiple interface versions)
- Custom processor marketplace/registry

**Last Reviewed**: Feb 18, 2026 | **Confidence**: High
