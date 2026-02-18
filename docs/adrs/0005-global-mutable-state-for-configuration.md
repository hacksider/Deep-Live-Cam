# ADR 0005: Global Mutable State for Configuration Management

## Status
**Accepted** (Initial design, Sep 2023; maintained through v2.0.3c despite tradeoffs)

## Context

Deep-Live-Cam needs to share runtime configuration across multiple modules:
- Processing flags (many_faces, mouth_mask, nsfw_filter)
- File paths (source, target, output)
- GPU settings (execution_providers, execution_threads, max_memory)
- Face mappings (source_target_map, simple_map embeddings)
- UI state (live_mirror, webcam_preview_running, processor flags)

Early design centralized all state in `modules/globals.py` as module-level variables, avoiding complex dependency injection or message passing.

## Decision

Use **module-level mutable state** in `modules/globals.py`:

```python
# Configuration (set by core.py argument parsing)
source_path: str = None
target_path: str = None
output_path: str = None
execution_providers: list = ['cpu']
execution_threads: int = 4

# Processing flags (toggleable by UI/CLI)
many_faces: bool = False
mouth_mask: bool = False
nsfw_filter: bool = False
map_faces: bool = False

# Face data (populated during analysis)
source_target_map: list = []
simple_map: dict = {}

# UI state
fp_ui: dict = {}  # Frame processor toggles
live_mirror: bool = False
webcam_preview_running: bool = False
```

### Initialization Flow
1. core.py parses CLI args → populates globals
2. UI reads globals → displays current state
3. User interacts with UI → updates globals
4. Processors read globals → execute with current settings
5. Loop continues (responsive to user toggles)

## Consequences

### Positive
✓ **Simple**: Easy to understand and debug; no complex dependency graph
✓ **Fast**: Direct variable access, no getter/setter indirection
✓ **Convenient**: No dependency injection plumbing
✓ **UI responsiveness**: Real-time config updates visible immediately
✓ **Backwards compatible**: All versions use this pattern

### Negative
✗ **Testability**: Unit tests must mock/reset globals; test isolation hard
✗ **Thread safety**: Multiple threads reading/writing without locks (potential race conditions)
✗ **Implicit dependencies**: Modules depend on globals without explicit imports
✗ **Refactoring**: Hard to rename/move globals (breaks all dependent code)
✗ **State management**: No history/undo; no clear initialization order
✗ **Memory leaks**: Persistent references in globals prevent garbage collection

### Mitigations
- **Test isolation**: Reset globals in test fixtures (setup/teardown)
- **Thread safety**: Face detection cache uses locks; frame processing reads config once
- **Documentation**: CLAUDE.md and globals.py clearly document all state variables
- **Type hints**: modules/typing.py defines types (Face, Frame) used by globals

## Evidence

### Codbase Usage
- 508 commits without architectural change to globals pattern
- Core.py sets globals from CLI args (line ~100-120)
- Every processor imports globals to read config
- UI callbacks update globals directly

### Functional Correctness
- Live mode responsiveness proves simple state management works
- No major bugs traced to global state race conditions (threading sufficiently careful)
- Face mapping (source_target_map) correctly persists across frames

### Known Limitations
- No formal configuration schema (implicit from code)
- No config serialization/persistence (state lost on restart)
- No A/B testing framework (hard to test config combinations)

## Related Decisions
- [ADR 0002: Plugin Architecture](0002-plugin-architecture-for-frame-processors.md) (plugins access globals)
- [ADR 0003: CustomTkinter GUI](0003-customtkinter-for-cross-platform-gui.md) (UI updates globals)
- [ADR 0008: ThreadPoolExecutor](0008-threadpoolexecutor-for-parallel-frame-processing.md) (thread safety concern)

## Future Improvements
- **Dependency Injection**: Replace globals with explicit parameter passing
- **Config Classes**: Structured configuration objects (Dataclass, Pydantic)
- **Event system**: Pub/sub for config changes (vs direct mutation)
- **Configuration persistence**: Save/load settings from YAML/JSON
- **Thread-safe wrappers**: Lock guards around critical globals

## Migration Path

If refactoring away from globals:
1. Phase 1: Add config classes alongside globals
2. Phase 2: Gradually migrate modules to accept config via parameters
3. Phase 3: Remove globals, fully transition to dependency injection

This could be driven by test-driven development of new features.

**Last Reviewed**: Feb 18, 2026 | **Confidence**: Medium (works but not ideal pattern)
