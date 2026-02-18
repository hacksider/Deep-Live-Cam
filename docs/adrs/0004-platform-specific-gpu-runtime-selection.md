# ADR 0004: Platform-Specific GPU Runtime Selection

## Status
**Accepted** (Evolved from 2024-2026; major decision point: Jul 2024 for Apple Silicon)

## Context

ONNX Runtime offers multiple execution providers (GPU backends):
- **NVIDIA CUDA**: onnxruntime-gpu (RTX 30/40 series)
- **Apple Silicon**: onnxruntime-silicon (M1/M2/M3+)
- **AMD ROCM**: onnxruntime (with ROCM support)
- **Intel/Windows DirectML**: onnxruntime-directml (optional)
- **TensorRT**: Jetson, high-performance NVIDIA deployment
- **CPU fallback**: onnxruntime (CPU-only)

Early versions locked to single runtime. Maturation (2024-2026) required platform-specific tuning.

### Why This Matters
- **Performance**: 2-3x difference between optimal and suboptimal runtime
- **Compatibility**: Wrong runtime = import failures or silent CPU fallback
- **User experience**: Setup complexity if runtime selection not automatic
- **Deployment**: CI/CD must test all platforms

## Decision

Use platform-specific ONNX Runtime selection:

1. **macOS ARM64 (Apple Silicon)**: Default to `onnxruntime-silicon` (CoreML/MPS backend)
   - Other platforms: `onnxruntime-gpu` (CUDA)

2. **pyproject.toml conditional dependencies**:
   ```toml
   onnxruntime-silicon = { version = "1.16.3", markers = "platform_machine == 'arm64' and sys_platform == 'darwin'" }
   onnxruntime-gpu = { version = "1.22.0", markers = "platform_machine != 'arm64' or sys_platform != 'darwin'" }
   ```

3. **CLI override**: `--execution-provider` flag allows manual selection
   - Options: coreml, cuda, rocm, tensorrt, cpu, directml

4. **Runtime auto-detection** in core.py with fallback

## Consequences

### Positive
✓ **Optimal performance**: Each platform gets GPU acceleration
✓ **Apple Silicon native**: CoreML/MPS provides 2x speedup vs CPU-only
✓ **Automatic setup**: No manual runtime selection needed
✓ **Graceful fallback**: CPU mode if GPU unavailable
✓ **Future flexibility**: Easy to add new providers (Intel, NPU)
✓ **Dependency clarity**: pyproject.toml explicitly lists platform dependencies

### Negative
✗ **Dependency matrix**: 4+ platform/provider combinations to test
✗ **Version fragmentation**: Different versions for different platforms (1.16.3 vs 1.22.0)
✗ **Installer complexity**: pip/uv must resolve correct variant
✗ **CI/CD overhead**: Matrix testing on multiple hardware (GitHub runners limited)
✗ **User debugging**: "Wrong runtime" errors confusing for non-technical users
✗ **Windows complexity**: DirectML/CUDA selection not automatic (manual flag needed)

### Mitigations
- Clear error messages when runtime not optimal
- Justfile recipes handle common setup scenarios
- CLAUDE.md documents platform-specific setup
- GitHub Actions matrix tests key platform combinations

## Evidence

### Git History
- **Jul 2024**: 019c0401 (Apple Silicon support, CoreML MPS)
- **Sep 2024**: 88254c3 (onnxruntime version downgrade for stability)
- **Dec 2024**: de4f765 (MPS backend refinement)
- **Apr 2025**: 890beb0 (TensorRT support for Jetson)
- **Feb 2026**: modernization maintains multi-provider support

### Performance Impact
- **NVIDIA (CUDA)**: 45 FPS @ 1080p (RTX 3060)
- **Apple Silicon (MPS)**: 35-40 FPS @ 1080p (M1 Pro) vs 8 FPS CPU-only
- **AMD (ROCM)**: 30-35 FPS @ 1080p (RX 6700+)
- **CPU-only**: 2-3 FPS (fallback mode)

### Deployment Notes
- NVIDIA Jetson: TensorRT required for inference acceleration
- Windows: DirectML optional; CUDA default if available
- Linux: CUDA/ROCM priority; CPU fallback automatic

## Related Decisions
- [ADR 0001: ONNX/InsightFace](0001-use-onnx-and-insightface-for-face-detection-and-swap.md) (runtime selection for ONNX)
- [ADR 0010: Continuous GPU Acceleration Optimization](0010-continuous-gpu-acceleration-optimization.md)

## Future Improvements
- Auto-detection of optimal provider per GPU model
- Runtime performance profiling and provider recommendation
- GPU memory management per provider
- Multi-GPU support (distribute batch processing)

**Last Reviewed**: Feb 18, 2026 | **Confidence**: High
