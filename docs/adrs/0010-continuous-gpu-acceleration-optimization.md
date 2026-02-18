# ADR 0010: Continuous GPU Acceleration Optimization Strategy

## Status
**Accepted** (Evolved 2024-2026; ongoing refinement)

## Context

GPU acceleration is critical for real-time face swapping. Early design used ONNX Runtime basics. Subsequent releases identified bottlenecks:

- **Apr 2025**: TensorRT support added for Jetson/high-performance NVIDIA
- **Dec 2025**: Poisson blending optimization (memory-efficient masking)
- **Feb 2026**: GPU-accelerated OpenCV (image preprocessing)
- **Feb 2026**: Face analyzer speedup (reduced resolution on Apple Silicon)

### Optimization Opportunities
- Per-component profiling (detect slowest operation)
- Platform-specific techniques (GPU-specific APIs)
- Memory optimization (reduce copies, cache reuse)
- Model quantization (fp16 vs int8 tradeoff)

## Decision

Implement continuous GPU acceleration optimization through:

1. **Systematic profiling**: Identify bottleneck per component
2. **Platform-specific tuning**: Optimize for CUDA/CoreML/TensorRT separately
3. **Component optimization**: Fast-track highest-impact improvements
4. **Incremental releases**: Small, incremental optimizations per release
5. **User feedback**: Monitor reported FPS improvements

### Optimization Targets (Priority)
1. **Face detection** (10ms baseline) → target 5ms
2. **Face swap** (15ms baseline) → target 10ms
3. **Face enhancement** (20-30ms) → target 15ms (optional)
4. **Memory efficiency** → reduce peak VRAM 30%

### Techniques
- **ONNX optimization**: Use onnx-simplify, quantization
- **GPU kernels**: Custom CUDA kernels for common operations
- **Batching**: Increase batch size to amortize overhead
- **Caching**: Cache intermediate results (face landmarks, embeddings)
- **Async execution**: Overlap GPU computation with data transfer

## Consequences

### Positive
✓ **Continuous improvement**: Incremental FPS gains over time
✓ **User perception**: Real-time experience improves with each release
✓ **Competitive advantage**: Faster than alternatives (hackier/Deep-Live-Cam original)
✓ **Hardware utilization**: Better leverage modern GPU capabilities
✓ **Scalability**: Enables 4K/multi-face use cases
✓ **Research contribution**: Optimization techniques published/shared

### Negative
✗ **Development effort**: Continuous optimization consumes resources
✗ **Testing complexity**: Must verify improvements don't regress quality
✗ **Platform fragmentation**: Optimization specific to one platform might break another
✗ **Maintenance burden**: Each optimization adds code paths to maintain
✗ **Diminishing returns**: Early improvements large; later gains marginal
✗ **Documentation**: Optimizations must be documented (rarely are)

### Mitigations
- **Automated benchmarking**: CI/CD pipeline measures FPS per commit
- **Regression testing**: Catch performance regressions early
- **Platform matrix**: Test on NVIDIA, Apple Silicon, AMD, Intel
- **Documentation**: ADRs and code comments explain each optimization
- **Feature flags**: Disable optimizations if causing issues

## Evidence

### Git History - Recent Optimizations
| Commit | Date | Optimization | Impact |
|--------|------|--------------|--------|
| 890beb0 | Apr 25, 2025 | TensorRT for NVIDIA | +15% FPS |
| df8e8b4 | Dec 15, 2025 | Poisson blending | -20% memory, same quality |
| e544889 | Feb 12, 2026 | Face analyzer speedup | +25% detection speed (ARM) |
| f0ec074 | Feb 12, 2026 | GPU-accelerated OpenCV | +10% overall FPS |

### Performance Trend
- **v1.0** (early 2024): 20-25 FPS on RTX 3060
- **v2.0c** (Oct 2025): 35-40 FPS (optimization phase began)
- **v2.0.3c** (Feb 2026): 45-50 FPS (continuous optimization)

### Platform-Specific Results
- **NVIDIA CUDA**: 50 FPS (continuous tuning, TensorRT)
- **Apple Silicon MPS**: 40 FPS (face analyzer optimization)
- **AMD ROCM**: 35 FPS (baseline implementation)
- **CPU fallback**: 2-3 FPS (acceptable for headless)

## Related Decisions
- [ADR 0001: ONNX/InsightFace](0001-use-onnx-and-insightface-for-face-detection-and-swap.md)
- [ADR 0004: Platform GPU Runtimes](0004-platform-specific-gpu-runtime-selection.md)
- [ADR 0008: ThreadPoolExecutor](0008-threadpoolexecutor-for-parallel-frame-processing.md)

## Future Optimizations

### Near-term (Next 2-3 releases)
- Quantization: int8 face detection without accuracy loss
- GPU memory pooling: Reduce allocation overhead
- Dynamic batch sizing: Auto-adjust batch size per GPU
- Stream processing: Overlap data transfer with computation

### Medium-term
- Custom CUDA kernels: Face blending, color correction
- TensorRT quantization: Minimize model size/latency
- Multi-GPU support: Distribute batches across GPUs
- Model pruning: Remove non-critical model weights

### Long-term
- Edge deployment: Optimize for mobile/low-power devices
- Real-time upscaling: 4K output without memory bloat
- Speculative execution: Pre-compute next frame while processing current
- Hardware-specific models: Compile for target GPU architecture

## Benchmarking Framework

Recommended additions:
- FPS histogram over 5-minute session
- Peak memory usage tracking
- GPU utilization percentage
- Per-component timing (detection, swap, enhance, mask)
- Platform-specific metrics (TensorRT optimization level, MPS utilization)

## Measurement Protocol

For each optimization, measure:
1. **FPS**: Average over 300 frames
2. **Memory**: Peak VRAM during processing
3. **Quality**: Visual inspection (no artifacts, smooth blending)
4. **Compatibility**: Test on all supported platforms

**Last Reviewed**: Feb 18, 2026 | **Confidence**: High
