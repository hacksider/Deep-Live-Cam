# GPU Acceleration Patterns

Derived from git history: multiple optimization commits across CUDA, CoreML, MPS, TensorRT
providers (2024-2026).

## Provider Selection

- Default to platform-appropriate provider — use `coreml` on macOS ARM, `cuda` on NVIDIA, `cpu`
  as fallback
- Never hardcode a single provider; read from `modules.globals.execution_providers`
- Use `--execution-provider` CLI flag or justfile recipes (`just start`, `just start-cpu`) — do
  not set provider directly in module code

## Adding Platform-Specific Code

When optimizing for a specific GPU:

1. Gate code behind provider check: `if 'cuda' in providers` or `if 'coreml' in providers`
2. Keep CPU fallback path functional — never assume GPU is available
3. Test on at least two platforms before merging (evidence: multiple Apple Silicon regressions
   fixed after NVIDIA-only testing)

## ONNX Runtime Dependencies

Platform-specific ONNX Runtime variants belong in `pyproject.toml` with `markers`:

```toml
onnxruntime-silicon = { version = "...", markers = "platform_machine == 'arm64' and sys_platform == 'darwin'" }
onnxruntime-gpu     = { version = "...", markers = "platform_machine != 'arm64' or sys_platform != 'darwin'" }
```

Do NOT install multiple ONNX Runtime variants in the same environment — they conflict.

## Optimization Strategy

Apply optimizations incrementally, per component:

1. Profile first — identify the actual bottleneck (face detection, swap, enhancement, I/O)
2. Optimize one component per PR; measure FPS before and after
3. Accepted benchmarking protocol: average FPS over 300 frames, report peak VRAM
4. GPU-accelerated OpenCV (`cv2.cuda`) is available for image ops — prefer it over CPU NumPy
   when the GPU provider is active

## Face Analyzer Performance

- Reduce `det_size` to `(160, 160)` on Apple Silicon in live mode for ~4x fewer detection FLOPs;
  restore `(320, 320)` for image/video processing where accuracy matters more than latency
- To change `det_size` at runtime, **recreate the `FaceAnalysis` instance** — `prepare()` is
  silently ignored after the first call (InsightFace singleton behavior)
- Cache detection results with time-based invalidation (≈0.033 s = 30 FPS interval)
- Use `DETECTION_INTERVAL` constant — do not inline magic numbers for frame-skip logic

## ThreadPoolExecutor Tuning

- `execution_threads` drives worker count; expose via CLI, never hardcode
- Batch size must fit within available VRAM — auto-calculate, do not use fixed batch sizes
- Re-use the executor across batches; do not create a new one per video segment
