# Live webcam performance correlation notes - 2026-06-25

Source: pasted live logs from API version `live-fast-detect-v6`. CSV data is saved beside this file as `live-webcam-perf-correlation-2026-06-25.csv`.

## Extracted sessions

| Session | n | Capture FPS | Pipeline | Detect N | Cache source | Avg server FPS | Avg wait ms | Avg process ms | Avg detect ms | Avg swap ms | Avg detect reuse % |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| A_30fps_pipe90_detect1_cache_on | 5 | 30 | 90 | 1 | True | 12.16 | 47.1 | 37.1 | 6.7 | 30.4 | 0.0 |
| B_30fps_pipe30_detect1_cache_on | 8 | 30 | 30 | 1 | True | 9.47 | 68.3 | 35.6 | 6.3 | 29.2 | 0.0 |
| C_15fps_pipe15_detect1_cache_on | 3 | 15 | 15 | 1 | True | 9.59 | 66.2 | 36.0 | 6.5 | 29.5 | 0.0 |
| D_15fps_pipe75_detect1_cache_on | 6 | 15 | 75 | 1 | True | 12.59 | 45.2 | 36.2 | 6.6 | 29.5 | 0.0 |
| E_15fps_pipe75_detect3_cache_off | 4 | 15 | 75 | 3 | False | 7.71 | 3.1 | 124.2 | 37.7 | 86.5 | 66.2 |
| F_15fps_pipe75_detect3_cache_on | 7 | 15 | 75 | 3 | True | 7.33 | 97.3 | 38.8 | 2.5 | 36.2 | 66.6 |
| G_15fps_pipe75_detect1_cache_on | 6 | 15 | 75 | 1 | True | 11.36 | 50.5 | 36.5 | 6.6 | 29.9 | 0.0 |

## Correlations across all perf samples

Pearson correlations are computed across the extracted `live_perf` rows. This is a small, manually extracted dataset, so treat correlations as directional, not definitive.

| Pair | r | Read |
|---|---:|---|
| wait_ms vs server_fps | -0.52 | higher wait generally means lower server FPS |
| wait_ms vs process_ms | -0.61 | wait is weakly tied to processing because wait also includes frame arrival/backpressure timing |
| wait_ms vs detect_ms | -0.71 | detect cost rises when `detect_every_n=1` or source cache is off |
| wait_ms vs swap_ms | -0.54 | swap cost dominates process cost, especially when source cache is off |
| wait_ms vs detect_reuse_pct | 0.12 | detect reuse rises with detect_every_n=3; source cache off is an outlier |
| wait_ms vs pipeline_frames | -0.22 | large pipelines correlate with mixed latency outcomes; they do not guarantee FPS |
| wait_ms vs capture_fps | 0.06 | capture FPS alone is not predictive because processing cannot sustain 30 FPS |

## Main findings

- `cache_source_face=false` is the largest bad outlier: avg process ~124 ms and avg swap ~86 ms despite `fp16`; this is because source face analysis is repeated every frame and is included inside the swap timing block.
- `detect_every_n=1` keeps `detect_reuse_pct=0` and costs ~6-8 ms per frame. `detect_every_n=3` with cache on reduces avg detect to ~2-4 ms.
- `pipeline_frames=75/90` can produce occasional high `server_fps` bursts, but also high `wait_ms`; it likely buffers/backlogs rather than reducing latency.
- Lowering capture from 30 to 15 FPS did not automatically improve stability when pipeline stayed high; the best latency setting still needs pipeline near the actual target FPS and source cache on.
- `wait_ms` is mostly a pipeline/frame-arrival/backpressure signal, not swapper compute. When processing cannot keep up cleanly, the server alternates between waiting for frames and draining bursts.

## Recommended next live preset from these logs

```text
Capture FPS: 15 or 20
Pipeline frames: same as capture FPS, or at most 2x capture FPS
Detector size: 256
Detect every N: 3
Swapper precision: fp16
InsightFace pack: buffalo_l
Cache source face: ON
Opacity: 1.0 while testing
```

Avoid `cache_source_face=false` except for a short diagnostic; it makes the source analysis cost recur every frame.

