# Apple Silicon + Cross-Platform Performance

End-to-end measurements from commit `f65aeae` on a MacBook Pro M3 Max
against `hacksider/Deep-Live-Cam` upstream `main@64d3f06`, same hardware,
same camera, same source/target faces.

| Mode | Upstream `main` | This fork | Delta |
|---|---|---|---|
| Face swap only | **<5 FPS** | **>20 FPS** | ~5× |
| Face swap + GFPGAN enhancer | **<2 FPS** | **>10 FPS** | ~5× |
| Camera resolution | 640×480 default | **960×540 MJPEG** | wider FoV |
| Camera frame rate | 15–30 fps (backend default) | **60 fps negotiated + measured** | up to 2–4× |

The gap is cumulative — no single change accounts for it. Each section
below describes one contributor, in rough order of impact on the
live-video pipeline.

## 1. Camera capture negotiation — `modules/video_capture.py`

Upstream calls `cv2.VideoCapture(device_index)` with no hints and
accepts whatever the camera defaults to. On most webcams that means
640×480 YUV at 15–30 fps, and `CAP_PROP_FPS` lies on DirectShow.

This fork:

- Requests `MJPG` fourcc to bypass USB bandwidth limits on uncompressed
  YUV at high resolutions.
- Requests 960×540 @ 60 fps up front.
- Reads back the camera's actual resolution via `CAP_PROP_FRAME_WIDTH/HEIGHT`.
- **Empirically measures FPS** by timing 30 frames after 10 warmup
  reads (`_measure_fps`) instead of trusting `CAP_PROP_FPS`. Costs
  ~0.5–1 s at startup; gives ground-truth numbers that downstream
  code (detection cadence, enhancer throttle) tunes against.
- Windows path tries `CAP_MSMF` before `CAP_DSHOW` (DirectShow often
  caps at 30 fps even when the camera supports 60).

This single change is why the resolution / FoV / FPS look different between
upstream and the fork before any ML work starts.

## 2. CoreML graph rewrites — `modules/onnx_optimize.py`

CoreML EP silently falls back to CPU for ops it doesn't support,
creating partition boundaries with CPU↔ANE round-trips between each.
Three pre-load rewrites eliminate the fallbacks:

### 2a. `Pad(mode=reflect)` → `Slice + Concat` (inswapper_128)

Verified on this machine:

| Config | Partitions | inswapper latency |
|---|---|---|
| Original ONNX, ORT 1.24.4 | **14** | 55.3 ms |
| Rewritten, ORT 1.24.4 (this pass) | **1** | 27.4 ms |
| Original ONNX, ORT 1.26 (main) | **1** | 27.2 ms |

The third row uses ORT built from `main` at `fb13eb3edd` which contains
[microsoft/onnxruntime#28073](https://github.com/microsoft/onnxruntime/pull/28073)
— native MIL `pad(mode="reflect")` under `MLProgram`. Once ORT ≥ 1.26 is the
floor, this pass can be deleted. See `REVIEW_TODOS.md` for the cleanup note.

### 2b. `Shape → Gather` folded to constants (det_10g)

Dynamic shape chains for FPN upsample target sizes forced parts of the
face detector onto CPU. When the input shape is known at load time we
run ONNX shape inference and replace the chains with `int64` constants.

Measured: **21 ms → 4 ms** on the detection model.

### 2c. `Split(axis=1, 2 outputs)` → `Slice` pairs (GFPGAN)

CoreML EP doesn't support `Split`. GFPGAN's SFT modulation layers use
channel-wise splits everywhere, forcing partition boundaries. Rewriting
each 2-way Split as two Slices eliminates the fallbacks.

Measured: **155 ms → 89 ms** on GFPGAN. This is the single largest
contributor to the "GFPGAN enabled" row in the headline table.

All three rewrites cache to disk with a `_coreml` suffix so the cost is
paid once per model per machine.

## 3. Pipeline overlap — `modules/processors/frame/core.py`, `face_swapper.py`

Face detection and face swap both use the Neural Engine. Running them
serially leaves the ANE idle during the detection half. The fork:

- Overlaps detection N+1 with swap N via a thread pool. Adds one frame
  of latency; doubles ANE utilization.
- Skips `landmark_2d_106` when only `face_swapper` is active (landmarks
  are unused unless mouth-mask or interpolation is on).
- Parallelizes landmark + recognition post-detection when both are
  needed.
- Routes `det_10g` to GPU (Metal) so ANE stays free for the swap model.

The one-frame detection lag is a known trade-off — acceptable for
video frame rates where the face barely moves frame-to-frame. Flagged
as a quality risk on fast motion / scene cuts in `REVIEW_TODOS.md`.

## 4. GFPGAN-specific — `modules/processors/frame/face_enhancer.py`, `_onnx_enhancer.py`

- Temporal cache: in live mode, run GFPGAN inference every Nth frame
  and reuse the enhancement; interpolate the affine paste-back each
  frame. Essentially free interpolation between inferences.
- Pre-computed FFHQ 512 landmark template (avoids per-frame matrix solve).
- Session created once under `create_onnx_session` with the same
  `ModelFormat=MLProgram + MLComputeUnits=ALL` config as the swap
  model — previously GFPGAN fell back to CPU-only.

## 5. Paste-back optimization — `modules/processors/frame/face_swapper.py`

`_fast_paste_back` replaces insightface's `paste_back` which operates
on the full frame:

- Computes face bbox from the affine matrix directly (no warp-and-scan
  of a white mask).
- Runs erosion, blur, and blend on the face bbox only.
- GPU path (CUDA) keeps mask arithmetic on GPU end-to-end
  (`torch.nn.functional.max_pool2d` / `avg_pool2d` for erode + blur).
- Writes in-place into `target_img` to avoid a full-frame copy.

## 6. Platform routing — `modules/platform_info.py`

Single source of truth for OS / accelerator detection. Consumed by
capture backend selection, provider config for `face_swapper` and
`face_enhancer`, and a one-line startup banner confirming which code
path the app took.

## 7. Windows CUDA path (not exercised in M3 Max numbers)

Not contributing to the Apple Silicon table but included in the same
commit:

- CUDA graphs via `enable_cuda_graph=1` + `io_binding` with
  pre-allocated `OrtValue` buffers. Replays the recorded kernel launch
  sequence each frame with near-zero CPU overhead. Requires static
  input shape — inswapper is always `1×3×128×128 + 1×512`.
- FP16 model auto-selected on GPUs with Tensor Cores (Turing+),
  falls back to FP32 on older GPUs where FP16 can produce NaN.
- DLL discovery fix for NVIDIA CUDA/cuDNN from pip-installed `torch`
  and `nvidia-*` wheels.

Enables 1080p @ 60 FPS on NVIDIA hardware (measured separately, not in
the table above).

## Reproducing the measurements

Cold run (kill all deep-live-cam processes, no active CoreML cache):

```bash
cd /path/to/Deep-Live-Cam
.venv/bin/python run.py
# Look for:
#   [platform] ...                 -> confirms accelerator selection
#   [VideoCapturer] 960x540 @ 60fps (reported=...)
#   Partitions: 1                  -> from CoreML EP verbose logs
```

For the CoreML partition / inswapper latency numbers specifically, see
the standalone test at `/Users/max/Development/onnxruntime_test/test_pad_reflect.py`
— runs the model with and without the graph rewrite on any installed ORT.

## Future cleanup

When ORT floor ≥ 1.26.0 lands (microsoft/onnxruntime#28073):

- Delete `_decompose_reflect_pad` in `modules/onnx_optimize.py`.
- Drop the `TODO(ort>=1.26)` markers.
- Update `requirements.txt`.

Does not change runtime performance — native MIL reflect matches the
Slice+Concat rewrite to within noise (27.2 vs 27.4 ms in the table
above). Purely a code-deletion cleanup.
