# Live InsightFace Pack Selection Plan

Date: 2026-06-25
Branch: live-webcam-stability

## Scope
- Add Live tab settings for InsightFace model pack selection (`buffalo_l`, `buffalo_m`, `buffalo_s`) and swapper precision (`fp32`, `fp16`).
- Send the selected pack to the Colab `/ws/live` session config.
- Load the selected pack and selected swapper precision in the Colab live engine by resetting cached models when settings change.
- Keep live source face embedding cached once per live engine/session.

## Decisions
- Provision `inswapper_128_fp16.onnx` from the FaceFusion model-pack mirror and keep model binaries ignored by git.
- Default remains `buffalo_l` because InsightFace documents it as the safest embedding source for `inswapper_128`.
- `buffalo_m` and `buffalo_s` are exposed as experimental speed options.
- The source face is detected once during `ModernEngine` initialization and reused via `engine.default_source` for all live frames.

## Deferred validation
- User-owned live Colab/webcam validation for pack download latency, speed, and swap quality.

## 2026-06-25 WebP live frame codec update

### Scope
- Add Live tab controls for send codec and return codec (`jpeg`, `webp`) while keeping JPEG as the safe default.
- Send selected codecs plus frame quality in the `/ws/live` config.
- Let the server auto-decode incoming JPEG/WebP bytes with OpenCV `imdecode` and encode returned frames with JPEG or WebP.
- Report `frame_codec`, `output_codec`, `encoded_codec`, `frame_quality`, `in_kb`, and `out_kb` in live diagnostics/perf logs.

### Decision
- Use one existing quality spinbox as generic frame quality for JPEG and WebP to keep the UI compact and preserve existing settings.
- Fallback to JPEG if WebP encode fails on either side.

### Deferred validation
- User-owned Colab webcam validation for WebP availability, preview compatibility, latency, and visual quality.

## 2026-06-25 Live preview size control

### Scope
- Add a small Live preview-panel control for `fit`, `1x`, `1.5x`, and `2x`.
- Keep `fit` as the default preview behavior.
- Cap requested pixel scales to the preview panel; if the scaled frame is larger than the panel, render fit instead.

### Deferred validation
- User-owned manual GUI validation for preview sizing behavior during live webcam playback.
