# Face Processing Patterns

Derived from recurring fix commits: None embeddings crash (PR #980), source_target_map typo
(#1565), face mapper bugs in live mode (#572, #598), and face enhancer device errors (#829).

## Face Detection

- Always guard against `None` face results before accessing embeddings or bounding boxes
  (source: "Fix face swapping crash due to None face embeddings")
- Log a clear message when no faces are found — do not silently skip frames
- Return the unmodified input frame when no face is detected; never return `None`

## InsightFace Singleton Behavior

- `FaceAnalysis.prepare()` sets `det_size` **only on the first call** — subsequent calls are
  silently ignored with `"warning: det_size is already set in detection model, ignore"`
- To change `det_size` at runtime (e.g., switching between live mode and video mode), you must
  **recreate the `FaceAnalysis` instance entirely** — calling `prepare()` again is a no-op
- Protect the singleton with a lock when recreating it from multiple threads

## Embedding Validation

- Validate face embedding shape and dtype before passing to ONNX swap model
- A mismatched embedding silently produces garbage output — fail fast with a descriptive error

## Face Mapping (`source_target_map`)

- Variable name is `source_target_map` — not `souce_target_map` (recurring typo, two separate
  fixes in git history)
- Keep map entries as `{source_face, target_face}` dicts; avoid positional indexing
- When live-mode faces exceed available map entries, fall back to the first map entry rather
  than crashing (evidence: PR #572)

## Frame Processor Pipeline

- Processors run sequentially: `face_swapper` → `face_enhancer` → `face_masking`
- Each processor receives the output of the previous one as its input frame
- A processor must return a valid frame even if it performs no operation (pass-through)
- Implement all five interface methods: `pre_check`, `pre_start`, `process_frame`,
  `process_image`, `process_video`

## Face Enhancer (GFPGAN)

- Load the model conditionally based on the active device — do not load on CPU if GPU is
  available and vice versa (evidence: "Make Face Enhancer Model device Conditional")
- Wrap enhancement in a semaphore to prevent concurrent VRAM exhaustion on multi-face frames
- Enhancement is optional; the pipeline must work correctly with it disabled

## Mouth Masking

- Use face landmarks for mouth region coordinates; do not use fixed pixel offsets
- Apply Poisson blending (`cv2.seamlessClone`) at the mask boundary for smooth transitions
  (added in v2.0.2c; do not revert to hard alpha compositing)
- Test with both `--mouth-mask` enabled and disabled before committing mouth-mask changes

## Model Download and Caching

- Use `utilities.conditional_download()` for all model files — never download directly in
  processor code
- Support fallback sources: if the primary URL fails, try the secondary URL before surfacing
  an error to the user (evidence: "Creating a fallback and switching of models", Aug 2025)
- Store models in the `models/` directory; do not use temp directories for model files
