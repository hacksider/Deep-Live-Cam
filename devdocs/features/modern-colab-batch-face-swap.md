# Modern Colab Batch Face Swap

## Goal and scope

Create `projects/Deep-Live-Cam-Remote` from the modern `tmp/Deep-Live-Cam` baseline and add an all-Colab, path-based batch processor. The workflow processes videos already available under `/content` or mounted Drive paths; it does not use ZMQ, Tailscale, FastRTC, Gradio, or a Windows controller.

## Behavior changes

- Add a reusable Colab CLI and self-contained notebook for folder processing.
- Decode and resize with FFmpeg before inference, cap input at 30 FPS and 420 pixels wide by default, overlap decode/encode with bounded queues, preserve audio, and package outputs as a ZIP.
- Persist a settings-aware manifest so identical work is skipped while changed inputs, faces, models, or processing options are reprocessed.
- Expose the modern headless engine features: many-face processing, face mapping, mouth masking, opacity, sharpening, temporal interpolation, Poisson blending, color correction, and GFPGAN/GPEN enhancement.
- Provision and validate the InsightFace swapper model before processing so a missing model fails once with an actionable error rather than producing unchanged output frames.

## Examples

Before: the modern project processes one selected target through its desktop UI.

After:

```bash
python colab_batch.py process \
  --source-face /content/source.png \
  --input-dir /content/in \
  --output-dir /content/out \
  --zip-output /content/face-swapped.zip
```

Face mapping is prepared separately with `colab_batch.py scan`, then supplied with `--map-config`.

## Tasks

- [x] Lock all-Colab scope and new project name.
- [x] Copy the modern project without nested repository/runtime artifacts.
- [x] Implement batch CLI, manifest, FFmpeg pipeline, advanced processing options, mapping scan, and ZIP output.
- [x] Add self-contained markerized notebook and generated `.ipynb`.
- [x] Add documentation and tests.
- [x] Validate script syntax, tests, notebook round trip, and repository diff.

## Decisions

- The workflow is batch-only and runs entirely in Colab.
- Inputs and source images are Colab/Drive paths.
- Outputs are downloaded as a single ZIP.
- The notebook and readable `.py` source are maintained with `ipynb-roundtrip`.
- Only one batch runs at a time; individual video failures do not stop the remaining batch.

## Summary since the previous baseline

This feature adds a separate modern project variant rather than modifying `projects/Deep-Live-Cam`. It combines the newer local processing engine with the proven FFmpeg-pipe batch architecture and provides a Colab-native interface without remote networking.
