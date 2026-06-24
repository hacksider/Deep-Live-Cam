# Future evolution: Live streaming without OpenCV in the desktop app

## Current Live implementation

The current desktop Live workflow uses OpenCV in `windows_app/app.py`:

```text
local webcam -> cv2 capture -> JPEG frames -> WebSocket -> Colab -> processed frames back
```

`cv2` is used for:

- opening the local camera with `cv2.VideoCapture(...)`;
- reading webcam frames;
- JPEG encoding frames before sending them over WebSocket;
- decoding processed frames returned by Colab;
- color conversion before optional virtual-camera output.

The current Live stack therefore pulls in:

- `opencv-python` / `cv2`;
- `numpy`;
- `pyvirtualcam` for optional virtual camera output.

This is easy to implement but makes standalone executables much larger. The first onefile build was about 107 MB, while a Lite build that excludes `cv2`, `numpy`, and `pyvirtualcam` was about 56 MB.

## Current packaging decision

Short term, keep two build flavors:

- **Full build**: includes Live webcam support through OpenCV.
- **Lite build**: excludes `cv2`, `numpy`, and `pyvirtualcam`; supports remote batch/photo/video controller flows but does not bundle Live webcam relay.

Commands:

```powershell
# Full onefile build
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_remote_app.ps1 -OneFile

# Lite onefile build
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_remote_app.ps1 -OneFile -Lite
```

## Alternative architectures

### 1. Browser-based capture

Use browser webcam APIs instead of desktop OpenCV capture:

```text
browser getUserMedia -> WebSocket/WebRTC -> Colab -> processed stream/result
```

Pros:

- No `cv2` dependency in the desktop executable.
- Browser handles camera permissions and capture.
- Better cross-platform story for Windows, macOS, Linux, and possibly mobile.

Cons:

- Requires web UI/frontend work.
- Colab runtime needs matching WebSocket or WebRTC handling.
- Virtual-camera output remains a separate desktop concern.

### 2. WebRTC direct streaming

Use real WebRTC for live media transport:

```text
browser or desktop WebRTC client -> signaling -> Colab WebRTC receiver/processor
```

Pros:

- Designed for live low-latency media.
- Can avoid OpenCV on the desktop if capture is browser-based.
- Better fit than JPEG-over-WebSocket for continuous video.

Cons:

- More complex signaling and media pipeline.
- Colab networking, Tailscale, and NAT behavior need careful design.
- Python WebRTC stacks such as `aiortc` may introduce different heavy dependencies.

### 3. FFmpeg-based local capture

Use FFmpeg to capture and stream from the local camera:

```text
ffmpeg dshow/avfoundation/v4l2 -> encoded stream -> Colab
```

Pros:

- Avoids `opencv-python` in the Python environment.
- FFmpeg is strong at device capture and encoding.
- Can offer efficient compressed streaming.

Cons:

- Requires FFmpeg installation or bundling.
- Device discovery and UI selection are platform-specific.
- Bundled FFmpeg may still be sizeable.

### 4. Keep Live outside the desktop EXE

Treat Live as a separate workflow:

- Lite desktop app controls batch/photo/video jobs only.
- Full desktop app includes current OpenCV Live mode.
- Browser/Colab notebook Live mode evolves independently.

Pros:

- Keeps the default release smaller.
- Avoids blocking batch/photo/video app releases on live-streaming complexity.
- Lets Live mode become a more focused future project.

Cons:

- Users need to understand the difference between Lite and Full builds.
- Live remains less integrated until a new architecture lands.

## Recommended direction

For near-term releases:

1. Ship **Lite** as the default desktop controller if photo/video batch control is the primary use case.
2. Offer **Full** for users who need local Live webcam relay.
3. Document clearly that Lite excludes Live webcam dependencies.

For future work:

1. Prototype browser-based capture first because it removes local camera dependencies from the desktop app and improves cross-platform reach.
2. Evaluate WebRTC only after the browser capture proof of concept works over the target Colab/Tailscale topology.
3. Consider FFmpeg capture if browser/WebRTC proves too complex or if local native streaming remains required.

## Open design questions

- Should the desktop app hide/disable the Live tab automatically in Lite builds?
- Should Lite and Full builds have different names, icons, or About text?
- Should browser-based Live run from the Colab notebook, a local static page, or inside the desktop app via an embedded web view?
- Is virtual camera output required for the first public Live release, or can processed preview/download be enough?
- Should WebRTC be routed over Tailscale directly, through Colab, or through a separate lightweight relay?
