# Deep-Live-Cam: Real-Time Face Swap Platform

## Overview

Deep-Live-Cam (v2.0.3c) is an open-source research and educational platform for real-time face swapping and video deepfake creation. It enables users to swap faces in live webcam feeds, images, and videos using ONNX-based inference and InsightFace detection, with optional face enhancement via GFPGAN.

**Key Value Proposition**: Single-click real-time face swapping with professional-grade face enhancement and ethical safeguards, targeting both technical users (developers, ML engineers) and non-technical content creators.

## User Stories

### Primary Personas

**Persona 1: Content Creator (Non-Technical)**
- As a video producer, I want to swap my face into movie scenes in real-time so that I can create engaging content for social media.
- As a streamer, I want to replace my face with a character on live shows so that I can entertain my audience.
- As a meme creator, I want to quickly swap faces in images and videos so that I can produce viral content.

**Persona 2: ML Researcher / Developer (Technical)**
- As a machine learning engineer, I want to understand how face detection and ONNX inference work so that I can optimize models for production use.
- As a researcher, I want to benchmark GPU acceleration across platforms (CUDA, CoreML, TensorRT) so that I can identify performance bottlenecks.
- As a developer, I want to extend the face processor pipeline with custom filters so that I can experiment with new face manipulation techniques.

## Core Features

### 1. Real-Time Face Swapping
- **Description**: Replace detected faces in live webcam feed with a single source image
- **Scope**: Core feature (essential for MVP and ongoing development)
- **User Story**: "As a content creator, I want to swap my face in real-time with a live camera feed so that I can use the output for streaming or recording"
- **Technical Implementation**:
  - Input: Source image (single face)
  - Detection: InsightFace buffalo_l model extracts face embeddings and bounding boxes
  - Swap: ONNX inswapper_128_fp16 model applies face replacement
  - Output: Real-time preview + recorded video/stream
  - Performance: 30-60 FPS on GPUs (platform-dependent)
- **Status**: Production (active development, ongoing optimization)

### 2. Face Detection & Analysis
- **Description**: Detect and analyze faces in input streams using InsightFace
- **Scope**: Core feature (foundation for all other features)
- **Acceptance Criteria**:
  - Multi-face detection (>2 faces per frame)
  - Face embedding extraction for similarity matching
  - Adaptive quality (lower resolution on Apple Silicon for performance)
  - Caching to reduce redundant detection (~30 FPS frame skip rate)
- **Status**: Production with ongoing optimization

### 3. Face Enhancement (Optional)
- **Description**: Improve facial details and resolution post-swap using GFPGAN
- **Scope**: Enhancement feature (toggleable via UI and CLI)
- **User Story**: "As a content creator, I want to enhance the swapped face with better detail and lighting so that the output looks more realistic"
- **Acceptance Criteria**:
  - GFPGAN v1.4 model restores facial detail
  - Conditional device support (CUDA, CoreML on macOS ARM)
  - Configurable strength/blending
  - Performance: <15ms per face on GPU
- **Status**: Active (refined for platform-specific GPU support)

### 4. Mouth Masking (Optional)
- **Description**: Preserve the original mouth region to maintain natural speech lip-sync
- **Scope**: Enhancement feature (toggleable)
- **User Story**: "As a content creator, I want to keep my original mouth visible so that my lip movements look natural and authentic"
- **Acceptance Criteria**:
  - Mouth region detection from face landmarks
  - Selective blending of original mouth over swapped face
  - Poisson blending for smooth edges (added v2.0.2c)
  - Minimal performance impact (<5ms per face)
- **Status**: Active (recently improved with Poisson blending)

### 5. Face Mapping (Multi-Person Swap)
- **Description**: Map different source faces to specific people in a video/stream
- **Scope**: Enhancement feature (advanced use case)
- **User Story**: "As a content creator, I want to apply different faces to different people in a scene so that I can create complex multi-person swaps"
- **Acceptance Criteria**:
  - Drag-and-drop UI for source↔target face matching
  - Face clustering via K-means (centroid-based matching)
  - Embedding storage for live mode persistence
  - Support for 2+ source faces per scene
- **Status**: Active (production use)

### 6. Live Webcam Streaming
- **Description**: Real-time preview and streaming capability
- **Scope**: Core feature (essential for live content creation)
- **User Story**: "As a streamer, I want to see a live preview of the face swap before recording so that I can adjust settings in real-time"
- **Acceptance Criteria**:
  - 30-60 FPS preview on target GPU
  - Resizable preview window (drag-to-resize)
  - Mirror option for self-view correction
  - Camera device enumeration (cross-platform)
  - Stop on window close (prevent dangling processes)
- **Status**: Production

### 7. Batch Video Processing
- **Description**: Process saved video files with output to MP4
- **Scope**: Core feature (headless mode for batch workflows)
- **User Story**: "As a content creator, I want to process saved video files in batch mode without opening the GUI so that I can automate content production"
- **Acceptance Criteria**:
  - Headless CLI mode (`run.py -s source.jpg -t input.mp4 -o output.mp4`)
  - Frame extraction via ffmpeg
  - Parallel processing with ThreadPoolExecutor
  - Audio passthrough to output
  - Hardware-accelerated encoding (h264_nvenc, hevc_videotoolbox)
- **Status**: Production

### 8. GPU Acceleration
- **Description**: Platform-specific GPU optimization for maximum performance
- **Scope**: Core enabler (required for real-time performance)
- **Supported Platforms**:
  - **NVIDIA**: CUDA (default) + TensorRT (opt-in)
  - **AMD**: ROCM (opt-in)
  - **Apple Silicon**: CoreML/Metal Performance Shaders (MPS) - default
  - **Windows**: DirectML (opt-in) + GPU acceleration via OpenCV CUDA
  - **CPU**: ONNX CPU fallback
- **Acceptance Criteria**:
  - Provider auto-detection based on hardware
  - Graceful fallback to CPU if GPU unavailable
  - Platform-specific ONNX runtimes (onnxruntime-silicon on ARM Mac)
  - GPU-accelerated OpenCV operations (v2.0.3c)
- **Status**: Active optimization (continuous refinement)

### 9. Content Safety (NSFW Filtering)
- **Description**: Optional filtering to prevent processing inappropriate content
- **Scope**: Ethical safeguard (toggleable, off by default)
- **User Story**: "As a responsible developer, I want content filtering to reduce misuse of this tool"
- **Acceptance Criteria**:
  - OpenNSFW2 model detects inappropriate content
  - Frame-level filtering (skip NSFW frames)
  - Performance: <50ms per frame on GPU
  - Disable via CLI flag for research purposes
- **Status**: Active (re-enabled in v1.x, off by default)

### 10. Internationalization (i18n)
- **Description**: Multi-language UI support
- **Scope**: Non-core enhancement (improves accessibility)
- **Supported Languages**: Indonesian, Japanese, Chinese (Simplified/Traditional), Korean, Russian, German, Spanish, French (8 languages as of v2.0.3c)
- **Status**: Active (8 language packs, community contributions)

## Non-Functional Requirements

### Performance
- **Live Streaming**: 30-60 FPS on target GPU (NVIDIA RTX 3060+, M1/M2+ macOS, AMD Radeon RX 6700+)
- **Batch Processing**: <100ms per frame average (1920x1080, single face)
- **Face Detection**: <20ms per 1080p frame
- **Face Enhancement (GFPGAN)**: <15ms per face on GPU
- **Model Loading**: <3 seconds total (lazy load, cache)

### Reliability
- **Model Availability**: Fallback mechanism for unavailable models (v2.x)
- **GPU Failover**: Automatic CPU fallback if GPU unavailable
- **No Memory Leaks**: Long-running live streams (15+ minutes) without FPS degradation
- **Cross-Platform**: Windows, macOS (Intel/ARM), Linux (NVIDIA, AMD, Intel)

### Usability
- **3-Click Workflow**: Select source image → select camera → press Live (core use case)
- **GUI Responsiveness**: <200ms latency between UI control and effect
- **Camera Enumeration**: Auto-detect available cameras (Windows/macOS/Linux)
- **Accessibility**: Resizable windows, high-contrast UI (CustomTkinter dark mode)

### Security & Ethics
- **No Cloud Upload**: All processing local (no external API calls except model downloads)
- **NSFW Safeguard**: Built-in check (disabled by default, can be enabled)
- **Consent Notice**: README prominently states ethical requirements
- **Watermarking**: Framework in place for mandatory output watermarking (legal compliance)

## Technical Constraints

### Environment
- **Python 3.10 only** (pinned via `.python-version`, pyproject.toml `requires-python = "==3.10.*"`)
- **ffmpeg**: Required on PATH (video processing)
- **Tcl/Tk**: Auto-configured by justfile (Tkinter/GUI)

### Models (Downloaded on First Use)
- **inswapper_128_fp16.onnx** (~379 MB) - Face swapping
- **GFPGANv1.4.pth** (~348 MB) - Face enhancement (optional)
- **buffalo_l.zip** (~370 MB) - InsightFace detection (unzipped: ~2GB)

### Dependencies
- **Core Inference**: ONNX Runtime (platform-specific flavor)
- **Detection**: InsightFace (0.7.3)
- **Enhancement**: PyTorch + GFPGAN + BasicSR (git versions)
- **GUI**: CustomTkinter (5.2.2) + Tcl/Tk
- **Video**: OpenCV (4.10+) + ffmpeg

## Success Metrics

### For Technical Users
- Model optimization accuracy (FPS improvement over time)
- Ease of extending processor pipeline
- Performance benchmarks across GPU platforms
- Code maintainability (modular plugin system)

### For Content Creators
- 3-click ease of use (UX time-to-first-swap)
- Output quality (face realism post-swap)
- Real-time performance (FPS consistency)
- Feature set comprehensiveness (mouth mask, face mapping, etc.)

### Community Health
- Open-source contributions (GitHub stars: 11K+, forks: 900+)
- Localization coverage (8+ languages)
- Issue resolution time
- Feature request fulfillment

## Roadmap & Future Work

### Short-term (Next 2-3 releases)
- Continuous optimization (GPU acceleration per component)
- Stability improvements (test coverage increase from 0% to >50%)
- Model experimentation (alternative face swap models)
- Performance benchmarking tools (FPS monitoring, profiling)

### Medium-term
- Additional platform support (Jetson, mobile devices)
- Advanced features (gaze redirection, expression transfer)
- Commercial variant (watermarking, usage restrictions)
- API server mode (remote face swapping via REST API)

### Long-term
- Photorealistic output quality (GAN-based post-processing)
- Real-time style transfer (character consistency)
- Mobile/edge deployment
- Commercial licensing options

## Version History

| Version | Release Date | Major Changes |
|---------|--------------|---------------|
| 0.x | Sep-Dec 2023 | Initial launch: ONNX face swapping with CustomTkinter GUI |
| 1.x | Jan-Sep 2024 | Feature explosion: Face enhancement, mouth mask, face mapping, i18n, GPU platform support |
| 2.0c | Oct 12, 2025 | Major release: Refactoring, model fallback, multi-language support |
| 2.0.2c - 2.0.3c | Dec 2025 - Feb 2026 | Optimization: Poisson blending, GPU-accelerated OpenCV, face analyzer speedup |

## References

- **Git History**: 508 commits (Sep 2023 - Feb 2026)
- **Repository**: https://github.com/hacksider/Deep-Live-Cam (original), https://github.com/laurigates/Deep-Live-Cam (fork)
- **Primary Documentation**: README.md, CONTRIBUTING.md, CLAUDE.md
- **Architecture Reference**: modules/globals.py (configuration), modules/processors/frame/core.py (plugin system)

---

**Document Status**: Generated from git history (Feb 18, 2026) | Confidence: High | Git Quality: 2/10
