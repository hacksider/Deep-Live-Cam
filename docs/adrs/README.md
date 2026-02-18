# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for Deep-Live-Cam. ADRs document significant architectural decisions, their context, rationale, and trade-offs.

## Overview

| ADR | Title | Status | Date | Component |
|-----|-------|--------|------|-----------|
| [0001](0001-use-onnx-and-insightface-for-face-detection-and-swap.md) | Use ONNX and InsightFace for Face Detection and Swapping | Accepted | Sep 2023 | Core Inference |
| [0002](0002-plugin-architecture-for-frame-processors.md) | Plugin Architecture for Frame Processors | Accepted | 2024 | Architecture |
| [0003](0003-customtkinter-for-cross-platform-gui.md) | CustomTkinter for Cross-Platform GUI | Accepted | 2023 | UI Framework |
| [0004](0004-platform-specific-gpu-runtime-selection.md) | Platform-Specific GPU Runtime Selection | Accepted | Jul 2024 | GPU Acceleration |
| [0005](0005-global-mutable-state-for-configuration.md) | Global Mutable State for Configuration Management | Accepted | Sep 2023 | Architecture |
| [0006](0006-dual-runtime-pytorch-onnx-separation.md) | Dual-Runtime Approach (PyTorch + ONNX) | Accepted | 2024-2025 | Inference |
| [0007](0007-model-fallback-and-switching-mechanism.md) | Model Fallback and Switching Mechanism | Accepted | Aug 2025 | Resilience |
| [0008](0008-threadpoolexecutor-for-parallel-frame-processing.md) | ThreadPoolExecutor for Parallel Frame Processing | Accepted | 2024 | Performance |
| [0009](0009-nsfw-content-filtering-optional-safeguard.md) | NSFW Content Filtering (Optional Safeguard) | Accepted | Aug 2024 | Safety |
| [0010](0010-continuous-gpu-acceleration-optimization.md) | Continuous GPU Acceleration Optimization Strategy | Accepted | 2024-2026 | Performance |

## ADR Status Definitions

- **Proposed**: Suggested but not yet accepted
- **Accepted**: Approved and currently in use
- **Deprecated**: Superseded by another decision
- **Superseded by**: Links to newer ADR that replaces this one

## Key Architectural Themes

### Core System
- **Inference**: ONNX-based face swapping with InsightFace detection ([ADR 0001](0001-use-onnx-and-insightface-for-face-detection-and-swap.md))
- **Extensibility**: Plugin architecture for frame processors ([ADR 0002](0002-plugin-architecture-for-frame-processors.md))
- **Configuration**: Global mutable state management ([ADR 0005](0005-global-mutable-state-for-configuration.md))

### User Interface
- **Framework**: CustomTkinter for desktop UI ([ADR 0003](0003-customtkinter-for-cross-platform-gui.md))

### GPU & Performance
- **Runtimes**: Platform-specific selection (CUDA, CoreML, TensorRT, etc.) ([ADR 0004](0004-platform-specific-gpu-runtime-selection.md))
- **Dual-Runtime**: PyTorch for enhancement, ONNX for swap ([ADR 0006](0006-dual-runtime-pytorch-onnx-separation.md))
- **Parallelism**: ThreadPoolExecutor for frame batching ([ADR 0008](0008-threadpoolexecutor-for-parallel-frame-processing.md))
- **Optimization**: Continuous improvement across versions ([ADR 0010](0010-continuous-gpu-acceleration-optimization.md))

### Resilience & Safety
- **Model Fallback**: Alternative sources if primary unavailable ([ADR 0007](0007-model-fallback-and-switching-mechanism.md))
- **Content Safety**: Optional NSFW filtering ([ADR 0009](0009-nsfw-content-filtering-optional-safeguard.md))

## How to Use This Directory

### For Contributors
- Read relevant ADRs before modifying related components
- Propose new ADRs for significant architectural changes
- Use existing ADRs as reference for similar decisions in other areas

### For New Features
- Check if any existing ADRs impact your feature
- If making a new architectural decision, create an ADR (see template below)
- Link to related ADRs in your feature documentation

### For Architecture Reviews
- Use ADRs to understand design rationale
- Challenge decisions if better alternatives exist (propose new ADR)
- Keep ADRs updated as architectural decisions evolve

## Creating New ADRs

Use this template for new decisions:

```markdown
# ADR NNNN: [Decision Title]

## Status
**Proposed** | **Accepted** | **Deprecated** | **Superseded by [ADR XXXX]**

## Context
[Why this decision matters, what problem does it solve]

## Decision
[What we decided to do and why]

## Consequences

### Positive
✓ [Benefits of this decision]

### Negative
✗ [Trade-offs and drawbacks]

### Mitigations
[How we address negative consequences]

## Evidence
[Git commits, performance metrics, architectural examples]

## Related Decisions
- [ADR XXXX: Title](aaaa-title.md)

## Future Improvements
[Potential refinements or migrations]

**Last Reviewed**: [Date] | **Confidence**: [High/Medium/Low]
```

### File Naming
- Numeric prefix: `NNNN-` (four digits, zero-padded)
- Kebab-case title: `descriptive-title.md`
- Example: `0001-use-onnx-and-insightface-for-face-detection-and-swap.md`

## ADR Lifecycle

1. **Proposed**: Create ADR with "Proposed" status
2. **Discussion**: Share with team, gather feedback
3. **Accepted**: Update status to "Accepted" after consensus
4. **Review**: Re-review periodically (annually or on major changes)
5. **Deprecate**: Mark as "Deprecated" when no longer relevant
6. **Supersede**: Link to newer ADR that replaces it

## References

- [MADR: Markdown Architecture Decision Records](https://adr.github.io/madr/) - Template inspiration
- [Architecture Decision Records (ADRs)](https://adr.github.io/) - General ADR information
- Deep-Live-Cam [CLAUDE.md](../CLAUDE.md) - Project guidelines
- Deep-Live-Cam [PRD](../prds/deep-live-cam-overview.md) - Feature requirements

---

**Last Updated**: Feb 18, 2026 | **Total ADRs**: 10 | **Accepted**: 10
