# ADR 0009: NSFW Content Filtering (Optional Safeguard)

## Status
**Accepted** (Re-enabled Aug 2024 after initial implementation; off by default)

## Context

Deepfake tools have ethical implications. Early design included NSFW detection to prevent misuse on inappropriate content. Implementation evolved:

1. **Initial** (v1.x): NSFW filter integrated but had false positives
2. **Disabled** (mid 2024): Performance overhead, user complaints about strictness
3. **Re-enabled** (Aug 2024): Off by default, users can opt-in

### Ethical Considerations
- Prevent non-consensual intimate imagery generation
- Block sensitive content (war footage, graphic violence)
- Research/creative use should remain possible
- User responsibility emphasized in README

## Decision

Implement optional OpenNSFW2-based content filtering:

```python
# modules/globals.py
nsfw_filter: bool = False  # Default OFF

# core.py: Pre-check phase
if modules.globals.nsfw_filter:
    load_nsfw_detector()  # Lazy load TensorFlow model

# Face processing: Skip NSFW frames
if modules.globals.nsfw_filter:
    if is_nsfw(frame):
        output_frame = input_frame  # Skip processing
        continue
```

### Configuration
- **Default**: OFF (users opt-in)
- **CLI**: `--nsfw-filter` flag to enable
- **UI**: Checkbox in face processor options
- **Performance**: ~50ms per frame overhead (TensorFlow inference)

## Consequences

### Positive
✓ **Ethical framework**: Built-in safeguard against misuse
✓ **User opt-in**: Doesn't penalize creative/research uses
✓ **Transparency**: README explicitly documents ethical stance
✓ **Legal defensibility**: Shows good-faith effort to prevent harm
✓ **Community trust**: Demonstrates responsible development
✓ **Future-proof**: Foundation for content moderation if regulations require

### Negative
✗ **False positives**: Rejects innocent content (e.g., artistic nudes)
✗ **Performance cost**: TensorFlow model adds 50ms/frame
✗ **Model dependency**: OpenNSFW2 model must be downloaded/loaded
✗ **Ineffectiveness**: Motivated users can disable filter easily
✗ **Legal gray area**: Liability unclear if filter fails or bypassed
✗ **Enforcement**: Can't prevent offline/closed-source usage

### Mitigations
- **Default off**: No forced restrictions on legitimate use
- **Clear documentation**: README explains when filter appropriate
- **User control**: Simple flag to enable/disable
- **No hard block**: Skips frames rather than crashing
- **Community alternatives**: Can fork and remove if needed

## Evidence

### Git History
- **Aug 21, 2024**: 7313a33 (Re-enabled NSFW filter, off by default)
- **Multiple references**: globals.py, core.py pre-checks
- **Stabilized**: No changes to filtering logic through v2.x

### Model Specifications
- **OpenNSFW2**: TensorFlow-based binary classifier
- **Input**: Image frame (any resolution)
- **Output**: Probability [0, 1] of inappropriate content
- **Threshold**: Tunable (default: 0.5)
- **Performance**: ~50ms on GPU, ~200ms on CPU

### Ethical Rationale
From README.md:
- "Users expected to use responsibly and legally"
- "If using real person's face, obtain consent"
- "Clearly label output as deepfake"
- "Built-in check prevents processing inappropriate media"

## Related Decisions
- [ADR 0002: Plugin Architecture](0002-plugin-architecture-for-frame-processors.md) (could extend to safety processors)
- [ADR 0005: Global State](0005-global-mutable-state-for-configuration.md) (nsfw_filter flag)

## Future Improvements
- **Content moderation model**: Replace OpenNSFW2 with more accurate model
- **Gradual filtering**: Instead of skip, blur/mask sensitive regions
- **Metadata tracking**: Log filtered frames for transparency
- **Policy system**: Different rules for different content types
- **Integration with platforms**: Automatic moderation for social media upload
- **User education**: In-app disclaimers about ethical use

## Deployment Notes
- Filter CPU-only (TensorFlow CPU fallback)
- Disabled by default to avoid UX friction
- Recommended for production services; optional for personal tools
- Should be paired with watermarking for commercial use

**Last Reviewed**: Feb 18, 2026 | **Confidence**: Medium (effectiveness limited by bypasses)
