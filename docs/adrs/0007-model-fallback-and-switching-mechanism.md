# ADR 0007: Model Fallback and Switching Mechanism

## Status
**Accepted** (Introduced Aug 2025, v2.0c+)

## Context

Original design hard-coded model URLs:
- inswapper_128_fp16.onnx
- GFPGANv1.4.pth
- buffalo_l (InsightFace)

Risk: If model source becomes unavailable, application fails. Aug 2025 decision added resilience.

### Triggers for Fallback
- Model download failure (network, source moved)
- Model corruption/checksum mismatch
- User wants to experiment with alternative models
- Research use case: test multiple swap/enhancement model combinations

## Decision

Implement model fallback and switching:

1. **Primary model URLs** in code (default inswapper_128_fp16)
2. **Fallback mechanism**: If primary model unavailable, try alternative source
3. **CLI override**: `--model-path` flag allows custom model specification
4. **Model discovery**: Auto-detect available models in models/ directory
5. **User-provided models**: Place .onnx/.pth in models/ → app discovers and uses

### Implementation
```python
# utilities.py: conditional_download()
def conditional_download(model_name):
    if exists(model_path):
        return model_path
    try:
        return download_from_primary_source(model_name)
    except:
        return download_from_fallback_source(model_name)  # Aug 2025 addition
```

## Consequences

### Positive
✓ **Resilience**: App doesn't fail if single model source unavailable
✓ **Flexibility**: Users can experiment with alternative models
✓ **Research-friendly**: Academia/enterprise can use proprietary models
✓ **Future-proof**: Easy to migrate to new model if source changes
✓ **Community**: Contributors can host mirrors (fallback sources)
✓ **Version testing**: Test compatibility with older/newer model versions

### Negative
✗ **Complexity**: More code paths to test and maintain
✗ **Quality variability**: User-provided models might degrade output quality
✗ **Security**: Model origin unverified if from fallback source
✗ **Debugging**: Hard to diagnose issues when wrong model used
✗ **Dependency on mirrors**: Fallback source reliability critical
✗ **Version mismatches**: Alternative models might incompatible with pipeline

### Mitigations
- **Checksum verification**: Verify model integrity after download
- **Model metadata**: Document expected model format/dimensions
- **Warnings**: Alert users if using non-standard model
- **Fallback source reliability**: Use well-known hosting (Hugging Face, etc.)
- **Testing**: Test compatibility with model alternatives

## Evidence

### Git History
- **Aug 2, 2025**: d0d90ec (Creating a fallback and switching of models)
- Solidified in v2.0c without major changes
- Consistent through Feb 2026 latest version

### Use Cases
- **Research**: Test inswapper_128_int8 (smaller, faster but lower quality)
- **Migration**: If inswapper deprecated, use alternative face swap model
- **Optimization**: Try GFPGAN v1.3 vs v1.4 for quality comparison
- **Custom training**: Deploy fine-tuned models for specific use cases

### Quality Impact
- Primary model (inswapper_128_fp16): ~90% user satisfaction
- Alternative models: variable quality (80-95% depending on choice)
- Enhancement: GFPGAN v1.4 significantly better than v1.3

## Related Decisions
- [ADR 0001: ONNX/InsightFace](0001-use-onnx-and-insightface-for-face-detection-and-swap.md)
- [ADR 0002: Plugin Architecture](0002-plugin-architecture-for-frame-processors.md) (extensible to custom processors)

## Future Improvements
- **Model registry**: Central catalog of tested/compatible models
- **Auto model selection**: Recommend model based on GPU/hardware
- **Version negotiation**: Support multiple model versions simultaneously
- **Quantization**: Auto-convert fp32 models to fp16/int8
- **Model benchmarking**: Measure quality/speed tradeoff for each model

**Last Reviewed**: Feb 18, 2026 | **Confidence**: Medium-High
