import os


PREFERRED_FACE_SWAPPER_MODELS = (
    "inswapper_128.onnx",
    "inswapper_128_fp16.onnx",
)


def resolve_face_swapper_model_path(models_dir: str) -> str:
    for filename in PREFERRED_FACE_SWAPPER_MODELS:
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            return model_path

    return os.path.join(models_dir, PREFERRED_FACE_SWAPPER_MODELS[0])
