import os
from pathlib import Path


PREFERRED_FACE_SWAPPER_MODELS = (
    "inswapper_128.onnx",
    "inswapper_128_fp16.onnx",
)


def resolve_face_swapper_model_path(models_dir: str | os.PathLike[str]) -> str:
    models_path = Path(models_dir)

    for filename in PREFERRED_FACE_SWAPPER_MODELS:
        model_path = models_path / filename
        if model_path.exists():
            return str(model_path)

    return str(models_path / PREFERRED_FACE_SWAPPER_MODELS[0])
