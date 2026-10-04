"""Per-person live controls.

The swap code reads process-wide globals. One GPU worker handles every
session, so each frame copies those globals, applies this person's look,
and puts the previous values back before the next session runs.
"""

import logging

logger = logging.getLogger(__name__)

ENHANCERS = ("none", "gfpgan", "gpen512", "gpen256")
ENHANCER_LABELS = {
    "gfpgan": "GFPGAN",
    "gpen512": "GPEN-512",
    "gpen256": "GPEN-256",
}
ENHANCER_FILES = {
    "gfpgan": "gfpgan-1024.onnx",
    "gpen512": "GPEN-BFR-512.onnx",
    "gpen256": "GPEN-BFR-256.onnx",
}
_ENHANCER_MODULES = {
    "gfpgan": "modules.processors.frame.face_enhancer",
    "gpen512": "modules.processors.frame.face_enhancer_gpen512",
    "gpen256": "modules.processors.frame.face_enhancer_gpen256",
}
_FP_UI = {
    "gfpgan": "face_enhancer",
    "gpen512": "face_enhancer_gpen512",
    "gpen256": "face_enhancer_gpen256",
}
_FIELDS = (
    "opacity",
    "sharpness",
    "mouth_mask",
    "mouth_mask_size",
    "show_mouth_mask_box",
    "many_faces",
    "poisson_blend",
    "enable_interpolation",
)


def default_look() -> dict:
    return {
        "opacity": 1.0,
        "sharpness": 0.0,
        "mouth": 0.0,
        "many_faces": False,
        "poisson": False,
        "enhancer": "none",
    }


def parse_look(payload: dict) -> tuple[dict | None, str | None]:
    if not isinstance(payload, dict):
        return None, "bad look"
    opacity = _number(payload.get("opacity"), 0.0, 1.0)
    sharpness = _number(payload.get("sharpness"), 0.0, 5.0)
    mouth = _number(payload.get("mouth"), 0.0, 100.0)
    many_faces = _flag(payload.get("many_faces"))
    poisson = _flag(payload.get("poisson"))
    enhancer = payload.get("enhancer")
    if (
        opacity is None
        or sharpness is None
        or mouth is None
        or many_faces is None
        or poisson is None
        or enhancer not in ENHANCERS
    ):
        return None, "bad look"
    return {
        "opacity": opacity,
        "sharpness": sharpness,
        "mouth": mouth,
        "many_faces": many_faces,
        "poisson": poisson,
        "enhancer": enhancer,
    }, None


def model_installed(name: str) -> bool:
    from modules.model_downloader import is_present

    filename = ENHANCER_FILES.get(name)
    if not filename:
        return False
    return is_present(filename)


def run_enhancer(name: str, frame, faces):
    import importlib

    module = importlib.import_module(_ENHANCER_MODULES[name])
    return module.process_frame(None, frame, detected_faces=faces)


def make_renderer(
    *,
    globals_module,
    detect_one,
    detect_many,
    swap,
    post,
    landmarks,
    enhance,
):
    def render(source_face, frame, look: dict):
        saved = _snapshot(globals_module)
        try:
            _apply(globals_module, look)
            if look["many_faces"]:
                found = detect_many(frame) or []
                faces = [face for face in found if face is not None]
            else:
                one = detect_one(frame)
                faces = [one] if one is not None else []
            if not faces:
                return None
            if look["mouth"] > 0:
                try:
                    landmarks(frame, faces)
                except Exception:
                    logger.exception("mouth landmarks failed")
            if look["enhancer"] != "none":
                try:
                    frame = enhance(look["enhancer"], frame, faces)
                except Exception:
                    logger.exception("enhancer %s failed", look["enhancer"])
            boxes = []
            result = frame
            for face in faces:
                result = swap(source_face, face, result)
                box = _box(face)
                if box is not None:
                    boxes.append(box)
            return post(result, boxes)
        finally:
            _restore(globals_module, saved)

    return render


def make_production_renderer():
    import modules.globals
    from modules.face_analyser import (
        detect_many_faces_fast,
        detect_one_face_fast,
        ensure_landmarks,
    )
    from modules.processors.frame.face_swapper import apply_post_processing, swap_face

    return make_renderer(
        globals_module=modules.globals,
        detect_one=detect_one_face_fast,
        detect_many=detect_many_faces_fast,
        swap=swap_face,
        post=apply_post_processing,
        landmarks=ensure_landmarks,
        enhance=run_enhancer,
    )


def _number(value, low: float, high: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number < low or number > high:
        return None
    return number


def _flag(value) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _snapshot(globals_module) -> dict:
    saved = {name: getattr(globals_module, name) for name in _FIELDS}
    saved["fp_ui"] = dict(getattr(globals_module, "fp_ui", {}))
    return saved


def _apply(globals_module, look: dict) -> None:
    globals_module.opacity = look["opacity"]
    globals_module.sharpness = look["sharpness"]
    globals_module.mouth_mask_size = look["mouth"]
    globals_module.mouth_mask = look["mouth"] > 0
    globals_module.show_mouth_mask_box = False
    globals_module.many_faces = look["many_faces"]
    globals_module.poisson_blend = look["poisson"]
    globals_module.enable_interpolation = False
    flags = dict(getattr(globals_module, "fp_ui", {}))
    for key in _FP_UI.values():
        flags[key] = False
    chosen = _FP_UI.get(look["enhancer"])
    if chosen:
        flags[chosen] = True
    globals_module.fp_ui = flags


def _restore(globals_module, saved: dict) -> None:
    for name in _FIELDS:
        setattr(globals_module, name, saved[name])
    globals_module.fp_ui = dict(saved["fp_ui"])


def _box(face):
    bbox = getattr(face, "bbox", None)
    if bbox is None:
        return None
    if hasattr(bbox, "astype"):
        bbox = bbox.astype(int)
    try:
        if len(bbox) != 4:
            return None
    except TypeError:
        return None
    return bbox
