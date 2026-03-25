import os
import shutil
from typing import Any
import insightface
import threading
import platform
import inspect

import cv2
import numpy as np
import modules.globals
from tqdm import tqdm
from modules.typing import Frame
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import (
    get_temp_directory_path,
    create_temp,
    extract_frames,
    clean_temp,
    get_temp_frame_paths,
)
from pathlib import Path
from insightface.app.common import Face

try:
    from uniface import (
        create_detector as create_mlx_detector,
        create_recognizer as create_mlx_recognizer,
    )
    import uniface.constants as uniface_constants

    HAS_MLX_UNIFACE = True
except Exception:
    HAS_MLX_UNIFACE = False
    uniface_constants = None

FACE_ANALYSER = None
FACE_ANALYSER_LOCK = threading.Lock()
FACE_ANALYSER_INFERENCE_LOCK = threading.Lock()
FACE_ANALYSER_CPU_ONLY = False
FACE_ANALYSER_COREML_WARNING_PRINTED = False
FACE_ANALYSER_RUNTIME_FALLBACK_PRINTED = False
FACE_ANALYSER_MLX_WARNING_PRINTED = False
FACE_ANALYSER_MLX_RUNTIME_FALLBACK_PRINTED = False

MLX_DETECTOR = None
MLX_RECOGNIZER = None
MLX_ANALYSER_LOCK = threading.Lock()
MLX_RUNTIME_LOCK = threading.Lock()
MLX_DETECTOR_MODEL = None
MLX_FAILED_DETECTORS: set[str] = set()
MLX_DETECTOR_WARNING_PRINTED: set[str] = set()
MLX_URL_PATCHED = False
MLX_MIN_DET_SCORE = 0.0
MLX_MAX_FACES = 8
MLX_INVALID_DETECTOR_WARNING_PRINTED = False

MLX_ENGINE_NAME = "mlx_uniface"
INSIGHTFACE_ENGINE_NAME = "insightface"


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _get_face_engine() -> str:
    return getattr(modules.globals, "face_analyser_engine", INSIGHTFACE_ENGINE_NAME)


def _get_mlx_detector_name() -> str:
    global MLX_INVALID_DETECTOR_WARNING_PRINTED
    detector_name = getattr(modules.globals, "mlx_face_detector", "retinaface")
    if detector_name not in ("retinaface",):
        if not MLX_INVALID_DETECTOR_WARNING_PRINTED:
            print(
                f"[DLC.FACE-ANALYSER] Unsupported MLX detector '{detector_name}'. "
                "Using 'retinaface'."
            )
            MLX_INVALID_DETECTOR_WARNING_PRINTED = True
        return "retinaface"
    return detector_name


def _mlx_backend_available() -> bool:
    return HAS_MLX_UNIFACE and _is_apple_silicon()


def _patch_mlx_weight_urls() -> None:
    global MLX_URL_PATCHED
    if MLX_URL_PATCHED or uniface_constants is None:
        return

    mlx_urls = getattr(uniface_constants, "MODEL_URLS_MLX", None)
    if not isinstance(mlx_urls, dict):
        MLX_URL_PATCHED = True
        return

    old_prefix = "https://github.com/yakhyo/uniface/releases/download/weights-mlx-v1/"
    new_prefix = "https://github.com/CodeWithBehnam/mlx-uniface/releases/download/weights-mlx-v1/"
    changed = 0
    for model_key, model_url in list(mlx_urls.items()):
        if isinstance(model_url, str) and model_url.startswith(old_prefix):
            mlx_urls[model_key] = model_url.replace(old_prefix, new_prefix, 1)
            changed += 1

    mlx_hashes = getattr(uniface_constants, "MODEL_SHA256_MLX", None)
    if isinstance(mlx_hashes, dict):
        RetinaFaceWeights = getattr(uniface_constants, "RetinaFaceWeights", None)
        ArcFaceWeights = getattr(uniface_constants, "ArcFaceWeights", None)
        if RetinaFaceWeights and ArcFaceWeights:
            correct_hashes = {
                RetinaFaceWeights.MNET_V2: "f619c9a726ca0c88333faf43abe98c8e132fc4bf8bafdd5de664b0ef16577058",
                ArcFaceWeights.MNET: "eb085fc4d0dee0e98a9868f68958820357c4926fd74b089e78a4e05f5069ba09",
            }
            for model_key, correct_hash in correct_hashes.items():
                if model_key in mlx_hashes:
                    mlx_hashes[model_key] = correct_hash

    MLX_URL_PATCHED = True
    if changed:
        print(
            f"[DLC.FACE-ANALYSER] Patched {changed} MLX model URL(s) and hashes to active release host."
        )


def _warn_mlx_unavailable_once() -> None:
    global FACE_ANALYSER_MLX_WARNING_PRINTED
    if FACE_ANALYSER_MLX_WARNING_PRINTED:
        return
    FACE_ANALYSER_MLX_WARNING_PRINTED = True
    print(
        "[DLC.FACE-ANALYSER] MLX-UniFace backend requested but unavailable. "
        "Requires Apple Silicon + mlx + mlx-uniface."
    )


def _get_mlx_analyser() -> tuple[Any, Any] | tuple[None, None]:
    global MLX_DETECTOR, MLX_RECOGNIZER, MLX_DETECTOR_MODEL

    if not _mlx_backend_available():
        return None, None

    detector_name = _get_mlx_detector_name()
    if detector_name in MLX_FAILED_DETECTORS:
        return None, None

    _patch_mlx_weight_urls()
    with MLX_ANALYSER_LOCK:
        if (
            MLX_DETECTOR is None
            or MLX_RECOGNIZER is None
            or MLX_DETECTOR_MODEL != detector_name
        ):
            try:
                detector_kwargs: dict[str, Any] = {}
                create_detector_sig = inspect.signature(create_mlx_detector)
                create_detector_params = create_detector_sig.parameters
                accepts_kwargs = any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in create_detector_params.values()
                )
                if "conf_thresh" in create_detector_params or accepts_kwargs:
                    detector_kwargs["conf_thresh"] = 0.35
                if "nms_thresh" in create_detector_params or accepts_kwargs:
                    detector_kwargs["nms_thresh"] = 0.4
                if "backend" in create_detector_sig.parameters or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in create_detector_sig.parameters.values()
                ):
                    detector_kwargs["backend"] = "mlx"
                MLX_DETECTOR = create_mlx_detector(detector_name, **detector_kwargs)
                MLX_RECOGNIZER = create_mlx_recognizer("arcface")
                MLX_DETECTOR_MODEL = detector_name
                print(
                    f"[DLC.FACE-ANALYSER] MLX-UniFace initialized (detector={detector_name}, recognizer=arcface)."
                )
            except Exception as e:
                MLX_DETECTOR = None
                MLX_RECOGNIZER = None
                MLX_DETECTOR_MODEL = None
                MLX_FAILED_DETECTORS.add(detector_name)
                if detector_name not in MLX_DETECTOR_WARNING_PRINTED:
                    print(
                        f"[DLC.FACE-ANALYSER] MLX init failed for detector={detector_name}. "
                        f"Falling back to InsightFace. ({e})"
                    )
                    MLX_DETECTOR_WARNING_PRINTED.add(detector_name)
                return None, None
    return MLX_DETECTOR, MLX_RECOGNIZER


def _mark_mlx_detector_failed(detector_name: str, reason: str) -> None:
    MLX_FAILED_DETECTORS.add(detector_name)
    if detector_name not in MLX_DETECTOR_WARNING_PRINTED:
        print(
            f"[DLC.FACE-ANALYSER] MLX detector={detector_name} disabled for this session. "
            f"Falling back to InsightFace. ({reason})"
        )
        MLX_DETECTOR_WARNING_PRINTED.add(detector_name)


def _coerce_uniface_face_dict(face_data: Any) -> dict[str, Any]:
    if isinstance(face_data, dict):
        return face_data
    if hasattr(face_data, "to_dict") and callable(face_data.to_dict):
        try:
            return face_data.to_dict()
        except Exception:
            pass
    return {
        "bbox": getattr(face_data, "bbox", None),
        "landmarks": getattr(face_data, "landmarks", None),
        "confidence": getattr(face_data, "confidence", None),
        "embedding": getattr(face_data, "embedding", None),
    }


def _mlx_face_to_insightface(
    face_data: Any, embedding: np.ndarray | None
) -> Face | None:
    face_dict = _coerce_uniface_face_dict(face_data)
    bbox = face_dict.get("bbox")
    kps = face_dict.get("landmarks")
    if kps is None:
        kps = face_dict.get("kps")
    if bbox is None or kps is None:
        return None

    try:
        bbox_arr = np.asarray(bbox, dtype=np.float32)
        kps_arr = np.asarray(kps, dtype=np.float32)
        emb_arr = (
            np.asarray(embedding, dtype=np.float32) if embedding is not None else None
        )
        det_score = float(face_dict.get("confidence", face_dict.get("score", 0.0)))
    except Exception:
        return None

    if bbox_arr.size < 4 or kps_arr.size < 10:
        return None
    if not np.all(np.isfinite(bbox_arr)) or not np.all(np.isfinite(kps_arr)):
        return None
    if emb_arr is not None and not np.all(np.isfinite(emb_arr)):
        emb_arr = None

    return Face(
        {
            "bbox": bbox_arr,
            "kps": kps_arr,
            "embedding": emb_arr,
            "det_score": det_score,
            "landmark_2d_106": None,
        }
    )


def _get_many_faces_mlx(
    frame: Frame, require_embedding: bool = True
) -> list[Face] | None:
    global FACE_ANALYSER_MLX_RUNTIME_FALLBACK_PRINTED

    detector, recognizer = _get_mlx_analyser()
    if detector is None or recognizer is None:
        return None

    # MLX UniFace backend is not thread-safe for concurrent inference calls.
    # Serialize detector/recognizer execution to avoid hard crashes on macOS.
    with MLX_RUNTIME_LOCK:
        try:
            if hasattr(detector, "detect_faces"):
                detected_faces = detector.detect_faces(frame) or []
            elif hasattr(detector, "detect"):
                max_num = (
                    1
                    if not getattr(modules.globals, "many_faces", False)
                    else MLX_MAX_FACES
                )
                detected_faces = (
                    detector.detect(frame, max_num=max_num, metric="default") or []
                )
            else:
                detected_faces = []
        except Exception as e:
            if not FACE_ANALYSER_MLX_RUNTIME_FALLBACK_PRINTED:
                print(
                    f"[DLC.FACE-ANALYSER] MLX detector error. Falling back to InsightFace. ({e})"
                )
                FACE_ANALYSER_MLX_RUNTIME_FALLBACK_PRINTED = True
            return None

        results: list[Face] = []
        for face_data in detected_faces:
            face_dict = _coerce_uniface_face_dict(face_data)
            embedding = None
            landmarks = face_dict.get("landmarks")
            if landmarks is None:
                landmarks = face_dict.get("kps")
            if require_embedding and embedding is None:
                native_embedding = face_dict.get("embedding")
                if native_embedding is not None:
                    try:
                        embedding = np.asarray(native_embedding, dtype=np.float32)
                    except Exception:
                        embedding = None
            if require_embedding and landmarks is not None:
                try:
                    embedding = recognizer.get_normalized_embedding(frame, landmarks)
                except Exception:
                    embedding = None

            face_obj = _mlx_face_to_insightface(face_dict, embedding)
            if face_obj is None:
                continue
            if float(getattr(face_obj, "det_score", 0.0) or 0.0) < MLX_MIN_DET_SCORE:
                continue
            if require_embedding:
                normed_embedding = getattr(face_obj, "normed_embedding", None)
                if normed_embedding is None:
                    continue
                try:
                    normed_embedding = np.asarray(normed_embedding, dtype=np.float32)
                except Exception:
                    continue
                if normed_embedding.size == 0 or not np.all(
                    np.isfinite(normed_embedding)
                ):
                    continue
            results.append(face_obj)
        if len(results) > MLX_MAX_FACES:
            results = sorted(
                results,
                key=lambda face: float(getattr(face, "det_score", 0.0) or 0.0),
                reverse=True,
            )[:MLX_MAX_FACES]
        return results


def _resolve_face_analyser_providers(force_cpu: bool = False) -> list[str]:
    global FACE_ANALYSER_COREML_WARNING_PRINTED

    if force_cpu:
        return ["CPUExecutionProvider"]

    providers = list(modules.globals.execution_providers or ["CPUExecutionProvider"])

    # On macOS, InsightFace detection/recognition models can fail with CoreML
    # due to dynamic shape incompatibilities. Keep CoreML for other modules,
    # but run face analyser on CPU for stability.
    if platform.system() == "Darwin" and "CoreMLExecutionProvider" in providers:
        providers = [
            provider for provider in providers if provider != "CoreMLExecutionProvider"
        ]
        if not providers:
            providers = ["CPUExecutionProvider"]
        if not FACE_ANALYSER_COREML_WARNING_PRINTED:
            print(
                "[DLC.FACE-ANALYSER] CoreML disabled for face analyser on macOS (dynamic-shape incompatibility)."
            )
            FACE_ANALYSER_COREML_WARNING_PRINTED = True
    return providers


_FACE_ANALYSER_CACHE = {}
_FACE_ANALYSER_CACHE_MAX_SIZE = 10
_FACE_ANALYSER_LAST_FRAME_HASH = None
_FACE_ANALYSER_CACHE_HIT = False


def _frame_hash(frame: Frame) -> int:
    """Fast hash for frame caching - uses downscaled version."""
    if frame is None:
        return 0
    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_NEAREST)
    return hash(small.tobytes())


def reset_face_analyser_state() -> None:
    """Reset runtime analysers/caches so backend/model switches apply immediately."""
    global \
        FACE_ANALYSER, \
        FACE_ANALYSER_CPU_ONLY, \
        FACE_ANALYSER_RUNTIME_FALLBACK_PRINTED, \
        FACE_ANALYSER_MLX_WARNING_PRINTED, \
        FACE_ANALYSER_MLX_RUNTIME_FALLBACK_PRINTED, \
        MLX_DETECTOR, \
        MLX_RECOGNIZER, \
        MLX_DETECTOR_MODEL, \
        _FACE_ANALYSER_CACHE, \
        _FACE_ANALYSER_LAST_FRAME_HASH, \
        _FACE_ANALYSER_CACHE_HIT

    with FACE_ANALYSER_LOCK:
        FACE_ANALYSER = None
        FACE_ANALYSER_CPU_ONLY = False
    with MLX_ANALYSER_LOCK:
        MLX_DETECTOR = None
        MLX_RECOGNIZER = None
        MLX_DETECTOR_MODEL = None

    MLX_FAILED_DETECTORS.clear()
    MLX_DETECTOR_WARNING_PRINTED.clear()
    FACE_ANALYSER_RUNTIME_FALLBACK_PRINTED = False
    FACE_ANALYSER_MLX_WARNING_PRINTED = False
    FACE_ANALYSER_MLX_RUNTIME_FALLBACK_PRINTED = False
    _FACE_ANALYSER_CACHE = {}
    _FACE_ANALYSER_LAST_FRAME_HASH = None
    _FACE_ANALYSER_CACHE_HIT = False


def _create_face_analyser(force_cpu: bool = False) -> Any:
    providers = _resolve_face_analyser_providers(force_cpu=force_cpu)
    use_cpu_only = all(provider == "CPUExecutionProvider" for provider in providers)
    ctx_id = -1 if use_cpu_only else 0
    # Optimized detector size for macOS live mode
    det_size = (320, 320)
    if platform.system() == "Darwin":
        if getattr(modules.globals, "live_mode", False):
            det_size = (128, 128)
        elif not modules.globals.many_faces:
            det_size = (256, 256)
    analyser = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=providers,
        allowed_modules=["detection", "recognition"],
    )
    analyser.prepare(ctx_id=ctx_id, det_size=det_size)
    return analyser, use_cpu_only


def get_face_analyser(force_cpu: bool = False) -> Any:
    """Get face analyser with thread-safe initialization."""
    global FACE_ANALYSER, FACE_ANALYSER_CPU_ONLY

    if FACE_ANALYSER is None or (force_cpu and not FACE_ANALYSER_CPU_ONLY):
        with FACE_ANALYSER_LOCK:
            # Double-check after acquiring lock
            if FACE_ANALYSER is None or (force_cpu and not FACE_ANALYSER_CPU_ONLY):
                FACE_ANALYSER, FACE_ANALYSER_CPU_ONLY = _create_face_analyser(
                    force_cpu=force_cpu
                )
    return FACE_ANALYSER


def get_one_face(
    frame: Frame,
    many_faces: Any = None,
    require_embedding: bool = True,
    allow_fallback: bool = True,
) -> Any:
    global \
        FACE_ANALYSER_RUNTIME_FALLBACK_PRINTED, \
        _FACE_ANALYSER_CACHE, \
        _FACE_ANALYSER_LAST_FRAME_HASH, \
        _FACE_ANALYSER_CACHE_HIT

    if many_faces is not None:
        try:
            return min(many_faces, key=lambda x: x.bbox[0])
        except Exception:
            pass

    if frame is None:
        return None

    is_live = getattr(modules.globals, "live_mode", False)
    if is_live and platform.system() == "Darwin":
        frame_h = _frame_hash(frame)
        if frame_h == _FACE_ANALYSER_LAST_FRAME_HASH and _FACE_ANALYSER_CACHE_HIT:
            cached = _FACE_ANALYSER_CACHE.get("one_face")
            if cached is not None:
                return cached
        _FACE_ANALYSER_LAST_FRAME_HASH = frame_h

    mlx_detector_name = _get_mlx_detector_name()
    mlx_empty_result = False
    if _get_face_engine() == MLX_ENGINE_NAME:
        if not _mlx_backend_available():
            _warn_mlx_unavailable_once()
        else:
            mlx_faces = _get_many_faces_mlx(frame, require_embedding=require_embedding)
            if mlx_faces:
                result = min(mlx_faces, key=lambda x: x.bbox[0])
                if is_live:
                    _FACE_ANALYSER_CACHE["one_face"] = result
                    _FACE_ANALYSER_CACHE_HIT = True
                return result
            mlx_empty_result = mlx_faces == []
            if not allow_fallback:
                if is_live:
                    _FACE_ANALYSER_CACHE["one_face"] = None
                    _FACE_ANALYSER_CACHE_HIT = False
                return None

    try:
        with FACE_ANALYSER_INFERENCE_LOCK:
            face = get_face_analyser().get(frame)
    except Exception as e:
        if not FACE_ANALYSER_RUNTIME_FALLBACK_PRINTED:
            print(
                f"[DLC.FACE-ANALYSER] Runtime analyser error. Falling back to CPU. ({e})"
            )
            FACE_ANALYSER_RUNTIME_FALLBACK_PRINTED = True
        try:
            with FACE_ANALYSER_INFERENCE_LOCK:
                face = get_face_analyser(force_cpu=True).get(frame)
        except Exception:
            return None
    try:
        selected_face = min(face, key=lambda x: x.bbox[0])
        if mlx_empty_result and selected_face is not None:
            _mark_mlx_detector_failed(
                mlx_detector_name,
                "no confident MLX detections while InsightFace found face(s)",
            )
        if is_live:
            _FACE_ANALYSER_CACHE["one_face"] = selected_face
            _FACE_ANALYSER_CACHE_HIT = True
        return selected_face
    except ValueError:
        if is_live:
            _FACE_ANALYSER_CACHE["one_face"] = None
            _FACE_ANALYSER_CACHE_HIT = False
        return None


def get_many_faces(
    frame: Frame,
    require_embedding: bool = True,
    allow_fallback: bool = True,
) -> Any:
    global FACE_ANALYSER_RUNTIME_FALLBACK_PRINTED

    if frame is None:
        return []

    mlx_detector_name = _get_mlx_detector_name()
    mlx_empty_result = False
    if _get_face_engine() == MLX_ENGINE_NAME:
        if not _mlx_backend_available():
            _warn_mlx_unavailable_once()
        else:
            mlx_faces = _get_many_faces_mlx(frame, require_embedding=require_embedding)
            if mlx_faces:
                return mlx_faces
            mlx_empty_result = mlx_faces == []
            if not allow_fallback:
                return []

    try:
        with FACE_ANALYSER_INFERENCE_LOCK:
            fallback_faces = get_face_analyser().get(frame)
        if mlx_empty_result and fallback_faces:
            _mark_mlx_detector_failed(
                mlx_detector_name,
                "no confident MLX detections while InsightFace found face(s)",
            )
        return fallback_faces
    except Exception as e:
        if not FACE_ANALYSER_RUNTIME_FALLBACK_PRINTED:
            print(
                f"[DLC.FACE-ANALYSER] Runtime analyser error. Falling back to CPU. ({e})"
            )
            FACE_ANALYSER_RUNTIME_FALLBACK_PRINTED = True
        try:
            with FACE_ANALYSER_INFERENCE_LOCK:
                fallback_faces = get_face_analyser(force_cpu=True).get(frame)
            if mlx_empty_result and fallback_faces:
                _mark_mlx_detector_failed(
                    mlx_detector_name,
                    "no confident MLX detections while CPU fallback found face(s)",
                )
            return fallback_faces
        except Exception:
            return []


def has_valid_map() -> bool:
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            return True
    return False


def default_source_face() -> Any:
    for map in modules.globals.source_target_map:
        if "source" in map:
            return map["source"]["face"]
    return None


def simplify_maps() -> Any:
    centroids = []
    faces = []
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            centroids.append(map["target"]["face"].normed_embedding)
            faces.append(map["source"]["face"])

    modules.globals.simple_map = {"source_faces": faces, "target_embeddings": centroids}
    return None


def add_blank_map() -> Any:
    try:
        max_id = -1
        if len(modules.globals.source_target_map) > 0:
            max_id = max(modules.globals.source_target_map, key=lambda x: x["id"])["id"]

        modules.globals.source_target_map.append({"id": max_id + 1})
    except ValueError:
        return None


def get_unique_faces_from_target_image() -> Any:
    try:
        modules.globals.source_target_map = []
        target_frame = cv2.imread(modules.globals.target_path)
        many_faces = get_many_faces(target_frame)
        i = 0

        for face in many_faces:
            x_min, y_min, x_max, y_max = face["bbox"]
            modules.globals.source_target_map.append(
                {
                    "id": i,
                    "target": {
                        "cv2": target_frame[
                            int(y_min) : int(y_max), int(x_min) : int(x_max)
                        ],
                        "face": face,
                    },
                }
            )
            i = i + 1
    except ValueError:
        return None


def get_unique_faces_from_target_video() -> Any:
    try:
        modules.globals.source_target_map = []
        frame_face_embeddings = []
        face_embeddings = []

        print("Creating temp resources...")
        clean_temp(modules.globals.target_path)
        create_temp(modules.globals.target_path)
        print("Extracting frames...")
        extract_frames(modules.globals.target_path)

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)

        i = 0
        for temp_frame_path in tqdm(
            temp_frame_paths, desc="Extracting face embeddings from frames"
        ):
            temp_frame = cv2.imread(temp_frame_path)
            many_faces = get_many_faces(temp_frame)

            for face in many_faces:
                face_embeddings.append(face.normed_embedding)

            frame_face_embeddings.append(
                {"frame": i, "faces": many_faces, "location": temp_frame_path}
            )
            i += 1

        centroids = find_cluster_centroids(face_embeddings)

        for frame in frame_face_embeddings:
            for face in frame["faces"]:
                closest_centroid_index, _ = find_closest_centroid(
                    centroids, face.normed_embedding
                )
                face["target_centroid"] = closest_centroid_index

        for i in range(len(centroids)):
            modules.globals.source_target_map.append({"id": i})

            temp = []
            for frame in tqdm(
                frame_face_embeddings, desc=f"Mapping frame embeddings to centroids-{i}"
            ):
                temp.append(
                    {
                        "frame": frame["frame"],
                        "faces": [
                            face
                            for face in frame["faces"]
                            if face["target_centroid"] == i
                        ],
                        "location": frame["location"],
                    }
                )

            modules.globals.source_target_map[i]["target_faces_in_frame"] = temp

        # dump_faces(centroids, frame_face_embeddings)
        default_target_face()
    except ValueError:
        return None


def default_target_face():
    for map in modules.globals.source_target_map:
        best_face = None
        best_frame = None
        for frame in map["target_faces_in_frame"]:
            if len(frame["faces"]) > 0:
                best_face = frame["faces"][0]
                best_frame = frame
                break

        for frame in map["target_faces_in_frame"]:
            for face in frame["faces"]:
                if face["det_score"] > best_face["det_score"]:
                    best_face = face
                    best_frame = frame

        x_min, y_min, x_max, y_max = best_face["bbox"]

        target_frame = cv2.imread(best_frame["location"])
        map["target"] = {
            "cv2": target_frame[int(y_min) : int(y_max), int(x_min) : int(x_max)],
            "face": best_face,
        }


def dump_faces(centroids: Any, frame_face_embeddings: list):
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)

    for i in range(len(centroids)):
        if os.path.exists(temp_directory_path + f"/{i}") and os.path.isdir(
            temp_directory_path + f"/{i}"
        ):
            shutil.rmtree(temp_directory_path + f"/{i}")
        Path(temp_directory_path + f"/{i}").mkdir(parents=True, exist_ok=True)

        for frame in tqdm(frame_face_embeddings, desc=f"Copying faces to temp/./{i}"):
            temp_frame = cv2.imread(frame["location"])

            j = 0
            for face in frame["faces"]:
                if face["target_centroid"] == i:
                    x_min, y_min, x_max, y_max = face["bbox"]

                    if (
                        temp_frame[
                            int(y_min) : int(y_max), int(x_min) : int(x_max)
                        ].size
                        > 0
                    ):
                        cv2.imwrite(
                            temp_directory_path + f"/{i}/{frame['frame']}_{j}.png",
                            temp_frame[
                                int(y_min) : int(y_max), int(x_min) : int(x_max)
                            ],
                        )
                j += 1
