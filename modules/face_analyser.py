import os
import shutil
from typing import Any
import insightface
import threading

import cv2
import numpy as np
import modules.globals
from tqdm import tqdm
from modules.typing import Frame
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import get_temp_directory_path, create_temp, extract_frames, clean_temp, get_temp_frame_paths
from pathlib import Path

FACE_ANALYSER = None
FACE_ANALYSER_LOCK = threading.Lock()

DET_SIZE = (640, 640)


def get_face_analyser() -> Any:
    """Get face analyser with thread-safe initialization."""
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        with FACE_ANALYSER_LOCK:
            # Double-check after acquiring lock
            if FACE_ANALYSER is None:
                from modules.processors.frame._onnx_enhancer import (
                    build_provider_config,
                )
                providers = build_provider_config()
                FACE_ANALYSER = insightface.app.FaceAnalysis(
                    name='buffalo_l',
                    providers=providers,
                    allowed_modules=['detection', 'recognition', 'landmark_2d_106']
                )
                FACE_ANALYSER.prepare(ctx_id=0, det_size=DET_SIZE)
                _optimize_det_model(FACE_ANALYSER, providers)
    return FACE_ANALYSER


def _optimize_det_model(fa: Any, providers) -> None:
    """Replace the detection model's ONNX session with a CoreML-optimized one.

    Folds dynamic Shape→Gather chains into constants (the input size is
    fixed at det_size), eliminating CPU↔ANE partition boundaries in the
    RetinaFace FPN upsampling path.  21ms → 4ms on M3 Max.
    """
    from modules.onnx_optimize import optimize_for_coreml, IS_APPLE_SILICON
    if not IS_APPLE_SILICON:
        return

    det_model = fa.det_model
    model_path = getattr(det_model, 'model_file', None)
    if model_path is None or not os.path.exists(model_path):
        return

    input_shape = (1, 3, DET_SIZE[1], DET_SIZE[0])
    optimized_path = optimize_for_coreml(model_path, input_shape=input_shape)
    if optimized_path == model_path:
        return

    import onnxruntime
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # Route detection to GPU shader cores (CPUAndGPU) instead of ANE.
    # This lets detection run concurrently with the swap model on the
    # ANE, overlapping the two inference calls.  Detection is fast
    # enough on GPU (~4ms) and this frees ANE for the heavier swap.
    det_providers = []
    for p in providers:
        name = p[0] if isinstance(p, tuple) else p
        if name == "CoreMLExecutionProvider":
            det_providers.append((
                "CoreMLExecutionProvider",
                {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"},
            ))
        else:
            det_providers.append(p)

    det_model.session = onnxruntime.InferenceSession(
        optimized_path, sess_options=session_options, providers=det_providers,
    )


def _needs_landmark() -> bool:
    """Check whether any active feature requires 106-point landmarks.

    Landmarks are needed by face enhancers and mouth masking, but not
    by the face swapper alone.
    """
    if getattr(modules.globals, "mouth_mask", False):
        return True
    processors = getattr(modules.globals, "frame_processors", [])
    return any(p in processors for p in
               ("face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"))


def _is_dml() -> bool:
    return any("DmlExecutionProvider" in p for p in modules.globals.execution_providers)


def _analyse_faces(frame: Frame) -> list:
    """Run face detection, then recognition (and optionally landmark).

    Replaces InsightFace's ``FaceAnalysis.get()`` to skip the
    landmark_2d_106 model when only face_swapper is active (saves ~1ms
    per face and avoids an unnecessary ONNX session call).
    """
    fa = get_face_analyser()

    bboxes, kpss = fa.det_model.detect(frame, max_num=0, metric="default")
    if bboxes.shape[0] == 0:
        return []

    need_landmark = _needs_landmark()
    rec_model = fa.models.get("recognition")
    lmk_model = fa.models.get("landmark_2d_106") if need_landmark else None

    from insightface.app.common import Face

    faces = []
    for i in range(bboxes.shape[0]):
        face = Face(bbox=bboxes[i, 0:4],
                    kps=kpss[i] if kpss is not None else None,
                    det_score=bboxes[i, 4])
        if rec_model is not None:
            rec_model.get(frame, face)
        if lmk_model is not None:
            lmk_model.get(frame, face)
        faces.append(face)

    return faces


def get_one_face(frame: Frame) -> Any:
    if _is_dml():
        with modules.globals.dml_lock:
            faces = _analyse_faces(frame)
    else:
        faces = _analyse_faces(frame)
    try:
        return min(faces, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        if _is_dml():
            with modules.globals.dml_lock:
                return _analyse_faces(frame)
        else:
            return _analyse_faces(frame)
    except IndexError:
        return None

def detect_one_face_fast(frame: Frame) -> Any:
    """Detection-only — skips landmark and recognition models.

    Returns a Face with bbox, kps, det_score (enough for face swap).
    ~10ms vs ~16ms for full get_one_face() at 1080p.
    """
    from insightface.app.common import Face
    fa = get_face_analyser()
    bboxes, kpss = fa.det_model.detect(frame, max_num=0, metric='default')
    if bboxes.shape[0] == 0:
        return None
    idx = int(bboxes[:, 0].argmin())
    return Face(bbox=bboxes[idx, :4], kps=kpss[idx], det_score=bboxes[idx, 4])


def detect_many_faces_fast(frame: Frame) -> Any:
    """Detection-only multi-face — skips landmark and recognition."""
    from insightface.app.common import Face
    fa = get_face_analyser()
    bboxes, kpss = fa.det_model.detect(frame, max_num=0, metric='default')
    if bboxes.shape[0] == 0:
        return None
    return [Face(bbox=bboxes[i, :4], kps=kpss[i], det_score=bboxes[i, 4])
            for i in range(bboxes.shape[0])]


def has_valid_map() -> bool:
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            return True
    return False

def default_source_face() -> Any:
    for map in modules.globals.source_target_map:
        if "source" in map:
            return map['source']['face']
    return None

def simplify_maps() -> Any:
    centroids = []
    faces = []
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            centroids.append(map['target']['face'].normed_embedding)
            faces.append(map['source']['face'])

    modules.globals.simple_map = {'source_faces': faces, 'target_embeddings': centroids}
    return None

def add_blank_map() -> Any:
    try:
        max_id = -1
        if len(modules.globals.source_target_map) > 0:
            max_id = max(modules.globals.source_target_map, key=lambda x: x['id'])['id']

        modules.globals.source_target_map.append({
                'id' : max_id + 1
                })
    except ValueError:
        return None
    
def get_unique_faces_from_target_image() -> Any:
    try:
        modules.globals.source_target_map = []
        target_frame = cv2.imread(modules.globals.target_path)
        many_faces = get_many_faces(target_frame)
        i = 0

        for face in many_faces:
            x_min, y_min, x_max, y_max = face['bbox']
            modules.globals.source_target_map.append({
                'id' : i, 
                'target' : {
                            'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                            'face' : face
                            }
                })
            i = i + 1
    except ValueError:
        return None
    
    
def get_unique_faces_from_target_video() -> Any:
    try:
        modules.globals.source_target_map = []
        frame_face_embeddings = []
        face_embeddings = []
    
        print('Creating temp resources...')
        clean_temp(modules.globals.target_path)
        create_temp(modules.globals.target_path)
        print('Extracting frames...')
        extract_frames(modules.globals.target_path)

        temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)

        i = 0
        for temp_frame_path in tqdm(temp_frame_paths, desc="Extracting face embeddings from frames"):
            temp_frame = cv2.imread(temp_frame_path)
            many_faces = get_many_faces(temp_frame)

            for face in many_faces:
                face_embeddings.append(face.normed_embedding)
            
            frame_face_embeddings.append({'frame': i, 'faces': many_faces, 'location': temp_frame_path})
            i += 1

        centroids = find_cluster_centroids(face_embeddings)

        for frame in frame_face_embeddings:
            for face in frame['faces']:
                closest_centroid_index, _ = find_closest_centroid(centroids, face.normed_embedding)
                face['target_centroid'] = closest_centroid_index

        for i in range(len(centroids)):
            modules.globals.source_target_map.append({
                'id' : i
            })

            temp = []
            for frame in tqdm(frame_face_embeddings, desc=f"Mapping frame embeddings to centroids-{i}"):
                temp.append({'frame': frame['frame'], 'faces': [face for face in frame['faces'] if face['target_centroid'] == i], 'location': frame['location']})

            modules.globals.source_target_map[i]['target_faces_in_frame'] = temp

        # dump_faces(centroids, frame_face_embeddings)
        default_target_face()
    except ValueError:
        return None
    

def default_target_face():
    for map in modules.globals.source_target_map:
        best_face = None
        best_frame = None
        for frame in map['target_faces_in_frame']:
            if len(frame['faces']) > 0:
                best_face = frame['faces'][0]
                best_frame = frame
                break

        for frame in map['target_faces_in_frame']:
            for face in frame['faces']:
                if face['det_score'] > best_face['det_score']:
                    best_face = face
                    best_frame = frame

        x_min, y_min, x_max, y_max = best_face['bbox']

        target_frame = cv2.imread(best_frame['location'])
        map['target'] = {
                        'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                        'face' : best_face
                        }


def dump_faces(centroids: Any, frame_face_embeddings: list):
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)

    for i in range(len(centroids)):
        if os.path.exists(temp_directory_path + f"/{i}") and os.path.isdir(temp_directory_path + f"/{i}"):
            shutil.rmtree(temp_directory_path + f"/{i}")
        Path(temp_directory_path + f"/{i}").mkdir(parents=True, exist_ok=True)

        for frame in tqdm(frame_face_embeddings, desc=f"Copying faces to temp/./{i}"):
            temp_frame = cv2.imread(frame['location'])

            j = 0
            for face in frame['faces']:
                if face['target_centroid'] == i:
                    x_min, y_min, x_max, y_max = face['bbox']

                    if temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)].size > 0:
                        cv2.imwrite(temp_directory_path + f"/{i}/{frame['frame']}_{j}.png", temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)])
                j += 1
