import os
import shutil
from typing import Any, List
import insightface

import cv2
import numpy as np
import modules.globals
from tqdm import tqdm
from modules.typing import Frame
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import get_temp_directory_path, create_temp, extract_frames, clean_temp, get_temp_frame_paths
from pathlib import Path

FACE_ANALYSER = None

# Gender filter options: "male", "female", or None (no filtering)
GENDER_FILTER = "female"  # set to None to disable filtering

def get_face_analyser() -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(
            name='buffalo_l',
            providers=modules.globals.execution_providers
        )
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER

def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

def get_many_faces(frame: Frame) -> List[Any]:
    """
    Original behaviour: return raw list of detected faces (may be empty/list).
    Do NOT apply gender filtering here — keep official behaviour so other code
    that expects all faces continues to work.
    """
    try:
        faces = get_face_analyser().get(frame)
        print(f"[DEBUG] faces detected: {len(faces) if faces else 0}")
        return faces if faces is not None else []
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
    """
    This function builds source_target_map for a single image target.
    We detect all faces (official behaviour) then apply gender filter HERE
    so source face logic elsewhere is unaffected.
    """
    try:
        modules.globals.source_target_map = []
        target_frame = cv2.imread(modules.globals.target_path)
        many_faces = get_many_faces(target_frame)  # raw faces
        if not many_faces:
            return None

        # apply gender filter only when building the map
        many_faces = filter_faces_by_gender(many_faces)

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
            i += 1
    except ValueError:
        return None


def get_unique_faces_from_target_video() -> Any:
    """
    Video flow: keep detection behaviour the same, but apply gender filter
    when compiling `face_embeddings` and frames into the mapping.
    """
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
            many_faces = get_many_faces(temp_frame)  # raw faces

            # apply gender filter to faces detected in this frame
            many_faces_filtered = filter_faces_by_gender(many_faces)

            for face in many_faces_filtered:
                face_embeddings.append(face.normed_embedding)

            frame_face_embeddings.append({'frame': i, 'faces': many_faces_filtered, 'location': temp_frame_path})
            i += 1

        # If no embeddings after filtering, abort gracefully
        if not face_embeddings:
            print("[DEBUG] No face embeddings found after gender filtering; aborting mapping.")
            return None

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
