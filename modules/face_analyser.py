import os
import shutil
from typing import Any, List
import cv2
import insightface
import modules
from tqdm import tqdm
from modules.typing import Frame
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import get_temp_directory_path, create_temp, extract_frames, clean_temp, get_temp_frame_paths
from pathlib import Path

FACE_ANALYSER = None


def get_face_analyser() -> Any:
    """Thread-safe singleton loader for the face analyser model."""
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=modules.globals.execution_providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_one_face(frame: Any) -> Any:
    """Get the most prominent face from a frame."""
    try:
        face = get_face_analyser().get(frame)
        return min(face, key=lambda x: x.bbox[0]) if face else None
    except Exception as e:
        print(f"Error in get_one_face: {e}")
        return None


def get_many_faces(frame: Any) -> Any:
    """Get all faces from a frame."""
    try:
        return get_face_analyser().get(frame)
    except Exception as e:
        print(f"Error in get_many_faces: {e}")
        return None


def has_valid_map() -> bool:
    """Check if the global source_target_map has valid mappings."""
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            return True
    return False


def default_source_face() -> Any:
    """Return the first source face from the global map, if available."""
    for map in modules.globals.source_target_map:
        if "source" in map:
            return map["source"]["face"]
    return None


def simplify_maps() -> None:
    """Simplify the global source_target_map into centroids and faces for fast lookup."""
    centroids = []
    faces = []
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            faces.append(map["source"]["face"])
            centroids.append(map["target"]["face"].normed_embedding)
    modules.globals.simple_map = {'source_faces': faces, 'target_embeddings': centroids}
    return None


def add_blank_map() -> None:
    """Add a blank map entry to the global source_target_map."""
    try:
        max_id = -1
        if len(modules.globals.source_target_map) > 0:
            max_id = max(map['id'] for map in modules.globals.source_target_map if 'id' in map)
        modules.globals.source_target_map.append({'id': max_id + 1})
    except Exception as e:
        print(f"Error in add_blank_map: {e}")
        return None


def get_unique_faces_from_target_image() -> Any:
    """Extract unique faces from the target image and update the global map."""
    try:
        modules.globals.source_target_map = []
        target_frame = cv2.imread(modules.globals.target_path)
        many_faces = get_many_faces(target_frame)
        i = 0
        for face in many_faces:
            modules.globals.source_target_map.append({
                'id': i,
                'target': {'face': face}
            })
            i += 1
    except Exception as e:
        print(f"Error in get_unique_faces_from_target_image: {e}")
        return None


def get_unique_faces_from_target_video() -> Any:
    """Extract unique faces from all frames of the target video and update the global map."""
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
            frame = cv2.imread(temp_frame_path)
            faces = get_many_faces(frame)
            if faces:
                for face in faces:
                    face_embeddings.append(face.normed_embedding)
                    frame_face_embeddings.append({'frame': temp_frame_path, 'face': face})
        centroids = find_cluster_centroids(face_embeddings)
        for frame in frame_face_embeddings:
            closest_centroid_index, _ = find_closest_centroid(centroids, frame['face'].normed_embedding)
            modules.globals.source_target_map.append({
                'id': closest_centroid_index,
                'target': {'face': frame['face'], 'location': frame['frame']}
            })
        for i in range(len(centroids)):
            pass  # Optionally, add more logic here
    except Exception as e:
        print(f"Error in get_unique_faces_from_target_video: {e}")
        return None


def default_target_face():
    """Return the first target face from the global map, if available."""
    for map in modules.globals.source_target_map:
        if "target" in map:
            return map["target"]["face"]
    return None


def dump_faces(centroids: Any, frame_face_embeddings: list) -> None:
    """Dump face crops to the temp directory for debugging or visualization."""
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)
    for i in range(len(centroids)):
        pass  # Implement as needed