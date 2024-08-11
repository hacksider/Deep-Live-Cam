from typing import Any, Optional
import insightface

import modules.globals
from modules.typing import Frame

FACE_ANALYSER: Optional[insightface.app.FaceAnalysis] = None

def get_face_analyser() -> insightface.app.FaceAnalysis:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(
            name='buffalo_l', 
            providers=modules.globals.execution_providers
        )
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    
    return FACE_ANALYSER

def get_one_face(frame: Frame) -> Optional[Any]:
    faces = get_face_analyser().get(frame)
    return min(faces, key=lambda x: x.bbox[0], default=None)

def get_many_faces(frame: Frame) -> Optional[Any]:
    faces = get_face_analyser().get(frame)
    return faces if faces else None
