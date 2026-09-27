"""Frame-by-frame face swapping for remote live (webcam) sessions.

The desktop app's live mode reads the webcam itself. On a cloud GPU the
webcam is on the user's machine, so frames arrive over the network instead
(see ``cloud/server.py``'s ``/live`` WebSocket and ``cloud/live_client.py``).
This module is the display-free equivalent of the desktop live loop in
``modules/ui.py`` (``_ProcessingWorker.run``): same fast detection, same
per-processor calls, same post-processing.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import cv2
import numpy as np

from modules.typing import Frame

ENHANCER_NAMES = {
    "DLC.FACE-ENHANCER": "face_enhancer",
    "DLC.FACE-ENHANCER-GPEN256": "face_enhancer_gpen256",
    "DLC.FACE-ENHANCER-GPEN512": "face_enhancer_gpen512",
}


def decode_image(data: bytes) -> Frame:
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image data")
    return frame


def encode_jpeg(frame: Frame, quality: int = 85) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("Could not encode frame")
    return buffer.tobytes()


class LiveSession:
    """Holds one live session's source face and detection cache.

    Deep-Live-Cam's settings are process-wide globals, so only one session
    (or file job) may be active per process; ``cloud/server.py`` enforces it.
    """

    def __init__(self) -> None:
        self.source_face: Any = None
        self.processors: list = []
        self.detect_every = 1
        self._frame_index = 0
        self._cached_target_face: Any = None
        self._cached_many_faces: Any = None

    def configure(
        self,
        source_image: Optional[bytes] = None,
        *,
        frame_processors: Iterable[str] = ("face_swapper",),
        many_faces: bool = False,
        mouth_mask: bool = False,
        det_size: Optional[int] = None,
        detect_every: int = 1,
        opacity: float = 1.0,
        execution_provider: Optional[str] = None,
    ) -> None:
        """(Re)configure the session. ``source_image`` is encoded image bytes;
        omit it on later calls to keep the current source face."""
        import modules.globals
        from modules.face_analyser import get_one_face
        from modules.headless import apply_settings

        if source_image is None and self.source_face is None:
            raise ValueError("A source face image is required to start a session")
        if not 1 <= int(detect_every) <= 30:
            raise ValueError("detect_every must be between 1 and 30")
        if not 0.0 <= float(opacity) <= 1.0:
            raise ValueError("opacity must be between 0 and 1")

        self.processors = apply_settings(
            frame_processors,
            many_faces=many_faces,
            mouth_mask=mouth_mask,
            det_size=det_size,
            execution_provider=execution_provider,
            opacity=float(opacity),
        )
        modules.globals.source_path = None
        modules.globals.target_path = None
        self.detect_every = int(detect_every)

        for processor in self.processors:
            # Load the swapper model now rather than on the first frame.
            # (The enhancers' pre_start insists on a target file, so they
            # load lazily on their first frame instead.)
            if processor.NAME == "DLC.FACE-SWAPPER" and not processor.pre_start():
                raise RuntimeError("Could not load the face swapper model")

        if source_image is not None:
            face = get_one_face(decode_image(source_image))
            if face is None:
                raise ValueError("No face found in the source image")
            self.source_face = face

        self._frame_index = 0
        self._cached_target_face = None
        self._cached_many_faces = None

    def process(self, frame: Frame) -> Frame:
        """Swap (and optionally enhance) one BGR frame. Mirrors ui.py's live loop."""
        import modules.globals
        from modules.face_analyser import (
            detect_many_faces_fast,
            detect_one_face_fast,
            ensure_landmarks,
        )

        if self._frame_index % self.detect_every == 0:
            if modules.globals.many_faces:
                self._cached_target_face = None
                self._cached_many_faces = detect_many_faces_fast(frame)
            else:
                self._cached_target_face = detect_one_face_fast(frame)
                self._cached_many_faces = None
        self._frame_index += 1

        cached_faces = None
        if self._cached_many_faces:
            cached_faces = self._cached_many_faces
        elif self._cached_target_face is not None:
            cached_faces = [self._cached_target_face]

        # Fast detection skips the 2d106 landmark model; the mouth mask needs it.
        if modules.globals.mouth_mask and cached_faces:
            ensure_landmarks(frame, cached_faces)

        for fp in self.processors:
            if fp.NAME in ENHANCER_NAMES:
                if modules.globals.fp_ui.get(ENHANCER_NAMES[fp.NAME], False):
                    frame = fp.process_frame(None, frame, detected_faces=cached_faces)
            elif fp.NAME == "DLC.FACE-SWAPPER":
                swapped_bboxes = []
                if modules.globals.many_faces and self._cached_many_faces:
                    result = frame.copy()
                    for target_face in self._cached_many_faces:
                        result = fp.swap_face(self.source_face, target_face, result)
                        if getattr(target_face, "bbox", None) is not None:
                            swapped_bboxes.append(target_face.bbox.astype(int))
                    frame = result
                elif self._cached_target_face is not None:
                    frame = fp.swap_face(self.source_face, self._cached_target_face, frame)
                    if getattr(self._cached_target_face, "bbox", None) is not None:
                        swapped_bboxes.append(self._cached_target_face.bbox.astype(int))
                frame = fp.apply_post_processing(frame, swapped_bboxes)
            else:
                frame = fp.process_frame(self.source_face, frame)
        return frame
