import logging
import threading

MAX_SESSIONS = 8
MAX_JPEG_BYTES = 1_000_000
MAX_SOURCE_BYTES = 8_000_000

logger = logging.getLogger(__name__)


class _Session:
    def __init__(self) -> None:
        self.source_face = None
        self.pending_source: bytes | None = None
        self.source_message: str | None = None
        self.latest_jpeg: bytes | None = None
        self.latest_seq = 0
        self.swapped_seq = 0
        self.outgoing: tuple[str, bytes | str] | None = None
        self.outgoing_seq = 0


class LiveEngine:
    def __init__(self, get_one_face, detect_one_face, swap_face, decode_image, encode_jpeg):
        self._get_one_face = get_one_face
        self._detect_one_face = detect_one_face
        self._swap_face = swap_face
        self._decode_image = decode_image
        self._encode_jpeg = encode_jpeg
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def open_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                return True
            if len(self._sessions) >= MAX_SESSIONS:
                return False
            self._sessions[session_id] = _Session()
            return True

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def submit_source(self, session_id: str, image_bytes: bytes) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.pending_source = image_bytes
            session.source_message = None
            session.source_face = None

    def source_message(self, session_id: str) -> str | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.source_message

    def submit_frame(self, session_id: str, jpeg: bytes) -> tuple[int, str | None]:
        if len(jpeg) > MAX_JPEG_BYTES:
            return 0, "bad frame"
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return 0, "bad frame"
            session.latest_seq += 1
            session.latest_jpeg = jpeg
            return session.latest_seq, None

    def frame_result(self, session_id: str, seq: int) -> tuple[str, bytes | str] | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.outgoing is None or session.outgoing_seq < seq:
                return None
            return session.outgoing

    def drain(self) -> None:
        with self._lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self._drain_session(session_id)

    def _drain_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            pending = session.pending_source
            session.pending_source = None
            jpeg = None
            seq = 0
            if session.latest_jpeg is not None and session.latest_seq > session.swapped_seq:
                jpeg = session.latest_jpeg
                seq = session.latest_seq
                session.swapped_seq = seq
            source_face = session.source_face
        if pending is not None:
            source_face = self._apply_source(session_id, pending)
        if jpeg is not None:
            self._apply_frame(session_id, jpeg, seq, source_face)

    def _apply_source(self, session_id: str, image_bytes: bytes):
        if len(image_bytes) > MAX_SOURCE_BYTES:
            self._set_source(session_id, None, "unreadable image")
            return None
        image = self._decode_image(image_bytes)
        if image is None:
            self._set_source(session_id, None, "unreadable image")
            return None
        face = self._get_one_face(image)
        if face is None:
            self._set_source(session_id, None, "no face found")
            return None
        self._set_source(session_id, face, "ready")
        return face

    def _set_source(self, session_id: str, face, message: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.source_face = face
            session.source_message = message

    def _apply_frame(self, session_id: str, jpeg: bytes, seq: int, source_face) -> None:
        if source_face is None:
            self._publish(session_id, seq, ("error", "no face found"))
            return
        frame = self._decode_image(jpeg)
        if frame is None:
            self._publish(session_id, seq, ("error", "bad frame"))
            return
        try:
            target = self._detect_one_face(frame)
            if target is None:
                self._publish(session_id, seq, ("jpeg", jpeg))
                return
            swapped = self._swap_face(source_face, target, frame)
            self._publish(session_id, seq, ("jpeg", self._encode_jpeg(swapped)))
        except Exception:
            logger.exception("face swap failed for session %s", session_id)
            self._publish(session_id, seq, ("jpeg", jpeg))

    def _publish(self, session_id: str, seq: int, outgoing: tuple[str, bytes | str]) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.outgoing = outgoing
            session.outgoing_seq = seq
