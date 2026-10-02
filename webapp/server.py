import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import logging
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.websockets import WebSocketDisconnect

from webapp.engine import LiveEngine

COOKIE_NAME = "dlc_session"
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
_WAIT_SECONDS = 5.0
_POLL_SECONDS = 0.002
LOGIN_FAIL_LIMIT = 30
LOGIN_FAIL_WINDOW = 60.0
_LOGIN_TRACKED_MAX = 10_000

logger = logging.getLogger(__name__)


def _b(text: str) -> bytes:
    # surrogatepass: never raises, even for lone surrogates in odd input.
    return text.encode("utf-8", errors="surrogatepass")


def session_token(password: str) -> str:
    return hmac.new(_b(password), b"dlc-web-v1", hashlib.sha256).hexdigest()


def _cookie_ok(cookies, password: str) -> bool:
    got = cookies.get(COOKIE_NAME)
    if not got:
        return False
    return hmac.compare_digest(_b(got), _b(session_token(password)))


def _password_ok(given: str, password: str) -> bool:
    return hmac.compare_digest(
        hashlib.sha256(_b(given)).digest(),
        hashlib.sha256(_b(password)).digest(),
    )


def login_client_key(peer: str | None, forwarded_for: str | None) -> str:
    fallback = peer if peer else "unknown"
    if peer in ("127.0.0.1", "::1") and forwarded_for:
        # Behind one trusted proxy the last hop is the address the proxy saw;
        # earlier entries are visitor-supplied and must not be trusted.
        last = forwarded_for.rsplit(",", 1)[-1].strip()
        if last:
            return last
        return fallback
    return fallback


class _LoginLimiter:
    def __init__(self, limit: int, window: float) -> None:
        self._limit = limit
        self._window = window
        self._failures: dict[str, deque[float]] = {}

    def _prune(self, key: str, now: float) -> deque[float]:
        stamps = self._failures.get(key)
        if stamps is None:
            return deque()
        while stamps and now - stamps[0] >= self._window:
            stamps.popleft()
        if not stamps:
            del self._failures[key]
        return stamps

    def blocked(self, key: str) -> bool:
        return len(self._prune(key, time.monotonic())) >= self._limit

    def record_failure(self, key: str) -> None:
        now = time.monotonic()
        if key not in self._failures and len(self._failures) >= _LOGIN_TRACKED_MAX:
            for other in list(self._failures):
                self._prune(other, now)
            if len(self._failures) >= _LOGIN_TRACKED_MAX:
                return
        self._failures.setdefault(key, deque()).append(now)


def create_app(engine: LiveEngine, password: str, secure_cookie: bool = False) -> FastAPI:
    limiter = _LoginLimiter(LOGIN_FAIL_LIMIT, LOGIN_FAIL_WINDOW)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        stop = threading.Event()

        def loop() -> None:
            while not stop.is_set():
                try:
                    engine.drain()
                except Exception:
                    logger.exception("gpu-worker pass failed")
                stop.wait(_POLL_SECONDS)

        thread = threading.Thread(target=loop, name="gpu-worker", daemon=True)
        thread.start()
        yield
        stop.set()
        thread.join(timeout=2)

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    def index():
        if not STATIC_INDEX.is_file():
            return Response("studio page is not built yet", status_code=503)
        return FileResponse(STATIC_INDEX)

    @app.post("/login")
    async def login(request: Request):
        peer = request.client.host if request.client else None
        forwarded_for = request.headers.get("x-forwarded-for")
        client = login_client_key(peer, forwarded_for)
        if limiter.blocked(client):
            logger.warning("login rate limit hit for %s", client)
            return JSONResponse({"error": "too many attempts"}, status_code=429)
        try:
            body = await request.json()
        except Exception:
            body = None
        given = body.get("password") if isinstance(body, dict) else None
        if not isinstance(given, str) or not _password_ok(given, password):
            limiter.record_failure(client)
            logger.warning("wrong password from %s", client)
            return JSONResponse({"error": "wrong password"}, status_code=401)
        response = JSONResponse({"ok": True})
        response.set_cookie(
            COOKIE_NAME,
            session_token(password),
            httponly=True,
            samesite="lax",
            secure=secure_cookie,
            path="/",
        )
        return response

    @app.get("/session")
    def session(request: Request):
        if not _cookie_ok(request.cookies, password):
            return Response(status_code=401)
        return Response(status_code=204)

    @app.websocket("/live")
    async def live(websocket: WebSocket):
        await websocket.accept()
        if not _cookie_ok(websocket.cookies, password):
            await websocket.close(code=4401)
            return
        session_id = uuid.uuid4().hex
        if not engine.open_session(session_id):
            await websocket.send_json({"type": "error", "message": "server is full"})
            await websocket.close()
            return
        try:
            while True:
                incoming = await websocket.receive()
                if incoming.get("type") == "websocket.disconnect":
                    break
                text = incoming.get("text")
                data = incoming.get("bytes")
                if text is not None:
                    await _handle_text(websocket, engine, session_id, text)
                elif data is not None:
                    await _handle_frame(websocket, engine, session_id, data)
        except WebSocketDisconnect:
            pass
        finally:
            engine.close_session(session_id)

    return app


async def _handle_text(websocket: WebSocket, engine: LiveEngine, session_id: str, text: str) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        await websocket.send_json({"type": "error", "message": "unknown message"})
        return
    if not isinstance(payload, dict) or payload.get("type") != "source":
        await websocket.send_json({"type": "error", "message": "unknown message"})
        return
    image = payload.get("image")
    if not isinstance(image, str):
        await websocket.send_json({"type": "error", "message": "unreadable image"})
        return
    try:
        raw = base64.b64decode(image, validate=True)
    except (binascii.Error, ValueError):
        await websocket.send_json({"type": "error", "message": "unreadable image"})
        return
    engine.submit_source(session_id, raw)
    message = await _wait_source(engine, session_id)
    if message is None:
        await websocket.send_json({"type": "error", "message": "unreadable image"})
        return
    if message == "ready":
        await websocket.send_json({"type": "ready"})
        return
    await websocket.send_json({"type": "error", "message": message})


async def _wait_source(engine: LiveEngine, session_id: str) -> str | None:
    loops = int(_WAIT_SECONDS / _POLL_SECONDS)
    for _ in range(loops):
        message = engine.source_message(session_id)
        if message is not None:
            return message
        await asyncio.sleep(_POLL_SECONDS)
    return None


async def _handle_frame(websocket: WebSocket, engine: LiveEngine, session_id: str, data: bytes) -> None:
    seq, rejected = engine.submit_frame(session_id, data)
    if rejected:
        await websocket.send_json({"type": "error", "message": rejected})
        return
    result = await _wait_frame(engine, session_id, seq)
    if result is None:
        await websocket.send_json({"type": "error", "message": "bad frame"})
        return
    kind, payload = result
    if kind == "jpeg":
        await websocket.send_bytes(payload)
        return
    await websocket.send_json({"type": "error", "message": payload})


async def _wait_frame(engine: LiveEngine, session_id: str, seq: int):
    loops = int(_WAIT_SECONDS / _POLL_SECONDS)
    for _ in range(loops):
        result = engine.frame_result(session_id, seq)
        if result is not None:
            return result
        await asyncio.sleep(_POLL_SECONDS)
    return None
