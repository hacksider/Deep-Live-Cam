import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.websockets import WebSocketDisconnect

from webapp.engine import LiveEngine

COOKIE_NAME = "dlc_session"
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
_WAIT_SECONDS = 5.0


def session_token(password: str) -> str:
    return hmac.new(password.encode("utf-8"), b"dlc-web-v1", hashlib.sha256).hexdigest()


def _cookie_ok(cookies, password: str) -> bool:
    got = cookies.get(COOKIE_NAME)
    if not got:
        return False
    return hmac.compare_digest(got, session_token(password))


def _password_ok(given: str, password: str) -> bool:
    return hmac.compare_digest(
        hashlib.sha256(given.encode("utf-8")).digest(),
        hashlib.sha256(password.encode("utf-8")).digest(),
    )


def create_app(engine: LiveEngine, password: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        stop = threading.Event()

        def loop() -> None:
            while not stop.is_set():
                engine.drain()
                stop.wait(0.01)

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
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "wrong password"}, status_code=401)
        given = body.get("password") if isinstance(body, dict) else None
        if not isinstance(given, str) or not _password_ok(given, password):
            return JSONResponse({"error": "wrong password"}, status_code=401)
        response = JSONResponse({"ok": True})
        response.set_cookie(
            COOKIE_NAME,
            session_token(password),
            httponly=True,
            samesite="lax",
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
    loops = int(_WAIT_SECONDS / 0.01)
    for _ in range(loops):
        message = engine.source_message(session_id)
        if message is not None:
            return message
        await asyncio.sleep(0.01)
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
    loops = int(_WAIT_SECONDS / 0.01)
    for _ in range(loops):
        result = engine.frame_result(session_id, seq)
        if result is not None:
            return result
        await asyncio.sleep(0.01)
    return None
