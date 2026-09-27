#!/usr/bin/env python3
"""Live webcam face swap on a remote (cloud) GPU.

Runs on YOUR computer: captures the webcam, streams frames to a Deep-Live-Cam
server's /live WebSocket (cloud/server.py, on Modal, RunPod, Docker ...),
and shows the swapped video. Add --virtual-cam to feed it to Zoom, Discord,
OBS, Meet ... as a regular camera.

    pip install -r requirements-live-client.txt
    python cloud/live_client.py --server wss://<your-endpoint> --source face.jpg
    python cloud/live_client.py --server ws://1.2.3.4:8000 --source face.jpg --virtual-cam

Keys in the preview window: q / Esc quit, m mirror, e cycle face enhancer,
f toggle many faces.

Virtual camera backends (used by pyvirtualcam): OBS Studio's virtual camera
on Windows/macOS (install OBS and start its virtual camera once), or
v4l2loopback on Linux (`sudo modprobe v4l2loopback exclusive_caps=1`).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import struct
import sys
import threading
import time
import urllib.parse
from collections import deque
from typing import Optional

import cv2
import numpy as np

try:
    import websockets
except ImportError:  # pragma: no cover - guidance for users
    sys.exit("The live client needs the 'websockets' package: pip install -r requirements-live-client.txt")

FRAME_HEADER = struct.Struct(">I")
ENHANCERS = ["none", "face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"]
FATAL_CLOSE_CODES = {1003, 1008}  # bad request / bad token: reconnecting won't help
STALE_FRAME_SECONDS = 5.0
WINDOW = "Deep-Live-Cam (cloud)"


def live_url(server: str, token: Optional[str]) -> str:
    """Accept ws://, wss://, http:// or https:// with or without the /live path."""
    parsed = urllib.parse.urlsplit(server if "://" in server else f"ws://{server}")
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    if scheme not in ("ws", "wss"):
        raise ValueError(f"Unsupported server URL scheme: {parsed.scheme}")
    path = parsed.path.rstrip("/")
    if not path.endswith("/live"):
        path += "/live"
    query = urllib.parse.parse_qs(parsed.query)
    if token:
        query["token"] = [token]
    return urllib.parse.urlunsplit((scheme, parsed.netloc, path, urllib.parse.urlencode(query, doseq=True), ""))


class Camera(threading.Thread):
    """Reads the webcam continuously and keeps only the newest frame."""

    def __init__(self, device: str, width: int, height: int, fps: int) -> None:
        super().__init__(daemon=True)
        self.capture = cv2.VideoCapture(int(device) if device.isdigit() else device)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera {device!r}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.is_file = not device.isdigit() and os.path.isfile(device)
        self.fps = fps
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.index = 0
        self.stopped = threading.Event()

    def run(self) -> None:
        while not self.stopped.is_set():
            ok, frame = self.capture.read()
            if not ok:
                if self.is_file:  # loop test videos
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.index += 1
            if self.is_file:
                time.sleep(1.0 / self.fps)
        self.capture.release()

    def latest(self) -> tuple[int, Optional[np.ndarray]]:
        with self.lock:
            return self.index, self.frame


class State:
    """Shared between the network thread and the display (main) thread."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.output: Optional[np.ndarray] = None
        self.output_index = 0
        self.status = "Connecting (a cold GPU can take up to a minute)..."
        self.latencies: deque = deque(maxlen=30)
        self.received: deque = deque(maxlen=60)
        self.stop = threading.Event()
        self.mirror = args.mirror
        self.enhancer = args.enhancer
        self.many_faces = args.many_faces
        self.options_dirty = False

    def options(self) -> dict:
        processors = ["face_swapper"]
        if self.enhancer != "none":
            processors.append(self.enhancer)
        return {
            "frame_processors": processors,
            "many_faces": self.many_faces,
            "mouth_mask": self.args.mouth_mask,
            "det_size": self.args.det_size,
            "detect_every": self.args.detect_every,
            "opacity": self.args.opacity,
            "jpeg_quality": self.args.jpeg_quality,
        }

    def set_status(self, text: str) -> None:
        with self.lock:
            self.status = text
        print(f"[live] {text}", flush=True)

    def stats(self) -> str:
        with self.lock:
            now = time.time()
            recent = [t for t in self.received if now - t < 2.0]
            fps = len(recent) / 2.0
            latency = sum(self.latencies) / len(self.latencies) * 1000 if self.latencies else 0
        return f"{fps:4.1f} fps  {latency:4.0f} ms round trip"


async def stream(state: State, camera: Camera, url: str, source_b64: str) -> None:
    args = state.args
    backoff = 1.0
    while not state.stop.is_set():
        try:
            async with websockets.connect(
                url, max_size=32 * 1024 * 1024, open_timeout=180, ping_interval=20, ping_timeout=60
            ) as ws:
                await ws.send(json.dumps({"type": "start", "source": source_b64, "options": state.options()}))
                reply = json.loads(await ws.recv())
                if reply.get("type") != "ready":
                    state.set_status(f"Server refused the session: {reply.get('message', reply)}")
                    state.stop.set()
                    return
                state.set_status(f"Streaming. Server providers: {', '.join(reply.get('providers', []))}")
                backoff = 1.0
                await run_session(state, camera, ws)
        except websockets.ConnectionClosed as closed:
            code = closed.rcvd.code if closed.rcvd else None
            reason = f": {closed.rcvd.reason}" if closed.rcvd and closed.rcvd.reason else ""
            if code in FATAL_CLOSE_CODES:
                state.set_status(f"Server closed the session ({code}){reason}")
                state.stop.set()
                return
            if state.stop.is_set():
                return
            state.set_status(f"Connection closed ({code}){reason}; reconnecting in {backoff:.0f}s...")
        except (OSError, asyncio.TimeoutError, websockets.InvalidURI, websockets.InvalidHandshake) as error:
            state.set_status(f"Cannot reach server ({error}); retrying in {backoff:.0f}s...")
        if state.stop.is_set():
            return
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 30.0)


async def run_session(state: State, camera: Camera, ws) -> None:
    args = state.args
    sent_at: dict[int, float] = {}
    next_id = 0
    last_camera_index = 0

    async def sender() -> None:
        nonlocal next_id, last_camera_index
        while not state.stop.is_set():
            now = time.time()
            for frame_id in [i for i, t in sent_at.items() if now - t > STALE_FRAME_SECONDS]:
                sent_at.pop(frame_id, None)  # lost/failed frame: stop waiting for it
            if state.options_dirty:
                state.options_dirty = False
                await ws.send(json.dumps({"type": "update", "options": state.options()}))
            index, frame = camera.latest()
            if frame is None or index == last_camera_index or len(sent_at) >= args.max_in_flight:
                await asyncio.sleep(0.002)
                continue
            last_camera_index = index
            if state.mirror:
                frame = cv2.flip(frame, 1)
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if not ok:
                continue
            frame_id = next_id
            next_id = (next_id + 1) % 2**32
            sent_at[frame_id] = time.time()
            await ws.send(FRAME_HEADER.pack(frame_id) + jpeg.tobytes())

    async def receiver() -> None:
        async for message in ws:
            if isinstance(message, bytes):
                frame_id = FRAME_HEADER.unpack(message[:FRAME_HEADER.size])[0]
                started = sent_at.pop(frame_id, None)
                frame = cv2.imdecode(np.frombuffer(message[FRAME_HEADER.size:], np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                now = time.time()
                with state.lock:
                    state.output = frame
                    state.output_index += 1
                    state.received.append(now)
                    if started is not None:
                        state.latencies.append(now - started)
                continue
            data = json.loads(message)
            if data.get("type") == "error":
                if "frame_id" in data:
                    sent_at.pop(data["frame_id"], None)
                state.set_status(f"Server: {data.get('message')}")
            elif data.get("type") == "updated":
                state.set_status("Settings applied.")

    tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()  # re-raise ConnectionClosed etc.
    finally:
        for task in tasks:
            task.cancel()


def draw_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    frame = frame.copy()
    for i, line in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def open_virtual_camera(args: argparse.Namespace):
    try:
        import pyvirtualcam
    except ImportError:
        sys.exit("--virtual-cam needs pyvirtualcam: pip install pyvirtualcam")
    try:
        cam = pyvirtualcam.Camera(
            width=args.width, height=args.height, fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR, device=args.virtual_cam_device,
        )
    except RuntimeError as error:
        sys.exit(f"Could not start the virtual camera: {error}\n"
                 "Windows/macOS: install OBS Studio and start its virtual camera once. "
                 "Linux: sudo modprobe v4l2loopback exclusive_caps=1")
    print(f"[live] Virtual camera: {cam.device} ({args.width}x{args.height} @ {args.fps} fps)")
    return cam


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True, help="server URL, e.g. wss://…modal.run or ws://1.2.3.4:8000")
    parser.add_argument("--source", required=True, help="image of the face to use")
    parser.add_argument("--token", default=os.environ.get("DLC_API_TOKEN"), help="API token (default: $DLC_API_TOKEN)")
    parser.add_argument("--camera", default="0", help="camera index, or a video file/stream URL for testing")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mirror", action="store_true", help="mirror the image like a selfie camera")
    parser.add_argument("--enhancer", choices=ENHANCERS, default="none")
    parser.add_argument("--many-faces", action="store_true")
    parser.add_argument("--mouth-mask", action="store_true")
    parser.add_argument("--opacity", type=float, default=1.0)
    parser.add_argument("--det-size", type=int, choices=[160, 320, 640], default=640)
    parser.add_argument("--detect-every", type=int, default=1,
                        help="run face detection every N frames (higher = faster, laggier tracking)")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="30-100; lower uses less bandwidth")
    parser.add_argument("--max-in-flight", type=int, default=2,
                        help="frames sent but not yet returned; higher = smoother on slow links, more delay")
    parser.add_argument("--virtual-cam", action="store_true", help="publish the output as a virtual webcam")
    parser.add_argument("--virtual-cam-device", default=None, help="e.g. /dev/video10 (Linux)")
    parser.add_argument("--no-preview", action="store_true", help="don't open a preview window")
    args = parser.parse_args(argv)
    if args.no_preview and not args.virtual_cam:
        parser.error("--no-preview needs --virtual-cam, otherwise there is nothing to show")
    if not 30 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 30 and 100")
    if not 1 <= args.max_in_flight <= 8:
        parser.error("--max-in-flight must be between 1 and 8")
    if args.width * args.height > 1920 * 1080:
        parser.error("the server accepts frames up to 1920x1080")
    return args


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    with open(args.source, "rb") as handle:
        source_b64 = base64.b64encode(handle.read()).decode("ascii")
    url = live_url(args.server, args.token)

    camera = Camera(args.camera, args.width, args.height, args.fps)
    camera.start()
    state = State(args)

    network = threading.Thread(
        target=lambda: asyncio.run(stream(state, camera, url, source_b64)), daemon=True
    )
    network.start()

    vcam = open_virtual_camera(args) if args.virtual_cam else None
    blank = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    last_report = time.time()
    try:
        while not state.stop.is_set():
            if time.time() - last_report >= 5 and state.output is not None:
                last_report = time.time()
                print(f"[live] {state.stats()}", flush=True)
            with state.lock:
                output = state.output
                status = state.status
            if output is not None and output.shape[:2] != (args.height, args.width):
                output = cv2.resize(output, (args.width, args.height))

            if vcam is not None:
                # Until the first swapped frame arrives, publish black rather
                # than the real (unswapped) camera.
                vcam.send(output if output is not None else blank)

            if not args.no_preview:
                if output is None:
                    _, raw = camera.latest()
                    shown = cv2.resize(raw, (args.width, args.height)) if raw is not None else blank
                    shown = draw_overlay(shown, ["NOT SWAPPED (waiting for server)", status])
                else:
                    shown = draw_overlay(output, [state.stats(), f"enhancer: {state.enhancer}  [q]uit [m]irror [e]nhancer [f]aces"])
                cv2.imshow(WINDOW, shown)
                key = cv2.waitKey(5) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("m"):
                    state.mirror = not state.mirror
                elif key == ord("e"):
                    state.enhancer = ENHANCERS[(ENHANCERS.index(state.enhancer) + 1) % len(ENHANCERS)]
                    state.options_dirty = True
                    state.set_status(f"Switching enhancer to {state.enhancer}...")
                elif key == ord("f"):
                    state.many_faces = not state.many_faces
                    state.options_dirty = True
                    state.set_status(f"many faces: {state.many_faces}")
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break

            if vcam is not None:
                vcam.sleep_until_next_frame()
            elif args.no_preview:
                time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop.set()
        camera.stopped.set()
        if vcam is not None:
            vcam.close()
        if not args.no_preview:
            cv2.destroyAllWindows()
        network.join(timeout=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
