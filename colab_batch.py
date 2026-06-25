"""Colab-native folder batch processor for the modern Deep-Live-Cam engine.

All media paths are paths already visible to the Colab runtime.  FFmpeg handles
seek, FPS capping, resize, raw-frame transport, audio muxing, and final encode;
Python only performs face analysis and inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MANIFEST_NAME = ".deep_live_cam_processed.json"
REPORT_NAME = "batch_report.json"
ENGINE_VERSION = "deep-live-cam-remote-v1"


@dataclass(frozen=True)
class ProcessConfig:
    input_dir: Path
    output_dir: Path
    source_face: Path | None
    map_config: Path | None
    ss: float = 0.0
    duration: float | None = None
    start_pct: float = 0.0
    end_pct: float = 100.0
    max_fps: float = 30.0
    max_width: int = 420
    decode_queue: int = 6
    encode_queue: int = 6
    recursive: bool = True
    overwrite: bool = False
    skip_processed: bool = True
    short_video_policy: str = "start"
    cuda_decode: bool = True
    encoder: str = "auto"
    preset: str = "p4"
    quality: int = 18
    many_faces: bool = False
    opacity: float = 1.0
    sharpness: float = 0.0
    mouth_mask_size: float = 0.0
    poisson_blend: bool = False
    color_correction: bool = False
    interpolation_weight: float = 0.0
    enhancer: str = "none"
    face_model_pack: str = "buffalo_l"
    swapper_precision: str = "fp32"
    cache_source_face: bool = True


def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=False, text=True, **kwargs)


def parse_fraction(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path: Path) -> dict[str, Any]:
    result = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ], capture_output=True)
    if result.returncode:
        raise RuntimeError(f"ffprobe failed for {path}:\n{result.stderr[-4000:]}")
    payload = json.loads(result.stdout)
    if not payload.get("streams"):
        raise RuntimeError(f"No video stream found: {path}")
    stream = payload["streams"][0]
    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(stream.get("r_frame_rate")) or 25.0
    duration_value = stream.get("duration") or payload.get("format", {}).get("duration")
    try:
        duration = float(duration_value)
    except (TypeError, ValueError):
        duration = None
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
        "duration": duration,
        "frames": int(stream["nb_frames"]) if str(stream.get("nb_frames", "")).isdigit() else None,
    }


def processing_geometry(width: int, height: int, source_fps: float, max_width: int, max_fps: float) -> tuple[int, int, float]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video geometry: {width}x{height}")
    scale = min(1.0, max_width / width)
    out_width = max(2, int(width * scale) // 2 * 2)
    out_height = max(2, int(round(height * out_width / width / 2.0)) * 2)
    return out_width, out_height, min(source_fps, max_fps)


def discover_videos(root: Path, recursive: bool = True) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def discover_images(root: Path, recursive: bool = True) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_exact(stream: Any, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return b""
        data.extend(chunk)
    return bytes(data)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": path.relative_to(root).as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def config_signature(config: ProcessConfig) -> str:
    ignored = {"input_dir", "output_dir", "overwrite", "skip_processed", "decode_queue", "encode_queue"}
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items() if key not in ignored}
    payload["engine"] = ENGINE_VERSION
    if config.source_face and config.source_face.is_file():
        payload["source_face_sha256"] = file_hash(config.source_face)
    if config.map_config and config.map_config.is_file():
        payload["map_config_sha256"] = file_hash(config.map_config)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def manifest_key(path: Path, root: Path, signature: str) -> str:
    return hashlib.sha256(json.dumps({"input": input_fingerprint(path, root), "config": signature}, sort_keys=True).encode()).hexdigest()


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ffmpeg_has_encoder(name: str) -> bool:
    result = run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True)
    return result.returncode == 0 and name in result.stdout


def decoder_command(path: Path, cuda: bool, start: float, duration: float | None, fps: float, width: int, height: int) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if cuda:
        command += ["-hwaccel", "cuda"]
    if start > 0:
        command += ["-ss", f"{start:.6f}"]
    command += ["-i", str(path)]
    if duration is not None:
        command += ["-t", f"{duration:.6f}"]
    command += [
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"fps={fps:.12g},scale={width}:{height}",
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    return command


def encoder_command(path: Path, output: Path, start: float, duration: float, fps: float, width: int, height: int, encoder: str, preset: str, quality: int) -> list[str]:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}",
        "-framerate", f"{fps:.12g}", "-i", "pipe:0",
    ]
    if start > 0:
        command += ["-ss", f"{start:.6f}"]
    command += ["-t", f"{duration:.6f}", "-i", str(path), "-map", "0:v:0", "-map", "1:a:0?", "-map_metadata", "1"]
    if encoder == "h264_nvenc":
        command += ["-c:v", encoder, "-preset", preset, "-cq", str(quality)]
    else:
        command += ["-c:v", "libx264", "-preset", "medium", "-crf", str(quality)]
    command += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(output)]
    return command


class ModernEngine:
    def __init__(self, config: ProcessConfig):
        import modules.globals as globals_module
        from modules.face_analyser import get_many_faces, get_one_face, set_face_analyser_model_pack
        from modules.processors.frame import face_swapper

        set_face_analyser_model_pack(config.face_model_pack)
        face_swapper.set_face_swapper_precision(config.swapper_precision)
        self.globals = globals_module
        self.get_one_face = get_one_face
        self.get_many_faces = get_many_faces
        self.swapper = face_swapper

        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        globals_module.execution_providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if provider in available_providers
        ]
        print("ONNX Runtime providers:", globals_module.execution_providers)

        print("Checking face swapper model...")
        if not face_swapper.pre_check():
            raise RuntimeError(
                "Could not provision models/inswapper_128.onnx. "
                "Check Colab internet access and rerun the cell."
            )
        if not face_swapper.pre_start():
            raise RuntimeError("The face swapper model could not be loaded.")

        self.source_cache: dict[str, Any] = {}
        self.source_face_path = config.source_face
        self.cache_source_face = bool(config.cache_source_face)
        self.mapping = self._load_mapping(config.map_config)
        self.default_source = self._source(config.source_face) if config.source_face else None
        self.enhancer = self._load_enhancer(config.enhancer)

        globals_module.many_faces = config.many_faces and not self.mapping
        globals_module.map_faces = bool(self.mapping)
        globals_module.opacity = config.opacity
        globals_module.sharpness = config.sharpness
        globals_module.mouth_mask_size = config.mouth_mask_size
        globals_module.mouth_mask = config.mouth_mask_size > 0
        globals_module.poisson_blend = config.poisson_blend
        globals_module.color_correction = config.color_correction
        globals_module.enable_interpolation = 0 < config.interpolation_weight < 1
        globals_module.interpolation_weight = config.interpolation_weight

        if self.default_source is None and not self.mapping:
            raise ValueError("--source-face is required when --map-config is not supplied")

    def _source(self, path: Path | None) -> Any:
        if path is None:
            return None
        key = str(path.resolve())
        if self.cache_source_face and key in self.source_cache:
            return self.source_cache[key]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read source image: {path}")
        face = self.get_one_face(image)
        if face is None:
            raise ValueError(f"No face detected in source image: {path}")
        if self.cache_source_face:
            print(f"Cached source face embedding once: {path}")
            self.source_cache[key] = face
        return face

    def refresh_default_source(self) -> Any:
        if self.source_face_path is None:
            return None
        self.default_source = self._source(self.source_face_path)
        return self.default_source

    def _load_mapping(self, path: Path | None) -> dict[str, list[dict[str, Any]]]:
        if path is None:
            return {}
        payload = load_json(path, {})
        if payload.get("version") != 1 or not isinstance(payload.get("videos"), dict):
            raise ValueError("Mapping JSON must contain version=1 and a videos object")
        base = path.parent
        mapping: dict[str, list[dict[str, Any]]] = {}
        for video, identities in payload["videos"].items():
            mapping[video] = []
            for identity in identities:
                source = identity.get("source_path")
                centroid = np.asarray(identity.get("centroid", []), dtype=np.float32)
                if source and centroid.size:
                    source_path = Path(source)
                    if not source_path.is_absolute():
                        source_path = base / source_path
                    centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
                    mapping[video].append({**identity, "source_path": source_path, "centroid_array": centroid})
        return mapping

    @staticmethod
    def _load_enhancer(name: str) -> Any:
        if name == "none":
            return None
        module_names = {
            "gfpgan": "modules.processors.frame.face_enhancer",
            "gpen256": "modules.processors.frame.face_enhancer_gpen256",
            "gpen512": "modules.processors.frame.face_enhancer_gpen512",
        }
        module = __import__(module_names[name], fromlist=["process_frame"])
        if hasattr(module, "pre_check") and not module.pre_check():
            raise RuntimeError(f"Enhancer pre-check failed: {name}")
        return module

    def reset_video_state(self) -> None:
        if hasattr(self.swapper, "PREVIOUS_FRAME_RESULT"):
            self.swapper.PREVIOUS_FRAME_RESULT = None
        if hasattr(self.swapper, "FACE_DETECTION_CACHE"):
            self.swapper.FACE_DETECTION_CACHE.clear()

    def process(self, frame: np.ndarray, relative_video: str) -> np.ndarray:
        if self.mapping:
            output = frame.copy()
            faces = self.get_many_faces(frame) or []
            entries = self.mapping.get(relative_video, [])
            bboxes = []
            for target in faces:
                embedding = np.asarray(getattr(target, "normed_embedding", target.embedding), dtype=np.float32)
                embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
                match = max(entries, key=lambda item: float(np.dot(embedding, item["centroid_array"])), default=None)
                if match and float(np.dot(embedding, match["centroid_array"])) >= float(match.get("threshold", 0.35)):
                    output = self.swapper.swap_face(self._source(match["source_path"]), target, output)
                    bboxes.append(target.bbox.astype(int))
                elif self.default_source is not None:
                    output = self.swapper.swap_face(self.default_source, target, output)
                    bboxes.append(target.bbox.astype(int))
            output = self.swapper.apply_post_processing(output, bboxes)
            detected = faces
        else:
            detected = self.get_many_faces(frame) if self.globals.many_faces else None
            output = self.swapper.process_frame(self.default_source, frame)
        if self.enhancer:
            output = self.enhancer.process_frame(None, output, detected_faces=detected)
        return output


def effective_segment(info: dict[str, Any], config: ProcessConfig, path: Path) -> tuple[float, float | None]:
    video_duration = info["duration"]

    # If explicit ss/duration are set, use them (legacy behavior)
    if config.ss > 0 or config.duration is not None:
        start = config.ss
        if video_duration is not None and start >= video_duration:
            if config.short_video_policy == "start":
                print(f"  ! shorter than SS={start:g}; using SS=0")
                start = 0.0
            else:
                raise ValueError(f"SS={start} is beyond the end of {path.name}")
        remaining = None if video_duration is None else max(0.0, video_duration - start)
        clip = remaining if config.duration is None else config.duration if remaining is None else min(config.duration, remaining)
    # Otherwise, use percentage-based range
    elif video_duration is not None and (config.start_pct > 0 or config.end_pct < 100):
        start = video_duration * (config.start_pct / 100.0)
        end = video_duration * (config.end_pct / 100.0)
        clip = max(0.0, end - start)
        if clip <= 0:
            raise ValueError(f"Invalid percentage range: {config.start_pct}% to {config.end_pct}%")
    else:
        # Full video
        start = 0.0
        clip = video_duration

    if clip is not None and clip <= 0:
        raise ValueError("No video remains after seek")
    return start, clip


def _start_decoder(path: Path, config: ProcessConfig, start: float, clip: float | None, fps: float, width: int, height: int, cuda: bool) -> subprocess.Popen:
    return subprocess.Popen(decoder_command(path, cuda, start, clip, fps, width, height), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)


def process_one(path: Path, output: Path, relative: str, config: ProcessConfig, engine: ModernEngine) -> dict[str, Any]:
    info = probe_video(path)
    width, height, fps = processing_geometry(info["width"], info["height"], info["fps"], config.max_width, config.max_fps)
    start, clip = effective_segment(info, config, path)
    expected = max(1, int(round(clip * fps))) if clip is not None else None
    encode_duration = clip or max(1 / fps, ((info["frames"] or 86400 * info["fps"]) / info["fps"]) - start)
    frame_size = width * height * 3
    engine.reset_video_state()
    print(f"  {info['width']}x{info['height']} @ {info['fps']:.3f} -> {width}x{height} @ {fps:.3f}")

    decoder = _start_decoder(path, config, start, clip, fps, width, height, config.cuda_decode)
    first = read_exact(decoder.stdout, frame_size)
    if not first and config.cuda_decode:
        error = decoder.stderr.read().decode(errors="replace")
        decoder.wait()
        print("  ! CUDA decode unavailable; using software decode")
        if error.strip():
            print("    " + error.strip().splitlines()[-1])
        decoder = _start_decoder(path, config, start, clip, fps, width, height, False)
        first = read_exact(decoder.stdout, frame_size)
    if not first:
        error = decoder.stderr.read().decode(errors="replace")
        decoder.wait()
        raise RuntimeError("FFmpeg produced no frames:\n" + error[-4000:])

    selected_encoder = config.encoder
    if selected_encoder == "auto":
        selected_encoder = "h264_nvenc" if ffmpeg_has_encoder("h264_nvenc") else "libx264"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    encoder = subprocess.Popen(encoder_command(path, output, start, encode_duration, fps, width, height, selected_encoder, config.preset, config.quality), stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)

    decoded: queue.Queue[Any] = queue.Queue(config.decode_queue)
    encoded: queue.Queue[Any] = queue.Queue(config.encode_queue)
    errors: queue.Queue[tuple[str, BaseException]] = queue.Queue()
    sentinel = object()
    stop = threading.Event()

    def decoder_worker() -> None:
        try:
            raw = first
            while raw and not stop.is_set():
                while not stop.is_set():
                    try:
                        decoded.put(raw, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                raw = read_exact(decoder.stdout, frame_size)
            while not stop.is_set():
                try:
                    decoded.put(sentinel, timeout=0.1)
                    break
                except queue.Full:
                    continue
        except BaseException as exc:
            errors.put(("decode", exc))
            try: decoded.put(sentinel, timeout=1)
            except queue.Full: pass

    def encoder_worker() -> None:
        try:
            while True:
                raw = encoded.get()
                if raw is sentinel:
                    return
                encoder.stdin.write(raw)
        except BaseException as exc:
            errors.put(("encode", exc))

    decode_thread = threading.Thread(target=decoder_worker, daemon=True)
    encode_thread = threading.Thread(target=encoder_worker, daemon=True)
    decode_thread.start(); encode_thread.start()
    frames = fallbacks = 0
    started = time.monotonic()
    try:
        while True:
            if not errors.empty():
                stage, exc = errors.get()
                raise RuntimeError(f"{stage} pipeline failed: {exc}") from exc
            raw = decoded.get(timeout=30)
            if raw is sentinel:
                break
            # Frames backed directly by immutable pipe bytes are read-only.
            # The modern swapper pastes results in-place, so it needs its own
            # writable contiguous frame buffer.
            frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()
            try:
                result = engine.process(frame, relative)
                if result is None:
                    result = frame; fallbacks += 1
            except Exception as exc:
                result = frame; fallbacks += 1
                if fallbacks <= 3:
                    print(f"  ! frame fallback: {exc}")
            if result.shape[:2] != (height, width):
                result = cv2.resize(result, (width, height))
            encoded.put(np.ascontiguousarray(result).tobytes())
            frames += 1
            if frames % 30 == 0 or frames == expected:
                suffix = f"/{expected}" if expected else ""
                print(f"\r  frames {frames}{suffix}", end="", flush=True)
            if expected and frames >= expected:
                stop.set(); break
        print()
        encoded.put(sentinel)
        encode_thread.join(timeout=60)
        encoder.stdin.close(); encoder.stdin = None
        if stop.is_set() and decoder.poll() is None:
            decoder.terminate()
        decoder.wait(timeout=20)
        rc = encoder.wait(timeout=120)
        error = encoder.stderr.read().decode(errors="replace")
        if rc:
            raise RuntimeError("FFmpeg encode failed:\n" + error[-4000:])
    finally:
        stop.set()
        for process in (decoder, encoder):
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill()
        decode_thread.join(timeout=5)
        encode_thread.join(timeout=5)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Output not created: {output}")
    return {"frames": frames, "fallback_frames": fallbacks, "fps": fps, "width": width, "height": height, "seconds": time.monotonic() - started, "size_mb": output.stat().st_size / 1048576}


def scan_identities(args: argparse.Namespace) -> int:
    import modules.globals as globals_module
    from modules.face_analyser import get_many_faces
    globals_module.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    input_dir, mapping_dir = Path(args.input_dir), Path(args.mapping_dir)
    mapping_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"version": 1, "instructions": "Set source_path for each identity, then pass this file to process --map-config.", "videos": {}}
    for video in discover_videos(input_dir, args.recursive):
        relative = video.relative_to(input_dir).as_posix()
        capture = cv2.VideoCapture(str(video))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        every = max(1, int(round(fps * args.sample_seconds)))
        samples: list[dict[str, Any]] = []
        index = 0
        while len(samples) < args.max_samples:
            ok, frame = capture.read()
            if not ok: break
            if index % every == 0:
                for face in get_many_faces(frame) or []:
                    vector = np.asarray(getattr(face, "normed_embedding", face.embedding), np.float32)
                    vector /= max(float(np.linalg.norm(vector)), 1e-8)
                    bbox = np.asarray(face.bbox, int)
                    x1, y1, x2, y2 = np.maximum(bbox, 0)
                    crop = frame[y1:y2, x1:x2].copy()
                    if crop.size: samples.append({"embedding": vector, "crop": crop})
            index += 1
        capture.release()
        clusters: list[dict[str, Any]] = []
        for sample in samples:
            match = max(clusters, key=lambda item: float(np.dot(sample["embedding"], item["centroid"])), default=None)
            if match is None or float(np.dot(sample["embedding"], match["centroid"])) < args.cluster_threshold:
                clusters.append({"centroid": sample["embedding"].copy(), "count": 1, "crop": sample["crop"]})
            else:
                match["count"] += 1
                match["centroid"] += (sample["embedding"] - match["centroid"]) / match["count"]
                match["centroid"] /= max(float(np.linalg.norm(match["centroid"])), 1e-8)
        identities = []
        thumbs = []
        stem = hashlib.sha1(relative.encode()).hexdigest()[:10]
        for number, cluster in enumerate(sorted(clusters, key=lambda item: item["count"], reverse=True), 1):
            thumb_name = f"{stem}_identity_{number:02d}.jpg"
            cv2.imwrite(str(mapping_dir / thumb_name), cluster["crop"])
            identities.append({"target_id": number, "samples": cluster["count"], "thumbnail": thumb_name, "source_path": "", "threshold": args.match_threshold, "centroid": cluster["centroid"].tolist()})
            thumb = cv2.resize(cluster["crop"], (180, 180)); cv2.putText(thumb, f"ID {number}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2); thumbs.append(thumb)
        if thumbs:
            columns = min(4, len(thumbs)); rows = math.ceil(len(thumbs) / columns)
            sheet = np.zeros((rows * 180, columns * 180, 3), np.uint8)
            for i, thumb in enumerate(thumbs): sheet[(i // columns)*180:(i // columns+1)*180, (i % columns)*180:(i % columns+1)*180] = thumb
            cv2.imwrite(str(mapping_dir / f"{stem}_contact_sheet.jpg"), sheet)
        payload["videos"][relative] = identities
        print(f"{relative}: {len(identities)} identities from {len(samples)} samples")
    output = mapping_dir / "face_mapping.json"
    atomic_json(output, payload)
    print(f"Mapping template: {output}")
    return 0


def process_batch(args: argparse.Namespace) -> int:
    config = ProcessConfig(
        input_dir=Path(args.input_dir), output_dir=Path(args.output_dir),
        source_face=Path(args.source_face) if args.source_face else None,
        map_config=Path(args.map_config) if args.map_config else None,
        ss=args.ss, duration=args.duration, start_pct=args.start_pct, end_pct=args.end_pct,
        max_fps=args.max_fps, max_width=args.max_width,
        decode_queue=args.decode_queue, encode_queue=args.encode_queue, recursive=args.recursive,
        overwrite=args.overwrite, skip_processed=args.skip_processed,
        short_video_policy=args.short_video_policy, cuda_decode=args.cuda_decode,
        encoder=args.encoder, preset=args.preset, quality=args.quality, many_faces=args.many_faces,
        opacity=args.opacity, sharpness=args.sharpness, mouth_mask_size=args.mouth_mask_size,
        poisson_blend=args.poisson_blend, color_correction=args.color_correction,
        interpolation_weight=args.interpolation_weight, enhancer=args.enhancer,
    )
    if not config.input_dir.is_dir(): raise NotADirectoryError(config.input_dir)
    if config.source_face and not config.source_face.is_file(): raise FileNotFoundError(config.source_face)
    if config.map_config and not config.map_config.is_file(): raise FileNotFoundError(config.map_config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    videos = discover_videos(config.input_dir, config.recursive)
    if not videos: raise FileNotFoundError(f"No videos found in {config.input_dir}")
    engine = ModernEngine(config)
    signature = config_signature(config)
    manifest_path = config.output_dir / MANIFEST_NAME
    manifest = load_json(manifest_path, {"version": 1, "items": {}})
    items = manifest.setdefault("items", {})
    report: dict[str, Any] = {"engine": ENGINE_VERSION, "config_signature": signature, "completed": [], "skipped": [], "failed": []}
    suffix = f"_ss{config.ss:g}" if config.ss else ""
    if config.duration is not None: suffix += f"_dur{config.duration:g}"
    cancel_event = getattr(args, "cancel_event", None)
    for number, video in enumerate(videos, 1):
        if cancel_event is not None and cancel_event.is_set():
            print("  cancel requested; stopping before next video")
            break
        relative = video.relative_to(config.input_dir).as_posix()
        output = config.output_dir / Path(relative).parent / f"{video.stem}_face_swapped{suffix}.mp4"
        key = manifest_key(video, config.input_dir, signature)
        print(f"\n[{number}/{len(videos)}] {relative}")
        if not config.overwrite and config.skip_processed and key in items and Path(items[key].get("output", "")).is_file():
            print("  skipped: matching input + source/model/config manifest entry")
            report["skipped"].append(relative); continue
        try:
            result = process_one(video, output, relative, config, engine)
            record = {"input": relative, "output": str(output), **result}
            report["completed"].append(record)
            items[key] = {**record, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            atomic_json(manifest_path, manifest)
            print(f"  done: {output} ({result['size_mb']:.1f} MB)")
        except Exception as exc:
            output.unlink(missing_ok=True)
            report["failed"].append({"input": relative, "error": str(exc)})
            print(f"  FAILED: {exc}")
    report_path = config.output_dir / REPORT_NAME
    atomic_json(report_path, report)
    if args.zip_output:
        zip_path = Path(args.zip_output)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        created = shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=config.output_dir)
        print(f"ZIP: {created}")
    print(f"\nCompleted {len(report['completed'])}; skipped {len(report['skipped'])}; failed {len(report['failed'])}")
    return 1 if report["failed"] else 0


def process_image_one(path: Path, output: Path, relative: str, config: ProcessConfig, engine: ModernEngine) -> dict[str, Any]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {path}")
    engine.reset_video_state()
    started = time.monotonic()
    result = engine.process(frame.copy(), relative)
    if result is None:
        result = frame
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    if not cv2.imwrite(str(output), np.ascontiguousarray(result)):
        raise RuntimeError(f"Could not write image: {output}")
    return {"width": int(result.shape[1]), "height": int(result.shape[0]), "seconds": time.monotonic() - started, "size_mb": output.stat().st_size / 1048576}


def process_photos(args: argparse.Namespace) -> int:
    config = ProcessConfig(
        input_dir=Path(args.input_dir), output_dir=Path(args.output_dir),
        source_face=Path(args.source_face) if args.source_face else None,
        map_config=Path(args.map_config) if args.map_config else None,
        recursive=args.recursive, overwrite=args.overwrite, skip_processed=args.skip_processed,
        many_faces=args.many_faces, opacity=args.opacity, sharpness=args.sharpness,
        mouth_mask_size=args.mouth_mask_size, poisson_blend=args.poisson_blend,
        color_correction=args.color_correction, interpolation_weight=args.interpolation_weight,
        enhancer=args.enhancer,
    )
    if not config.input_dir.is_dir(): raise NotADirectoryError(config.input_dir)
    if config.source_face and not config.source_face.is_file(): raise FileNotFoundError(config.source_face)
    if config.map_config and not config.map_config.is_file(): raise FileNotFoundError(config.map_config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    images = discover_images(config.input_dir, config.recursive)
    if not images: raise FileNotFoundError(f"No images found in {config.input_dir}")
    engine = ModernEngine(config)
    signature = config_signature(config)
    manifest_path = config.output_dir / MANIFEST_NAME
    manifest = load_json(manifest_path, {"version": 1, "items": {}})
    items = manifest.setdefault("items", {})
    report: dict[str, Any] = {"engine": ENGINE_VERSION, "config_signature": signature, "completed": [], "skipped": [], "failed": []}
    cancel_event = getattr(args, "cancel_event", None)
    for number, image in enumerate(images, 1):
        if cancel_event is not None and cancel_event.is_set():
            print("  cancel requested; stopping before next image")
            break
        relative = image.relative_to(config.input_dir).as_posix()
        output = config.output_dir / Path(relative).parent / f"{image.stem}_face_swapped{image.suffix.lower()}"
        key = manifest_key(image, config.input_dir, signature)
        print(f"\n[{number}/{len(images)}] {relative}")
        if not config.overwrite and config.skip_processed and key in items and Path(items[key].get("output", "")).is_file():
            print("  skipped: matching input + source/model/config manifest entry")
            report["skipped"].append(relative); continue
        try:
            result = process_image_one(image, output, relative, config, engine)
            record = {"input": relative, "output": str(output), **result}
            report["completed"].append(record)
            items[key] = {**record, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            atomic_json(manifest_path, manifest)
            print(f"  done: {output} ({result['size_mb']:.1f} MB)")
        except Exception as exc:
            output.unlink(missing_ok=True)
            report["failed"].append({"input": relative, "error": str(exc)})
            print(f"  FAILED: {exc}")
    report_path = config.output_dir / REPORT_NAME
    atomic_json(report_path, report)
    print(f"\nCompleted {len(report['completed'])}; skipped {len(report['skipped'])}; failed {len(report['failed'])}")
    return 1 if report["failed"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan", help="scan target identities and write contact sheets + editable JSON")
    scan.add_argument("--input-dir", required=True); scan.add_argument("--mapping-dir", required=True)
    scan.add_argument("--sample-seconds", type=float, default=1.0); scan.add_argument("--max-samples", type=int, default=300)
    scan.add_argument("--cluster-threshold", type=float, default=0.55); scan.add_argument("--match-threshold", type=float, default=0.35)
    scan.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True); scan.set_defaults(func=scan_identities)
    def add_common_process_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--source-face"); command.add_argument("--input-dir", required=True); command.add_argument("--output-dir", required=True)
        command.add_argument("--map-config")
        command.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
        command.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
        command.add_argument("--skip-processed", action=argparse.BooleanOptionalAction, default=True)
        command.add_argument("--many-faces", action=argparse.BooleanOptionalAction, default=False)
        command.add_argument("--opacity", type=float, default=1.0); command.add_argument("--sharpness", type=float, default=0.0)
        command.add_argument("--mouth-mask-size", type=float, default=0.0)
        command.add_argument("--poisson-blend", action=argparse.BooleanOptionalAction, default=False)
        command.add_argument("--color-correction", action=argparse.BooleanOptionalAction, default=False)
        command.add_argument("--interpolation-weight", type=float, default=0.0)
        command.add_argument("--enhancer", choices=["none", "gfpgan", "gpen256", "gpen512"], default="none")

    process = subparsers.add_parser("process", help="process every input video through the modern engine")
    add_common_process_args(process)
    process.add_argument("--zip-output")
    process.add_argument("--ss", type=float, default=0.0); process.add_argument("--duration", type=float)
    process.add_argument("--start-pct", type=float, default=0.0); process.add_argument("--end-pct", type=float, default=100.0)
    process.add_argument("--max-fps", type=float, default=30.0); process.add_argument("--max-width", type=int, default=420)
    process.add_argument("--decode-queue", type=int, default=6); process.add_argument("--encode-queue", type=int, default=6)
    process.add_argument("--short-video-policy", choices=["start", "skip"], default="start")
    process.add_argument("--cuda-decode", action=argparse.BooleanOptionalAction, default=True)
    process.add_argument("--encoder", choices=["auto", "h264_nvenc", "libx264"], default="auto")
    process.add_argument("--preset", default="p4"); process.add_argument("--quality", type=int, default=18)
    process.set_defaults(func=process_batch)
    photos = subparsers.add_parser("photos", help="process every input image through the modern engine")
    add_common_process_args(photos)
    photos.set_defaults(func=process_photos)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "ss", 0) < 0: raise ValueError("--ss must be non-negative")
    if getattr(args, "duration", None) is not None and args.duration <= 0: raise ValueError("--duration must be positive")
    if getattr(args, "max_fps", 1) <= 0 or getattr(args, "max_width", 1) <= 0: raise ValueError("FPS and width limits must be positive")
    if not 0 <= getattr(args, "opacity", 1) <= 1: raise ValueError("--opacity must be between 0 and 1")
    start_pct, end_pct = getattr(args, "start_pct", 0.0), getattr(args, "end_pct", 100.0)
    if not 0 <= start_pct < 100: raise ValueError("--start-pct must be between 0 and 99")
    if not 0 < end_pct <= 100: raise ValueError("--end-pct must be between 1 and 100")
    if start_pct >= end_pct: raise ValueError("--start-pct must be less than --end-pct")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
