import os
import platform
import ssl
import threading
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from tqdm import tqdm

from modules.paths import MODELS_DIR

HF_REPO_ID = "hacksider/deep-live-cam"
HF_RESOLVE_BASE = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/"

MODEL_SIZES: Dict[str, int] = {
    "inswapper_128.onnx": 554253681,
    "inswapper_128_fp16.onnx": 277680638,
    "gfpgan-1024.onnx": 365875079,
    "GPEN-BFR-256.onnx": 75715262,
    "GPEN-BFR-512.onnx": 284244491,
    "buffalo_l/buffalo_l/1k3d68.onnx": 143607619,
    "buffalo_l/buffalo_l/2d106det.onnx": 5030888,
    "buffalo_l/buffalo_l/det_10g.onnx": 16923827,
    "buffalo_l/buffalo_l/genderage.onnx": 1322532,
    "buffalo_l/buffalo_l/w600k_r50.onnx": 174383860,
}

_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

CHUNK_SIZE = 1024 * 256


def _ssl_context():
    if platform.system().lower() == "darwin":
        return ssl._create_unverified_context()
    return None


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if key not in _LOCKS:
            _LOCKS[key] = threading.Lock()
        return _LOCKS[key]


def resolve_url(name: str) -> str:
    return HF_RESOLVE_BASE + name.replace(os.sep, "/")


def local_path(name: str, dest_dir: Optional[str] = None) -> str:
    if dest_dir is not None:
        return os.path.join(dest_dir, os.path.basename(name))
    return os.path.join(MODELS_DIR, *name.replace("/", os.sep).split(os.sep))


def expected_size(name: str) -> Optional[int]:
    return MODEL_SIZES.get(name.replace(os.sep, "/"))


def is_present(name: str, dest_dir: Optional[str] = None) -> bool:
    path = local_path(name, dest_dir)
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _download(name: str, url: str, target: str, size: Optional[int]) -> bool:
    os.makedirs(os.path.dirname(target) or MODELS_DIR, exist_ok=True)
    partial = target + ".part"
    resume_from = os.path.getsize(partial) if os.path.isfile(partial) else 0

    headers = {"User-Agent": "Deep-Live-Cam"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    try:
        request = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(request, context=_ssl_context(), timeout=60)
    except urllib.error.HTTPError as error:
        if resume_from and error.code in (416, 501):
            try:
                os.remove(partial)
            except OSError:
                pass
            return _download(name, url, target, size)
        print(f"[DLC.MODELS] Failed to download {name}: HTTP {error.code}")
        return False
    except (urllib.error.URLError, OSError) as error:
        print(f"[DLC.MODELS] Failed to download {name}: {error}")
        return False

    with response:
        if resume_from and getattr(response, "status", 200) != 206:
            resume_from = 0
        remaining = int(response.headers.get("Content-Length", 0) or 0)
        total = size or (resume_from + remaining) or None
        mode = "ab" if resume_from else "wb"
        try:
            with open(partial, mode) as handle:
                with tqdm(
                    total=total,
                    initial=resume_from,
                    desc=f"Downloading {os.path.basename(name)}",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress:
                    while True:
                        buffer = response.read(CHUNK_SIZE)
                        if not buffer:
                            break
                        handle.write(buffer)
                        progress.update(len(buffer))
        except (urllib.error.URLError, OSError) as error:
            print(f"[DLC.MODELS] Download of {name} interrupted: {error}")
            return False

    downloaded = os.path.getsize(partial)
    if size is not None and downloaded != size:
        print(f"[DLC.MODELS] {name} is {downloaded} bytes, expected {size}. Discarding.")
        try:
            os.remove(partial)
        except OSError:
            pass
        return False

    try:
        os.replace(partial, target)
    except OSError as error:
        print(f"[DLC.MODELS] Could not finalise {name}: {error}")
        return False
    return True


def ensure_model(
    name: str, quiet: bool = False, dest_dir: Optional[str] = None
) -> Optional[str]:
    name = name.replace(os.sep, "/")
    target = local_path(name, dest_dir)

    with _lock_for(target):
        if is_present(name, dest_dir):
            return target
        if not quiet:
            print(f"[DLC.MODELS] {name} not found in models folder, downloading...")
        if _download(name, resolve_url(name), target, expected_size(name)):
            return target
    return None


def ensure_any(names: List[str]) -> Optional[str]:
    for name in names:
        if is_present(name):
            return local_path(name)
    for name in names:
        path = ensure_model(name)
        if path is not None:
            return path
    return None


def ensure_insightface_pack(name: str = "buffalo_l") -> bool:
    members = [n for n in MODEL_SIZES if n.startswith(f"{name}/")]
    if not members:
        return False

    dest_dir = os.path.join(os.path.expanduser("~"), ".insightface", "models", name)
    if all(is_present(member, dest_dir) for member in members):
        return True

    print(f"[DLC.MODELS] insightface pack '{name}' is missing, downloading...")
    ok = True
    for member in members:
        if ensure_model(member, quiet=True, dest_dir=dest_dir) is None:
            ok = False
    if not ok:
        print(f"[DLC.MODELS] Could not pre-fill '{name}'; insightface will retry.")
    return ok
