from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import os
import re
MODELS = {
    "inswapper_128_fp16.onnx": {
        "url": (
            "https://huggingface.co/hacksider/deep-live-cam/"
            "resolve/main/inswapper_128_fp16.onnx?download=true"
        ),
        "max_size": 900_000_000,
    },
    "gfpgan-1024.onnx": {
        "url": (
            "https://huggingface.co/hacksider/deep-live-cam/"
            "resolve/main/gfpgan-1024.onnx?download=true"
        ),
        "max_size": 1000_000_000,
    },
}
CHUNK_SIZE = 1024 * 1024
TIMEOUT = 60
MAX_RETRIES = 5
def _content_range(response):
    """Parse Content-Range: bytes START-END/TOTAL."""
    value = response.headers.get("Content-Range", "")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value)
    if not match:
        return None
    start, end, total = map(int, match.groups())
    if start > end or end >= total:
        return None
    return start, end, total
def _download_one(filename, config, directory):
    """Download or resume one model, enforcing size limits."""
    destination = directory / filename
    temporary = directory / (filename + ".part")
    url = config["url"]
    max_size = config["max_size"]
    if destination.exists():
        size = destination.stat().st_size
        if 0 < size <= max_size:
            print(f"[SKIP] Already exists: {destination}")
            return
        print(f"[WARN] Existing model has an invalid size: {destination}")
        destination.unlink()
    for attempt in range(1, MAX_RETRIES + 1):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > max_size:
            print(f"[WARN] Partial file exceeds limit; restarting {filename}")
            temporary.unlink()
            offset = 0
        headers = {
            "User-Agent": "Python-Model-Downloader/1.0",
            "Accept-Encoding": "identity",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=TIMEOUT) as response:
                status = response.status
                content_length = response.headers.get("Content-Length")
                if offset:
                    # A resumed request must return a valid partial response.
                    if status == 206:
                        content_range = _content_range(response)
                        if not content_range:
                            raise RuntimeError(
                                "Invalid Content-Range on resumed download."
                            )
                        start, end, total = content_range
                        if start != offset:
                            raise RuntimeError(
                                "Server returned an unexpected resume offset."
                            )
                        if total > max_size:
                            raise RuntimeError(
                                f"{filename} exceeds its maximum size."
                            )
                        expected_bytes = end - start + 1
                        if (
                            content_length is not None
                            and int(content_length) != expected_bytes
                        ):
                            raise RuntimeError(
                                "Content-Length does not match Content-Range."
                            )
                        if end + 1 != total:
                            raise RuntimeError(
                                "Incomplete range response; refusing to append "
                                "a partial segment."
                            )
                        mode = "ab"
                    elif status == 200:
                        # Server ignored Range; restart safely from byte zero.
                        offset = 0
                        mode = "wb"
                    else:
                        raise RuntimeError(
                            f"Unexpected HTTP status: {status}"
                        )
                else:
                    if status != 200:
                        raise RuntimeError(
                            f"Unexpected HTTP status: {status}"
                        )
                    mode = "wb"
                if content_length is not None:
                    try:
                        response_length = int(content_length)
                    except ValueError as error:
                        raise RuntimeError(
                            "Invalid Content-Length header."
                        ) from error
                    if response_length < 0:
                        raise RuntimeError("Negative Content-Length.")
                    if offset + response_length > max_size:
                        raise RuntimeError(
                            f"{filename} exceeds its maximum size."
                        )
                written = offset
                with open(temporary, mode) as file:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > max_size:
                            raise RuntimeError(
                                f"{filename} exceeded its maximum size."
                            )
                        file.write(chunk)
                    file.flush()
                    os.fsync(file.fileno())
                final_size = temporary.stat().st_size
                if content_length is not None:
                    expected_size = offset + int(content_length)
                    if final_size != expected_size:
                        raise ConnectionError(
                            f"Incomplete download: expected {expected_size} "
                            f"bytes, received {final_size}."
                        )
                if final_size == 0:
                    raise RuntimeError("Downloaded file is empty.")
                # A complete file must have the expected total size when
                # the server supplied a validated Content-Range.
                if offset and status == 206:
                    if final_size != total:
                        raise ConnectionError(
                            f"Incomplete model: expected {total} bytes, "
                            f"received {final_size}."
                        )
                # Atomically publish the completed file.
                os.replace(temporary, destination)
                print(
                    f"[OK] Saved {destination} "
                    f"({destination.stat().st_size:,} bytes)"
                )
                return
        except HTTPError as error:
            if error.code == 416 and temporary.exists():
                # The server says the requested range is unsatisfiable.
                # Do not assume the partial file is complete.
                if temporary.stat().st_size > max_size:
                    temporary.unlink()
                print(
                    f"[WARN] Cannot resume {filename}; "
                    "restarting from zero."
                )
                temporary.unlink(missing_ok=True)
                continue
            print(f"[WARN] Attempt {attempt}/{MAX_RETRIES}: {error}")
        except (OSError, ConnectionError, TimeoutError) as error:
            # Keep the partial file for the next attempt.
            print(f"[WARN] Attempt {attempt}/{MAX_RETRIES}: {error}")
        except Exception:
            # Preserve partial data on failure, but never publish it as final.
            raise
        if attempt < MAX_RETRIES:
            print(f"[RETRY] Retrying {filename}...")
    raise RuntimeError(
        f"Failed to download {filename} after {MAX_RETRIES} attempts. "
        f"Partial data, if any, is preserved at {temporary}."
    )
def download_models():
    """Download both models into ./models, resuming interrupted transfers."""
    models_dir = Path("./models")
    models_dir.mkdir(parents=True, exist_ok=True)
    for filename, config in MODELS.items():
        print(f"[DOWNLOAD] {filename}")
        _download_one(filename, config, models_dir)
    print("[DONE] All models are ready.")
if __name__ == "__main__":
    download_models()
