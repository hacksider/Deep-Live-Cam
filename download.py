from pathlib import Path
from urllib.request import Request, urlopen
import os
MODELS = {
    "inswapper_128_fp16.onnx": (
        "https://huggingface.co/hacksider/deep-live-cam/"
        "resolve/main/inswapper_128_fp16.onnx?download=true"
    ),
    "gfpgan-1024.onnx": (
        "https://huggingface.co/hacksider/deep-live-cam/"
        "resolve/main/gfpgan-1024.onnx?download=true"
    ),
}
def download_models():
    """Download required ONNX models into ./models."""
    models_dir = Path("./models")
    models_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        destination = models_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".part")
        if destination.exists() and destination.stat().st_size > 0:
            print(f"[SKIP] Already downloaded: {destination}")
            continue
        print(f"[DOWNLOAD] {filename}")
        try:
            request = Request(
                url,
                headers={"User-Agent": "Python-Model-Downloader/1.0"},
            )
            with urlopen(request, timeout=60) as response:
                with open(temporary, "wb") as file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        file.write(chunk)
            if temporary.stat().st_size == 0:
                raise RuntimeError("Downloaded file is empty.")
            os.replace(temporary, destination)
            print(f"[OK] Saved to: {destination}")
        except Exception as error:
            if temporary.exists():
                temporary.unlink()
            print(f"[ERROR] Failed to download {filename}: {error}")
            raise
    print("[DONE] All models are ready.")
if __name__ == "__main__":
    download_models()
