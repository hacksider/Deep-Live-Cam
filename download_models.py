#!/usr/bin/env python3
"""Download required model files into the models/ directory."""

import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = [
    {
        "filename": "GFPGANv1.4.onnx",
        "url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx",
    },
    {
        "filename": "inswapper_128_fp16.onnx",
        "url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx",
    },
]

# Utility function to download .onnx file from provided URL and place into dst dir
def download(url, dest_path):
    filename = os.path.basename(dest_path)
    print(f"Downloading {filename}...")
    
    # Used to output current state / progress of the download.
    def reporthook(count, block_size, total_size):
        if total_size <= 0:
            return

        percent = min(100, count * block_size * 100 // total_size)
        downloaded = min(count * block_size, total_size)
        mb_done = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024

        print(f"\r  {percent:3d}%  {mb_done:.1f} / {mb_total:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for model in MODELS:
        dest = os.path.join(MODELS_DIR, model["filename"])

        if os.path.exists(dest):
            size_mb = os.path.getsize(dest) / 1024 / 1024
            print(f"Skipping {model['filename']} (already exists, {size_mb:.1f} MB)")
            continue

        try:
            download(model["url"], dest)
            size_mb = os.path.getsize(dest) / 1024 / 1024
            print(f"  Saved to {dest} ({size_mb:.1f} MB)")

        except Exception as e:
            print(f"\nERROR downloading {model['filename']}: {e}", file=sys.stderr)
            if os.path.exists(dest):
                os.remove(dest)
            sys.exit(1)

    print("\nAll models ready.")


if __name__ == "__main__":
    main()
