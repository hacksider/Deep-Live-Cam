#!/usr/bin/env python3

import os
import sys

# Add the project root to PATH so bundled ffmpeg/ffprobe are found
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")

# On Windows, add NVIDIA CUDA DLL directories to PATH so onnxruntime-gpu can
# find cuDNN/cublas. PyTorch bundles cuDNN in its lib/ dir; pip nvidia-* pkgs
# use bin/. Skipped on macOS/Linux where loader paths handle this.
if sys.platform == "win32":
    _site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    _venv_site_packages = os.path.join(project_root, "venv", "Lib", "site-packages")
    for _sp in (_site_packages, _venv_site_packages):
        _torch_lib = os.path.join(_sp, "torch", "lib")
        if os.path.isdir(_torch_lib):
            os.environ["PATH"] = _torch_lib + os.pathsep + os.environ["PATH"]
        _nvidia_dir = os.path.join(_sp, "nvidia")
        if os.path.isdir(_nvidia_dir):
            for _pkg in os.listdir(_nvidia_dir):
                _bin_dir = os.path.join(_nvidia_dir, _pkg, "bin")
                if os.path.isdir(_bin_dir):
                    os.environ["PATH"] = _bin_dir + os.pathsep + os.environ["PATH"]

# Import the tkinter fix to patch the ScreenChanged error
import tkinter_fix

from modules import platform_info
platform_info.print_banner()

from modules import core

if __name__ == '__main__':
    core.run()
