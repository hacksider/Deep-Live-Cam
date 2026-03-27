#!/usr/bin/env python3

import os
import sys

# Add the project root to PATH so bundled ffmpeg/ffprobe are found
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")

# Add NVIDIA CUDA DLL directories to PATH so onnxruntime-gpu can find them
nvidia_dir = os.path.join(project_root, "venv", "Lib", "site-packages", "nvidia")
if os.path.isdir(nvidia_dir):
    for pkg in os.listdir(nvidia_dir):
        bin_dir = os.path.join(nvidia_dir, pkg, "bin")
        if os.path.isdir(bin_dir):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]

# Import the tkinter fix to patch the ScreenChanged error
import tkinter_fix

from modules import core

if __name__ == '__main__':
    core.run()
