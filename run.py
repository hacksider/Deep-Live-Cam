#!/usr/bin/env python3

import os
import sys

# Add the project root to PATH so bundled ffmpeg/ffprobe are found
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] = project_root + os.pathsep + os.environ.get("PATH", "")

# On Windows, register NVIDIA CUDA DLL directories so onnxruntime-gpu can
# find cuDNN/cublas. Python 3.8+ ignores PATH for extension-module native deps —
# os.add_dll_directory() is required. Also keep PATH for child processes/ffmpeg.
if sys.platform == "win32":
    _site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    _venv_site_packages = os.path.join(project_root, "venv", "Lib", "site-packages")
    for _sp in (_site_packages, _venv_site_packages):
        _candidate_dirs = []
        _torch_lib = os.path.join(_sp, "torch", "lib")
        if os.path.isdir(_torch_lib):
            _candidate_dirs.append(_torch_lib)
        _nvidia_dir = os.path.join(_sp, "nvidia")
        if os.path.isdir(_nvidia_dir):
            for _pkg in os.listdir(_nvidia_dir):
                _bin_dir = os.path.join(_nvidia_dir, _pkg, "bin")
                if os.path.isdir(_bin_dir):
                    _candidate_dirs.append(_bin_dir)
        for _d in _candidate_dirs:
            os.environ["PATH"] = _d + os.pathsep + os.environ["PATH"]
            try:
                os.add_dll_directory(_d)
            except (OSError, AttributeError):
                pass

    # On Windows, register OpenVINO DLL directories so onnxruntime's
    # OpenVINOExecutionProvider can find openvino.dll.  This must happen
    # before any ONNX InferenceSession is created.  Failure is non-fatal:
    # OpenVINO simply isn't installed, and onnxruntime will fall back to CPU.
    try:
        from onnxruntime.tools.add_openvino_win_libs import (  # type: ignore[import-untyped]  # noqa: E501
            add_openvino_libs_to_path,
        )
        add_openvino_libs_to_path()
    except ImportError:
        # onnxruntime build without the OpenVINO tooling module — no-op.
        pass
    except FileNotFoundError:
        # OpenVINO site-packages dir absent — no-op.
        pass
    except SystemExit as exc:
        # add_openvino_libs_to_path() calls sys.exit() when OpenVINO libs
        # can't be located (e.g. OPENVINO_LIB_PATHS unset).  Log the message
        # it raised with so the failure is visible, but keep startup alive.
        print(
            f"[startup] OpenVINO DLL registration skipped: {exc}",
            flush=True,
        )

# On Linux, pre-load NVIDIA shared libraries (cuDNN, cuBLAS, nvrtc...) shipped
# inside the venv via pip wheels (nvidia-cudnn-cu12, etc.). LD_LIBRARY_PATH
# cannot be set after Python starts, so we use ctypes.CDLL with RTLD_GLOBAL
# instead. This makes symbols available to onnxruntime when it dlopens its
# CUDA provider.
if sys.platform.startswith("linux"):
    import ctypes
    import glob
    _py_lib = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _site_packages_candidates = [
        os.path.join(project_root, "venv", "lib", _py_lib, "site-packages"),
        os.path.join(sys.prefix, "lib", _py_lib, "site-packages"),
    ]
    for _sp in _site_packages_candidates:
        _nvidia_dir = os.path.join(_sp, "nvidia")
        if not os.path.isdir(_nvidia_dir):
            continue
        for _pkg in os.listdir(_nvidia_dir):
            _lib_dir = os.path.join(_nvidia_dir, _pkg, "lib")
            if not os.path.isdir(_lib_dir):
                continue
            # Also expose the directory to child processes, without
            # duplicating an entry that is already present.
            _ldp = os.environ.get("LD_LIBRARY_PATH", "")
            if _lib_dir not in _ldp.split(os.pathsep):
                os.environ["LD_LIBRARY_PATH"] = (
                    _lib_dir + (os.pathsep + _ldp if _ldp else "")
                )
            for _so in sorted(glob.glob(os.path.join(_lib_dir, "lib*.so*"))):
                try:
                    ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        break

from modules import platform_info
platform_info.print_banner()

from modules import core

if __name__ == '__main__':
    core.run()
