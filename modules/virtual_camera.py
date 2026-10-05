from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


VIRTUAL_CAMERA_LABEL = "Deep Live Cam"
VIRTUAL_CAMERA_DEVICE_ENV = "DEEP_LIVE_CAM_VIRTUAL_DEVICE"


class VirtualCameraError(RuntimeError):
    pass


def find_virtual_camera_device(
    sys_class_root: Path | str = "/sys/class/video4linux",
    dev_root: Path | str = "/dev",
    label: str = VIRTUAL_CAMERA_LABEL,
) -> Optional[str]:
    configured = os.environ.get(VIRTUAL_CAMERA_DEVICE_ENV)
    if configured:
        return configured

    sys_class_root = Path(sys_class_root)
    dev_root = Path(dev_root)
    wanted = label.casefold()
    for name_file in sorted(sys_class_root.glob("video*/name")):
        try:
            device_name = name_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if device_name.casefold() != wanted:
            continue
        device_path = dev_root / name_file.parent.name
        if device_path.exists():
            return str(device_path)
    return None


class VirtualCameraSink:
    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        device: Optional[str] = None,
    ) -> None:
        try:
            import pyvirtualcam
        except ImportError as exc:
            raise VirtualCameraError(
                "pyvirtualcam is missing; reinstall the project requirements."
            ) from exc

        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.fps = max(1.0, float(fps))
        self.device = device or find_virtual_camera_device()
        if self.device is None:
            raise VirtualCameraError(
                "No 'Deep Live Cam' virtual camera was found. Run "
                "./setup-virtual-camera-linux.sh first."
            )

        try:
            self._camera = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.YUYV,
                device=self.device,
                print_fps=False,
            )
        except Exception as exc:
            raise VirtualCameraError(
                f"Could not open virtual camera {self.device}: {exc}"
            ) from exc

    @property
    def backend(self) -> str:
        return self._camera.backend

    def send(self, bgr_frame: np.ndarray) -> None:
        frame = bgr_frame
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(
                frame,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        yuyv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_YUYV)
        self._camera.send(yuyv_frame)

    def close(self) -> None:
        self._camera.close()

    def __enter__(self) -> "VirtualCameraSink":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
