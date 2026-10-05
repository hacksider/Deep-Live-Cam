#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if command -v omarchy >/dev/null 2>&1; then
    omarchy pkg add v4l2loopback-dkms v4l2loopback-utils
elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --needed v4l2loopback-dkms v4l2loopback-utils
elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y v4l2loopback-dkms v4l2loopback-utils
else
    echo "Unsupported distribution: install v4l2loopback and its utilities first." >&2
    exit 1
fi

sudo install -Dm644 \
    "$project_dir/linux/deep-live-cam-v4l2loopback.conf" \
    /etc/modprobe.d/deep-live-cam-v4l2loopback.conf
sudo install -Dm644 \
    "$project_dir/linux/deep-live-cam.modules-load.conf" \
    /etc/modules-load.d/deep-live-cam.conf

if [[ -e /dev/video10 ]]; then
    device_name="$(< /sys/class/video4linux/video10/name)"
    if [[ "$device_name" != "Deep Live Cam" ]]; then
        echo "/dev/video10 is already used by $device_name." >&2
        exit 1
    fi
else
    if [[ -d /sys/module/v4l2loopback ]]; then
        echo "v4l2loopback is already loaded with different options." >&2
        echo "Close programs using virtual cameras, then reboot or reload the module." >&2
        exit 1
    fi
    sudo modprobe v4l2loopback
fi

echo "Deep Live Cam virtual camera ready at /dev/video10."
