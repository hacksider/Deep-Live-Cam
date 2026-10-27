FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    QT_QPA_PLATFORM=offscreen

WORKDIR /app

# Install system dependencies for OpenCV, Qt/PySide6,
# InsightFace, ONNX Runtime, and image/video processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    libglx0 \
    libxcb1 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libice6 \
    libfontconfig1 \
    libdbus-1-3 \
    libegl1 \
    libopengl0 \
    libgl1-mesa-dri \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer-plugins-good1.0-0 \
    libv4l-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to take advantage of Docker layer caching.
COPY requirements.txt /app/requirements.txt

# Install Python dependencies.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements.txt

# Copy the application source code.
COPY . /app

# Create model directory.
RUN mkdir -p /app/models

# Railway provides PORT when applicable.
# Bind the application to all interfaces.
EXPOSE 7860

CMD ["python", "run.py"]
