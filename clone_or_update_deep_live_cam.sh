#!/bin/zsh
# clone_or_update_deep_live_cam.sh - Clone or update Deep-Live-Cam repo in a separate folder (macOS/Linux)
REPO_URL="https://github.com/hacksider/Deep-Live-Cam.git"
TARGET_DIR="Deep-Live-Cam-remote"

if [ -d "$TARGET_DIR" ]; then
  echo "Updating existing repo in $TARGET_DIR ..."
  cd "$TARGET_DIR"
  git pull
  cd ..
else
  echo "Cloning repo to $TARGET_DIR ..."
  git clone "$REPO_URL" "$TARGET_DIR"
fi

# Sync updated code to local working folder (excluding .git and models)
LOCAL_DIR="Deep-Live-Cam"
rsync -av --exclude='.git' --exclude='models' --exclude='*.pth' --exclude='*.onnx' "$TARGET_DIR"/ "$LOCAL_DIR"/

echo "Done. Latest code is in $LOCAL_DIR."
