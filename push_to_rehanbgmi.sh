#!/bin/zsh
# push_to_rehanbgmi.sh - Commit and push changes to your fork (rehanbgmi/deeplivceam) in Deep-Live-Cam-remote

REPO_DIR="Deep-Live-Cam-remote"
FORK_URL="https://github.com/rehanbgmi/deeplivceam.git"
BRANCH_NAME="feature-$(date +%Y%m%d-%H%M%S)"

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Error: $REPO_DIR is not a git repository. Run the clone_or_update_deep_live_cam.sh script first."
  exit 1
fi

cd "$REPO_DIR"
# Set your fork as a remote if not already set
git remote | grep rehanbgmi > /dev/null || git remote add rehanbgmi "$FORK_URL"

git add .
echo "Enter a commit message: "
read COMMIT_MSG
git commit -m "$COMMIT_MSG"
git checkout -b "$BRANCH_NAME"
git push rehanbgmi "$BRANCH_NAME"
echo "Pushed to branch $BRANCH_NAME on your fork (rehanbgmi/deeplivceam)."
