#!/bin/zsh
# push_to_new_branch.sh - Commit and push changes to a new branch in Deep-Live-Cam-remote

REPO_DIR="Deep-Live-Cam-remote"
BRANCH_NAME="feature-$(date +%Y%m%d-%H%M%S)"

if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Error: $REPO_DIR is not a git repository. Run the clone_or_update_deep_live_cam.sh script first."
  exit 1
fi

cd "$REPO_DIR"
git add .
echo "Enter a commit message: "
read COMMIT_MSG
git commit -m "$COMMIT_MSG"
git checkout -b "$BRANCH_NAME"
git push origin "$BRANCH_NAME"
echo "Pushed to branch $BRANCH_NAME on remote."
