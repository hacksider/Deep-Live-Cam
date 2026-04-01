#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

export NO_ALBUMENTATIONS_UPDATE=1

if [ -x "./venv/bin/python" ]; then
  PYTHON="./venv/bin/python"
elif [ -x "./.venv/bin/python" ]; then
  PYTHON="./.venv/bin/python"
elif [ -x "/opt/homebrew/bin/python3.10" ]; then
  PYTHON="/opt/homebrew/bin/python3.10"
else
  PYTHON="$(command -v python3)"
fi

if [ -z "${PYTHON:-}" ] || [ ! -x "$PYTHON" ]; then
  echo "No usable Python 3.10 interpreter found."
  exit 1
fi

HAS_EXECUTION_PROVIDER=0
for arg in "$@"; do
  if [[ "$arg" == "--execution-provider" ]]; then
    HAS_EXECUTION_PROVIDER=1
    break
  fi
done

if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" && $HAS_EXECUTION_PROVIDER -eq 0 ]]; then
  exec "$PYTHON" run.py --execution-provider coreml cpu "$@"
else
  exec "$PYTHON" run.py "$@"
fi
