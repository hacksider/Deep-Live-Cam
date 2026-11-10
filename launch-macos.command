#!/bin/bash
set -euo pipefail

cd -- "$(dirname -- "$0")"
export PATH="/opt/homebrew/bin:$PATH"
export KERAS_BACKEND=torch
export PYTHONUNBUFFERED=1
export SSL_CERT_FILE
SSL_CERT_FILE="$(.venv/bin/python -m certifi)"

exec .venv/bin/python run.py --execution-provider coreml --nsfw-filter "$@"
