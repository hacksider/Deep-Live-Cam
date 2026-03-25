#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python não encontrado: $PYTHON_BIN"
  echo "Instale com: brew install python@3.11"
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Evita conflito comum de webcam no macOS.
python -m pip uninstall -y opencv-python-headless || true
python -m pip install --upgrade "numpy<2" "opencv-python==4.10.0.84"

echo
echo "Ambiente macOS configurado com sucesso."
echo "Ative com: source $VENV_DIR/bin/activate"
echo "Execute com: python run.py"
