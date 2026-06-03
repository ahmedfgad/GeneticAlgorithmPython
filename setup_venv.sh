#!/usr/bin/env bash
# Set up a virtual environment for PyGAD.
#
# Usage:
#   ./setup_venv.sh
#   PYTHON=python3.12 ./setup_venv.sh

set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR=".venv"

cd "$(dirname "$0")"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Error: $PYTHON not found." >&2
    exit 1
fi

if [ -d "$VENV_DIR" ]; then
    echo "$VENV_DIR already exists. Delete it first to rebuild: rm -rf $VENV_DIR"
else
    echo "Creating $VENV_DIR with $PYTHON ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e ".[visualize]"
python -m pip install pytest

echo ""
echo "Done. Activate with: source $VENV_DIR/bin/activate"
