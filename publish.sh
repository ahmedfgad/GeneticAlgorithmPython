#!/bin/bash
# Publish the `pygad` Python package to PyPI (or TestPyPI).
#
# This is the manual release script. Run it from anywhere, it resolves
# paths relative to its own location and uses whatever Python environment
# is currently active (installing `build` + `twine` into it if they are
# missing).
#
# Pipeline:
#   1. Check build tooling
#   2. Wipe stale dist/ artefacts (confirmation prompt)
#   3. Build sdist + wheel into dist/
#   4. `twine check` the artefacts for README/metadata issues
#   5. Prompt to upload to TestPyPI
#   6. Pause so you can `pip install -i https://test.pypi.org/simple/ pygad`
#      in a scratch venv to confirm the release works end-to-end
#   7. Prompt to upload to production PyPI (the irreversible step)
#
# All prompts default to the SAFE answer ('no' for irreversible
# actions) and require a typed 'y' to proceed.

set -euo pipefail

# Colour helpers — silently no-op when stdout isn't a TTY (e.g. piped to
# `tee` or run from a CI runner).
if [ -t 1 ]; then
  CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
  RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
else
  CYAN=''; GREEN=''; YELLOW=''; RED=''; BOLD=''; NC=''
fi

heading() { echo -e "\n${CYAN}${BOLD}== $* ==${NC}"; }
info()    { echo -e "${CYAN}$*${NC}"; }
warn()    { echo -e "${YELLOW}$*${NC}"; }
error()   { echo -e "${RED}$*${NC}" >&2; }
success() { echo -e "${GREEN}$*${NC}"; }

# Default to "no" — caller has to type `y` (case-insensitive) to confirm.
# Used for every destructive / irreversible step.
confirm() {
  local prompt="$1"
  local answer
  read -r -p "$(echo -e "${YELLOW}${prompt} [y/N]:${NC} ")" answer
  case "${answer:-}" in
    y|Y|yes|YES) return 0 ;;
    *)           return 1 ;;
  esac
}

# Resolve the script's own directory so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 1. Check build tooling ----
heading "Checking build tooling"
if ! command -v python >/dev/null 2>&1; then
  error "No 'python' on PATH. Activate a virtual environment and rerun."
  exit 1
fi
info "Active python: $(command -v python)"
info "Active pip:    $(command -v pip)"

# Confirm `build` and `twine` are present — install if missing so a
# fresh checkout works without a separate setup step.
if ! python -c "import build" >/dev/null 2>&1 || \
   ! python -c "import twine" >/dev/null 2>&1; then
  warn "Installing missing build tooling (build, twine)..."
  pip install --quiet build twine
fi

cd "$SCRIPT_DIR"

# ---- 2. Wipe stale dist/ ----
heading "Cleaning previous artefacts"
if [ -d "dist" ] && [ -n "$(ls -A dist 2>/dev/null)" ]; then
  echo "Existing dist/ contents:"
  ls -1 dist
  if confirm "Delete dist/ before building?"; then
    rm -rf dist
    success "Removed stale dist/"
  else
    warn "Keeping existing dist/. Note: twine will refuse to upload duplicates."
  fi
else
  info "No stale artefacts found."
fi

# ---- 3. Build ----
heading "Building sdist + wheel"
python -m build
ls -1 dist

# ---- 4. twine check ----
heading "Running twine check"
python -m twine check dist/*

# ---- 5. Upload to TestPyPI ----
heading "Upload to TestPyPI"
warn "TestPyPI lives at https://test.pypi.org/ and is the safe place to"
warn "verify the upload before touching production."
if confirm "Upload to TestPyPI now?"; then
  python -m twine upload --repository testpypi dist/*
  success "Uploaded to TestPyPI."
  echo
  info "Verify with (in a fresh scratch venv):"
  info "  pip install --index-url https://test.pypi.org/simple/ \\"
  info "      --extra-index-url https://pypi.org/simple/ pygad"
  echo
  read -r -p "Press Enter once the TestPyPI install looks good..."
else
  warn "Skipped TestPyPI upload."
fi

# ---- 6. Upload to production PyPI ----
heading "Upload to PRODUCTION PyPI"
warn "This step is IRREVERSIBLE. Once a version is published you cannot"
warn "re-upload the same filename — you'd have to bump the version"
warn "(pyproject.toml, setup.py, and pygad/__init__.py) and rebuild."
warn "Make sure the TestPyPI smoke test passed."
if confirm "Upload to production PyPI now?"; then
  python -m twine upload dist/*
  success "Uploaded to PyPI: https://pypi.org/project/pygad/"
else
  warn "Skipped production upload. Run this script again when ready."
fi

heading "Done"
