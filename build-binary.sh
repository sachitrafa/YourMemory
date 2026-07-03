#!/usr/bin/env bash
# Build a standalone YourMemory binary for the CURRENT platform.
#
#   ./build-binary.sh
#
# Produces:  dist/yourmemory   (single self-contained executable)
#
# The binary bundles Python + all deps, so end users need no Python install.
# It is large (~1–2 GB) because it includes torch/sentence-transformers.
# For multi-platform release binaries, use the GitHub Actions workflow
# (.github/workflows/build-binary.yml) which builds macOS/Linux/Windows.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "==> Installing build deps into a clean venv (keeps your system Python clean)"
python3 -m venv .build-venv
# shellcheck disable=SC1091
source .build-venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -e .            # the app + all runtime deps
pip install --quiet pyinstaller

echo "==> Building binary (this takes a few minutes and needs a few GB free)"
pyinstaller yourmemory.spec --noconfirm --clean

deactivate
echo
echo "✓ Built: dist/yourmemory"
echo "  Smoke test:  ./dist/yourmemory path"
