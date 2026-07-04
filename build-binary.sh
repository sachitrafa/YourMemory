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
python -m spacy download en_core_web_sm   # bundled into the binary

echo "==> Building binary (downloads the embedding model once, ~420 MB)"
echo "    Result is large (~2 GB) — it bundles Python, all deps, AND both models."
pyinstaller yourmemory.spec --noconfirm --clean

deactivate
echo
echo "✓ Built: dist/yourmemory"
echo "  Smoke test:  ./dist/yourmemory path"
