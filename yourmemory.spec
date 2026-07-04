# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone YourMemory binary.

Build:  pyinstaller yourmemory.spec --noconfirm
Output: dist/yourmemory   (single self-contained executable)

The binary bundles Python + every dependency (torch, sentence-transformers,
spaCy, DuckDB, FastAPI/uvicorn, MCP, …), the app's data files (hook templates
and SQL schemas), AND both ML models (the embedding model + spaCy en_core_web_sm).

It is therefore fully self-contained and works offline on first run — nothing is
downloaded. The trade-off is size (~2 GB): the embedding model is ~420 MB and
torch is the bulk of the rest.
"""
import glob
import os

from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas, binaries, hiddenimports = [], [], []

# Heavy / native packages PyInstaller can't fully trace on its own.
for pkg in (
    "torch",
    "sentence_transformers",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "sklearn",
    "scipy",
    "spacy",
    "thinc",
    "srsly",
    "catalogue",
    "cymem",
    "preshed",
    "blis",
    "wasabi",
    "duckdb",
    "networkx",
    "mcp",
    "uvicorn",
    "fastapi",
    "starlette",
    "pydantic",
    "apscheduler",
    "dateutil",
    "dotenv",
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # a missing optional dep shouldn't abort the build
        print(f"[spec] skip collect_all({pkg}): {exc}")

# --- Bundle the ML models so the binary is fully self-contained (no download) ---

# spaCy model (en_core_web_sm) — used for extraction / dedup / entity graph.
try:
    d, b, h = collect_all("en_core_web_sm")
    datas += d
    binaries += b
    hiddenimports += h
except Exception as exc:
    print(f"[spec] en_core_web_sm not collected (install it first): {exc}")

# Embedding model (sentence-transformers) — download once into _bundled_models/
# and ship it inside the binary. Loaded at runtime via YOURMEMORY_EMBED_MODEL.
EMBED_MODEL = "multi-qa-mpnet-base-dot-v1"
_model_dir = os.path.join("_bundled_models", EMBED_MODEL)
if not os.path.isdir(_model_dir):
    print(f"[spec] downloading embedding model {EMBED_MODEL} (~420 MB, one-time) ...")
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(EMBED_MODEL).save(_model_dir)
datas += [(_model_dir, os.path.join("_bundled_models", EMBED_MODEL))]

# App data files (setup() loads these via importlib.resources).
datas += [("src/hook_templates", "src/hook_templates")]
for sql in glob.glob("src/**/*.sql", recursive=True):
    datas.append((sql, os.path.dirname(sql)))

# uvicorn/starlette pull a few things in dynamically.
hiddenimports += [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "src.app",
]

a = Analysis(
    ["binary_entry.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide6", "IPython", "notebook"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="yourmemory",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
