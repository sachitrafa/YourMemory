#!/usr/bin/env python3
"""Single-binary entry point for YourMemory.

Bundled by PyInstaller into one executable that dispatches every CLI verb, so
users who don't have Python can run:

    yourmemory                      # start the memory server (MCP + HTTP)
    yourmemory setup                # detect AI clients, install config + hooks
    yourmemory register <token>     # activate with your access token
    yourmemory ask "<question>"     # answer from memory, no LLM call
    yourmemory path                 # print the install path

It maps these verbs onto the existing functions in ``memory_mcp`` by rewriting
``sys.argv`` so each function sees the arguments it already expects.
"""
import os
import sys

# When frozen by PyInstaller, the embedding + spaCy models are bundled inside the
# binary — point the app at them and force offline so there is ZERO first-run
# download. Must run before ``memory_mcp`` (and its embed module) is imported.
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _bundled_embed = os.path.join(sys._MEIPASS, "_bundled_models", "multi-qa-mpnet-base-dot-v1")
    if os.path.isdir(_bundled_embed):
        os.environ.setdefault("YOURMEMORY_EMBED_MODEL", _bundled_embed)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Fall back to CPU if an Apple-Silicon MPS op fails, rather than crashing.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else None

    import memory_mcp as m

    if cmd == "setup":
        sys.argv = ["yourmemory", *args[1:]]
        m.setup()
    elif cmd == "register":
        # register() reads the token from sys.argv[1]
        sys.argv = ["yourmemory", *args[1:]]
        m.register()
    elif cmd in ("path", "--path"):
        sys.argv = ["yourmemory"]
        m.print_path()
    elif cmd in ("-h", "--help", "help"):
        print(__doc__)
    else:
        # Default: server. `yourmemory ask "..."` is dispatched inside run().
        sys.argv = ["yourmemory", *args]
        m.run()


if __name__ == "__main__":
    main()
