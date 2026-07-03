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
import sys


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
