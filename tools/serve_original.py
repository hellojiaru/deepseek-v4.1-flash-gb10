#!/usr/bin/env python3
"""Start a component with original checkpoint behavior and no steering assets.

Accepts tools/run.py's rank/frontend/vision arguments. This entry point never
enables activation steering, even if inherited environment variables request it.
"""
import sys
import run


def main():
    forbidden = ("--enable-steering", "--creative-file", "--creative-scope")
    if any(arg.split("=", 1)[0] in forbidden for arg in sys.argv[1:]):
        raise SystemExit("serve_original does not accept steering options; use run.py for optional steering")
    run.main()


if __name__ == "__main__":
    main()
