#!/usr/bin/env python3
# run_interactive.py — prompt-driven CLI entry point.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # windows openmp fix

import sys
from src.interactive_cli import run_interactive


def main():
    try:
        run_interactive()
    except Exception as e:
        print(f"\nfailed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
