#!/usr/bin/env python3
# run_pipeline.py — inference entry point.
# usage:
#   python run_pipeline.py
#   python run_pipeline.py --input data/raw --output data/result.tiff
#   python run_pipeline.py --debug
#   python run_pipeline.py --device cuda

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # windows openmp fix

import argparse, sys
from src.pipeline import load_config, run


def main():
    p = argparse.ArgumentParser(description="shred_net v2 — document reconstruction")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--input",  default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--device", default=None, help="cpu | cuda")
    p.add_argument("--debug",  action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.input:  cfg["pipeline"]["input_dir"]    = args.input
    if args.output: cfg["pipeline"]["output_image"] = args.output
    if args.device: cfg["features"]["device"]       = args.device

    try:
        run(cfg, debug=args.debug)
    except Exception as e:
        print(f"\nfailed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()