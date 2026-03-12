#!/usr/bin/env python3
# run_pipeline.py — cli entry point.
# usage:
#   python run_pipeline.py
#   python run_pipeline.py --input data/raw --output data/out.tiff
#   python run_pipeline.py --debug        <- saves debug images to data/debug/

import os
# windows: pytorch and faiss each ship their own openmp runtime.
# this must be set before any import that loads either library.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from src.pipeline import load_config, run


def parse_args():
    p = argparse.ArgumentParser(description="shred_net: document fragment reconstruction")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--input",  default=None, help="override pipeline.input_dir")
    p.add_argument("--output", default=None, help="override pipeline.output_image")
    p.add_argument("--device", default=None, help="cpu | cuda")
    p.add_argument("--debug",  action="store_true",
                   help="save segmentation crops, match scores, and layout map to data/debug/")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.input:  cfg["pipeline"]["input_dir"]    = args.input
    if args.output: cfg["pipeline"]["output_image"] = args.output
    if args.device: cfg["embeddings"]["device"]     = args.device

    try:
        run(cfg, debug=args.debug)
    except Exception as e:
        print(f"\npipeline failed: {e}", file=sys.stderr)
        raise   # show full traceback in debug, not just message
        sys.exit(1)


if __name__ == "__main__":
    main()