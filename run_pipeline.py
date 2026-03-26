#!/usr/bin/env python3
# run_pipeline.py — inference entry point.
# usage:
#   python run_pipeline.py
#   python run_pipeline.py --input data/raw --output data/result.tiff
#   python run_pipeline.py --debug
#   python run_pipeline.py --device cuda

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # windows openmp fix

import argparse
import sys
from src.pipeline import load_config, run


def main():
    p = argparse.ArgumentParser(description="document reconstruction")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--input",  default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--device", default=None, help="cpu | cuda")
    p.add_argument("--debug",  action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.input:  cfg["pipeline"]["input_dir"]    = args.input
    if args.output: cfg["pipeline"]["output_image"] = args.output

    # device is only relevant if EAC-Net is used (matching.method = "eac").
    # geometry matching has no device dependency.
    # we still accept the flag so existing scripts don't break.
    if args.device:
        try:
            import torch
            device = args.device
            if device == "cuda" and not torch.cuda.is_available():
                print("[run] CUDA requested but not available — falling back to CPU")
                device = "cpu"
            cfg.setdefault("features", {})["device"] = device
        except ImportError:
            pass   # torch not installed — geometry mode needs no device

    try:
        run(cfg, debug=args.debug)
    except Exception as e:
        print(f"\nfailed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()