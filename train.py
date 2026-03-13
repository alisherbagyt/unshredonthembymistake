#!/usr/bin/env python3
# train.py — standalone eac-net training entry point.
#
# workflow:
#   1. collect 10k+ document images into data/training_docs/
#   2. python train.py --generate-data    (creates synthetic fragment pairs)
#   3. python train.py                    (trains eac-net, saves checkpoints/)
#   4. set features.weights_path in config.yaml → "checkpoints/eac_net_best.pt"
#   5. python run_pipeline.py             (now uses trained weights)

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse, json
from pathlib import Path
from src.pipeline import load_config


def main():
    p = argparse.ArgumentParser(description="train eac-net on synthetic fragment pairs")
    p.add_argument("--config",         default="configs/config.yaml")
    p.add_argument("--generate-data",  action="store_true",
                   help="run synthetic data generation first")
    p.add_argument("--device",         default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["features"]["device"] = args.device

    tc = cfg["training"]

    if args.generate_data:
        from src.training.synth_generator import generate_dataset
        print(f"[train] generating synthetic data from {tc['data_dir']} ...")
        metadata = generate_dataset(
            docs_dir  = tc["data_dir"],
            out_dir   = tc["out_dir"],
            n_frags   = tc["fragments_per_doc"],
            noise_std = tc["noise_std"],
        )
        meta_path = Path(tc["out_dir"]) / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f)
        print(f"[train] metadata saved → {meta_path} ({len(metadata)} pairs)")

    from src.training.trainer import train
    train(cfg)


if __name__ == "__main__":
    main()