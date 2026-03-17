#!/usr/bin/env python3
# train.py — eac-net training entry point.
#
# workflow:
#   1. place DocLayNet dataset at data/DocLayNet/ (or set --doclayet-root)
#   2. python train.py --generate-data       (creates synthetic fragment pairs)
#   3. python train.py                       (trains eac-net, saves checkpoints/)
#   4. set features.weights_path in config.yaml → "checkpoints/eac_net_best.pt"
#   5. python run_pipeline.py                (now uses trained weights)
#
# --generate-data will auto-detect whether doclayet-root contains train/val/test
# subdirectories (DocLayNet layout) or flat images (legacy layout) and call the
# appropriate generator function.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse, json
from pathlib import Path
from src.pipeline import load_config


def main():
    p = argparse.ArgumentParser(description="train eac-net on synthetic fragment pairs")
    p.add_argument("--config",          default="configs/config.yaml")
    p.add_argument("--generate-data",   action="store_true",
                   help="run synthetic data generation first")
    p.add_argument("--device",          default=None)
    p.add_argument("--doclayet-root",   default=None,
                   help="path to DocLayNet root (overrides training.data_dir). "
                        "if the root contains train/val/test subdirs, split "
                        "generation is used; otherwise falls back to flat mode.")
    p.add_argument("--train-budget",    type=int, default=4000,
                   help="minimum number of pairs to generate for train split")
    p.add_argument("--val-budget",      type=int, default=1500,
                   help="minimum number of pairs to generate for val split")
    p.add_argument("--test-budget",     type=int, default=1500,
                   help="minimum number of pairs to generate for test split")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["features"]["device"] = args.device

    tc = cfg["training"]

    if args.generate_data:
        from src.training.synth_generator import generate_dataset, generate_splits

        data_root = args.doclayet_root or tc.get("data_dir", "data/training_docs")
        out_dir   = tc.get("out_dir", "data/training_fragments")
        n_frags   = tc.get("fragments_per_doc", 6)
        noise_std = tc.get("noise_std", 5.0)

        data_root_path = Path(data_root)

        # detect DocLayNet layout: has at least one of train/val/test subdirs
        has_splits = any(
            (data_root_path / s).exists()
            for s in ("train", "val", "test")
        )

        if has_splits:
            print(f"[train] DocLayNet layout detected at {data_root}")
            print(f"[train] budgets — train:{args.train_budget}  "
                  f"val:{args.val_budget}  test:{args.test_budget}")
            all_meta = generate_splits(
                doclayet_root  = data_root,
                out_dir        = out_dir,
                split_budgets  = {
                    "train": args.train_budget,
                    "val":   args.val_budget,
                    "test":  args.test_budget,
                },
                n_frags        = n_frags,
                noise_std      = noise_std,
            )
            # trainer.py reads out_dir/train/metadata.json by convention;
            # also write a top-level metadata.json with all pairs for
            # backwards compatibility with any code that reads tc["out_dir"]/metadata.json
            combined = []
            for split_meta in all_meta.values():
                combined.extend(split_meta)
            top_meta_path = Path(out_dir) / "metadata.json"
            with open(top_meta_path, "w") as f:
                json.dump(combined, f, indent=2)
            print(f"[train] combined metadata ({len(combined)} pairs) → {top_meta_path}")
            total = sum(len(v) for v in all_meta.values())
            print(f"[train] total pairs generated: {total}")

        else:
            # legacy flat-directory mode
            print(f"[train] flat image directory detected at {data_root}")
            print(f"[train] generating synthetic data ...")
            metadata = generate_dataset(
                docs_dir  = data_root,
                out_dir   = out_dir,
                n_frags   = n_frags,
                noise_std = noise_std,
            )
            meta_path = Path(out_dir) / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"[train] metadata saved → {meta_path} ({len(metadata)} pairs)")

    from src.training.trainer import train
    train(cfg)


if __name__ == "__main__":
    main()