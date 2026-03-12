# src/pipeline.py
# orchestrator — calls each phase in order, passes fragment list through.
# every stage is a pure function: (List[fragment], cfg) -> List[fragment].

import yaml
import time
from pathlib import Path

from src.segmentation.segmenter import load_fragments
from src.embeddings.encoder import embed_fragments
from src.retrieval.faiss_index import retrieve_candidates
from src.solver.matcher import solve_layout
from src.rendering.renderer import render_reconstruction, save_result


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run(cfg: dict, debug: bool = False) -> str:
    """execute the full 6-phase pipeline. returns path to output image."""
    t0 = time.perf_counter()

    frags = load_fragments(cfg["pipeline"]["input_dir"], cfg)
    if not frags:
        raise RuntimeError("no valid fragments found — check input_dir and segmentation config")

    frags = embed_fragments(frags, cfg)
    frags = retrieve_candidates(frags, cfg)
    frags = solve_layout(frags, cfg)

    if debug:
        # import here so debug deps don't slow normal runs
        from src.utils.debug_viz import (
            save_segmentation_debug, save_match_debug, save_layout_debug
        )
        dbg = cfg["pipeline"].get("debug_dir", "data/debug")
        save_segmentation_debug(frags, dbg)
        save_match_debug(frags, dbg)
        save_layout_debug(frags, dbg)

    canvas = render_reconstruction(frags, cfg)
    output_path = cfg["pipeline"]["output_image"]
    save_result(canvas, output_path)

    elapsed = time.perf_counter() - t0
    print(f"\npipeline complete in {elapsed:.1f}s -> {output_path}")
    return output_path