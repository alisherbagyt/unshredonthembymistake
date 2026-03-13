# src/pipeline.py
# orchestrator — chains the 6 phases of the v2 pipeline in order.
# phase 1: segmentation (adaptive threshold + equidistant resampling)
# phase 2: feature extraction (eac-net: geometry + texture → contrastive embeddings)
# phase 3: candidate retrieval (faiss over mean boundary embeddings)
# phase 4: pairwise scoring (cross-attention → matched point pairs)
# phase 5: registration (ransac se(2) per candidate pair)
# phase 6: global layout (pose graph optimization)
# phase 7: rendering (alpha-blend or poisson at continuous poses)

import yaml
import time


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict, debug: bool = False) -> str:
    t0 = time.perf_counter()

    from src.segmentation.segmenter      import load_fragments
    from src.features.eac_net           import embed_fragments
    from src.matching.faiss_index       import retrieve_candidates
    from src.matching.cross_attention   import score_all_candidates
    from src.registration.ransac        import estimate_pairwise_transforms
    from src.registration.pose_graph    import solve_global_layout
    from src.rendering.renderer         import render_reconstruction, save_result

    frags = load_fragments(cfg["pipeline"]["input_dir"], cfg)
    if not frags:
        raise RuntimeError("no fragments found — check input_dir")

    frags = embed_fragments(frags, cfg)
    frags = retrieve_candidates(frags, cfg)
    frags = score_all_candidates(frags, cfg)
    frags = estimate_pairwise_transforms(frags, cfg)
    frags = solve_global_layout(frags, cfg)

    if debug:
        from src.utils.debug_viz import (
            save_segmentation_debug, save_match_debug,
            save_pose_debug, save_layout_debug,
        )
        d = cfg["pipeline"].get("debug_dir", "data/debug")
        save_segmentation_debug(frags, d)
        save_match_debug(frags, d)
        save_pose_debug(frags, d)
        save_layout_debug(frags, d)

    canvas = render_reconstruction(frags, cfg)
    out    = cfg["pipeline"]["output_image"]
    save_result(canvas, out)

    print(f"\npipeline complete in {time.perf_counter()-t0:.1f}s → {out}")
    return out