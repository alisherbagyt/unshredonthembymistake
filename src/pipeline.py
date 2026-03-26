# src/pipeline.py
# orchestrator — 6 stages, geometry-based matching (no EAC-Net in matching path).
#
# phase 1: segmentation      — detect fragment blobs, resample contours
# phase 2: geometric features — curvature sequences for the matcher
# phase 3: matching          — curvature cross-correlation, pure numpy
#                              point correspondences in centroid-relative coords
# phase 4: registration      — ransac SE(2) + pose graph optimization
#                              poses are centroid-relative after this stage
# phase 5: pose conversion   — convert centroid-relative poses to image-space
# phase 6: rendering         — alpha-blend at image-space poses -> .tiff

import yaml
import time


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict, debug: bool = False) -> str:
    t0 = time.perf_counter()

    from src.segmentation.segmenter        import load_fragments
    # from src.features.geometry             import compute_geometric_features_batch
    from src.matching.geometry_matcher     import match_fragments
    from src.registration.ransac           import estimate_pairwise_transforms
    from src.registration.pose_graph       import solve_global_layout, convert_poses_to_image_space
    from src.rendering.renderer            import render_reconstruction, save_result

    # phase 1: segmentation
    frags = load_fragments(cfg["pipeline"]["input_dir"], cfg)
    if not frags:
        raise RuntimeError("no fragments found — check input_dir")

    # # phase 2: geometric features (curvature sequences)
    # frags = compute_geometric_features_batch(frags)

    # phase 3: geometry-based matching
    # matched_points are in centroid-relative coordinates
    frags = match_fragments(frags, cfg)

    # phase 4: registration
    # pairwise_transforms and global_pose are in centroid-relative space
    frags = estimate_pairwise_transforms(frags, cfg)
    frags = solve_global_layout(frags, cfg)

    # phase 5: convert centroid-relative poses to image-pixel-space for renderer
    frags = convert_poses_to_image_space(frags)

    # optional debug output
    if debug:
        from src.utils.debug_viz import save_debug_outputs
        save_debug_outputs(frags, cfg["pipeline"].get("debug_dir", "data/debug"))

    # phase 6: rendering
    canvas = render_reconstruction(frags, cfg)
    out    = cfg["pipeline"]["output_image"]
    save_result(canvas, out)

    print(f"\npipeline complete in {time.perf_counter() - t0:.1f}s -> {out}")
    return out