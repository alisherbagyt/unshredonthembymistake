# src/data_models.py
# the Fragment dataclass is the single contract between all pipeline stages.
# fields are filled sequentially: segmenter → features → matching → registration → render.

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np


@dataclass
class fragment:
    """one document piece flowing through the entire pipeline."""

    # ── set by segmenter ──────────────────────────────────────────────────────
    id:              str
    image_rgba:      np.ndarray      # (H, W, 4) uint8 — alpha = segmentation mask
    contour_coords:  np.ndarray      # (N, 2) float32 — equidistant resampled, LOCAL coords

    # crop_offset: (x0, y0) of this fragment's crop within the original scan image.
    # contour_coords are in local crop space (origin at top-left of crop).
    # to convert local contour point (px, py) to scan space: (px+x0, py+y0).
    # essential for computing correct inter-fragment placement on the canvas.
    crop_offset:     Tuple[int, int] = (0, 0)

    # ── set by feature extractor ──────────────────────────────────────────────
    geo_features:    Optional[np.ndarray] = None   # (N, 5) curvature + tangent
    tex_features:    Optional[np.ndarray] = None   # (N, SW, 3) texture strip
    edge_embeddings: Optional[np.ndarray] = None   # (N, embed_dim) eac-net output

    # ── set by rotation stage ─────────────────────────────────────────────────
    canonical_theta: Optional[float] = None   # upright correction (radians)
    piece_type:      Optional[str]   = None   # "corner" | "edge" | "interior"

    # ── set by matcher ────────────────────────────────────────────────────────
    match_candidates: Optional[List[Tuple[str, float]]] = None
    matched_points:   Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None

    # ── set by registration ───────────────────────────────────────────────────
    pairwise_transforms: Optional[Dict[str, Tuple[float, float, float, int]]] = None

    # ── set by pose graph solver ──────────────────────────────────────────────
    global_pose: Optional[Tuple[float, float, float]] = None   # (tx, ty, theta_rad)

    def __repr__(self) -> str:
        return (f"fragment(id={self.id!r}, crop_offset={self.crop_offset}, "
                f"img={self.image_rgba.shape}, contour={len(self.contour_coords)}, "
                f"pose={self.global_pose})")