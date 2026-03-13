# src/data_models.py
# the fragment dataclass is the single contract between all pipeline stages.
# fields are filled sequentially: segmenter → features → matching → registration → render.
# adding a field here instantly makes it available everywhere — no interface changes.

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class fragment:
    """one document piece flowing through the entire pipeline."""

    # ── set by segmenter ──────────────────────────────────────────────────────
    id:              str             # unique stem, e.g. "scan_003"
    image_rgba:      np.ndarray      # (h, w, 4) uint8 — alpha = segmentation mask
    contour_coords:  np.ndarray      # (n, 2) float32 — equidistant resampled boundary

    # ── set by feature extractor ──────────────────────────────────────────────
    geo_features:    Optional[np.ndarray] = None   # (n, geo_dim) curvature embedding
    tex_features:    Optional[np.ndarray] = None   # (n, tex_dim) texture strip embedding
    edge_embeddings: Optional[np.ndarray] = None   # (n, embed_dim) fused eac-net output

    # ── set by matcher ────────────────────────────────────────────────────────
    match_candidates: Optional[List[Tuple[str, float]]] = None  # [(id, score)]

    # ── set by registration ───────────────────────────────────────────────────
    # pairwise rigid transforms to neighbors: {other_id: (tx, ty, theta_rad)}
    pairwise_transforms: Optional[dict] = None

    # ── set by pose graph solver ──────────────────────────────────────────────
    global_pose: Optional[Tuple[float, float, float]] = None  # (tx, ty, theta_rad)

    def __repr__(self) -> str:
        emb = self.edge_embeddings.shape if self.edge_embeddings is not None else None
        return (f"fragment(id={self.id!r}, img={self.image_rgba.shape}, "
                f"contour={len(self.contour_coords)}, emb={emb}, pose={self.global_pose})")