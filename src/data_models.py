# src/data_models.py
# the fragment dataclass is the contract between all pipeline stages.
# every module receives and returns fragment objects — never raw tensors or dicts.
# adding new fields here is the only change needed to extend the pipeline.

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class fragment:
    """single document fragment — flows through the entire pipeline unchanged."""

    id: str                                    # unique filename stem, e.g. "piece_001"
    image_rgba: np.ndarray                     # (h, w, 4) uint8 — deskewed, alpha-masked
    contour_coords: np.ndarray                 # (n, 2) float32 edge points in pixel space
    edge_embeddings: Optional[np.ndarray] = None   # (e, d) float32 — set by encoder stage
    global_pose: Optional[tuple] = None        # (x, y, theta) — set by solver stage
    match_candidates: Optional[list] = None    # list of (fragment_id, score) — set by retrieval

    def __repr__(self):
        emb_shape = self.edge_embeddings.shape if self.edge_embeddings is not None else None
        return (
            f"fragment(id={self.id!r}, "
            f"image={self.image_rgba.shape}, "
            f"contour_pts={len(self.contour_coords)}, "
            f"embeddings={emb_shape}, "
            f"pose={self.global_pose})"
        )
