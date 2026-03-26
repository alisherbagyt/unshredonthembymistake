# src/features/geometry.py
# computes rotation-invariant geometric features along a resampled contour.
#
# features per point (5 total):
#   col 0 — signed curvature κ(i): turning angle over a local window, normalized
#            to [-1, 1]. rotation-invariant scalar — the primary edge-shape signal.
#   col 1 — normalized arc-length position [0, 1]: where on the perimeter this
#            point sits. gives the model spatial context along the boundary.
#   col 2 — tangent x  (cos θ_tangent): local edge direction, x component.
#   col 3 — tangent y  (sin θ_tangent): local edge direction, y component.
#   col 4 — normal y   (-sin θ_tangent): inward normal y component.
#            FIX: was abs(tangent_x), which was a duplicate of col 2 with no
#            new information. the inward normal direction is the actual signal
#            needed: it tells the texture branch which way is "into" the fragment,
#            completing the (tangent, normal) local frame at each contour point.
#
# together, cols 2-4 give the model a full oriented local frame: tangent (2,3)
# + normal y (4). normal x = tangent y by construction (90° rotation), so it
# is redundant and omitted to keep the vector at 5 dims.

import numpy as np


def compute_geometric_features(contour: np.ndarray, window: int = 5) -> np.ndarray:
    """compute per-point geometric feature vector.

    args:
        contour: (n, 2) float32 equidistant contour
        window:  half-window for finite difference curvature estimate

    returns:
        (n, 5) float32 — [curvature, arc_position, tangent_x, tangent_y, normal_y]
    """
    n = len(contour)
    w = window

    # ── tangent vectors via central finite difference ─────────────────────────
    # roll handles wraparound at the closed contour boundary
    prev_pts = np.roll(contour,  w, axis=0)
    next_pts = np.roll(contour, -w, axis=0)
    tangents = next_pts - prev_pts                       # (n, 2) unnormalized
    t_len    = np.linalg.norm(tangents, axis=1, keepdims=True).clip(1e-8)
    tangents = tangents / t_len                          # unit tangents (n, 2)

    # inward normal: rotate tangent 90° clockwise → (ty, -tx)
    # normal_x =  tangents[:, 1]   (= tangent_y, redundant, not stored)
    # normal_y = -tangents[:, 0]   (independent signal, stored as col 4)
    normal_y = -tangents[:, 0]                           # (n,)

    # ── signed curvature via cross product of successive tangents ─────────────
    # κ > 0 = left turn (convex), κ < 0 = right turn (concave / corner)
    t_prev    = np.roll(tangents,  1, axis=0)
    t_next    = np.roll(tangents, -1, axis=0)
    cross     = t_prev[:, 0] * t_next[:, 1] - t_prev[:, 1] * t_next[:, 0]
    curvature = np.arcsin(cross.clip(-1, 1))             # (n,) radians

    # ── normalized arc-length position ────────────────────────────────────────
    seg_len = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)
    arc_pos = np.cumsum(seg_len) / (seg_len.sum() + 1e-8)   # (n,) in [0, 1]

    # ── assemble feature matrix ───────────────────────────────────────────────
    feats = np.column_stack([
        curvature,          # col 0 — signed curvature
        arc_pos,            # col 1 — perimeter position
        tangents[:, 0],     # col 2 — tangent x
        tangents[:, 1],     # col 3 — tangent y
        normal_y,           # col 4 — inward normal y  ← FIX (was abs(tangent_x))
    ]).astype(np.float32)

    # normalize curvature to [-1, 1] for stable training
    c_max = np.abs(feats[:, 0]).max()
    if c_max > 1e-6:
        feats[:, 0] /= c_max

    return feats   # (n, 5)


def compute_geometric_features_batch(frags: list) -> list:
    """compute and store geo_features for every fragment in the list.
    called by pipeline.py as a lightweight phase 3 when EAC-Net is not used.
    """
    for frag in frags:
        frag.geo_features = compute_geometric_features(frag.contour_coords)
    print(f"[geometry] computed curvature features for {len(frags)} fragments")
    return frags