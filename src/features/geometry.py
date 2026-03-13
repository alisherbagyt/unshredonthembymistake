# src/features/geometry.py
# computes rotation-invariant geometric features along a resampled contour.
#
# critique fix: instead of the brittle pca-on-ink-pixels orientation trick,
# we describe each contour point using LOCAL geometric properties that are
# intrinsically invariant to global rotation.
#
# features per point:
#   - signed curvature   κ(i) = turning angle over a local window
#   - arc-length         normalized position along the perimeter [0, 1]
#   - unit tangent       (cos θ, sin θ) — encodes local orientation
#   - normal direction   (−sin θ, cos θ) — for orienting texture strips
#
# these features feed the geometry branch of eac-net.

import numpy as np


def compute_geometric_features(contour: np.ndarray, window: int = 5) -> np.ndarray:
    """compute per-point geometric feature vector (rotation-invariant).

    args:
        contour: (n, 2) float32 equidistant contour
        window:  half-window for finite difference curvature estimate

    returns:
        (n, 5) float32 — [curvature, arc_position, tangent_x, tangent_y, |normal_x|]
        each row is the feature vector for one contour point.
    """
    n = len(contour)
    w = window

    # ── tangent vectors via central finite difference ─────────────────────────
    # roll handles wraparound at the closed contour boundary
    prev_pts = np.roll(contour,  w, axis=0)
    next_pts = np.roll(contour, -w, axis=0)
    tangents = next_pts - prev_pts                      # (n, 2) unnormalized
    t_len    = np.linalg.norm(tangents, axis=1, keepdims=True).clip(1e-8)
    tangents = tangents / t_len                          # unit tangents

    # ── signed curvature via cross product of successive tangents ─────────────
    # κ = (t × t') / |Δs|  — positive = left turn, negative = right turn
    t_prev   = np.roll(tangents,  1, axis=0)
    t_next   = np.roll(tangents, -1, axis=0)
    # 2d cross product: t_prev × t_next = tx * ny - ty * nx
    cross    = t_prev[:, 0] * t_next[:, 1] - t_prev[:, 1] * t_next[:, 0]
    curvature = np.arcsin(cross.clip(-1, 1))             # (n,) in radians

    # ── normalized arc-length position ────────────────────────────────────────
    seg_len  = np.linalg.norm(np.diff(contour, axis=0, append=contour[:1]), axis=1)
    arc_pos  = np.cumsum(seg_len) / (seg_len.sum() + 1e-8)  # (n,) in [0, 1]

    # ── assemble feature matrix ───────────────────────────────────────────────
    feats = np.column_stack([
        curvature,          # col 0 — signed curvature (rotation-invariant scalar)
        arc_pos,            # col 1 — perimeter position
        tangents[:, 0],     # col 2 — tangent x
        tangents[:, 1],     # col 3 — tangent y
        np.abs(tangents[:, 0]),  # col 4 — |normal projection| (symmetry feature)
    ]).astype(np.float32)

    # normalize curvature to [-1, 1] for stable training
    c_max = np.abs(feats[:, 0]).max()
    if c_max > 1e-6:
        feats[:, 0] /= c_max

    return feats   # (n, 5)