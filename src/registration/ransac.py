# src/registration/ransac.py
# ransac-based rigid pose estimation in se(2).
#
# critique fix: replaces the axis-aligned snap (4 discrete orientations) with
# a continuous rigid transform T ∈ SE(2) = (tx, ty, theta).
#
# for each high-scoring candidate pair (a, b):
#   1. take the matched point correspondences from cross_attention
#   2. run ransac: repeatedly sample 2 point pairs → solve minimal transform
#   3. count inliers (points that agree with this transform within threshold)
#   4. refine on all inliers via least-squares
#
# the result is an exact (tx, ty, theta) that aligns fragment b onto fragment a.

import numpy as np
from typing import Tuple, Optional, List
from src.data_models import fragment


# ── public api ────────────────────────────────────────────────────────────────

def estimate_pairwise_transforms(frags: List[fragment], cfg: dict) -> List[fragment]:
    """run ransac for every scored candidate pair.
    populates frag.pairwise_transforms = {other_id: (tx, ty, theta_rad, n_inliers)}."""
    rc      = cfg["registration"]
    n_iter  = rc["ransac_iters"]
    thr     = rc["ransac_inlier_thr"]
    min_inl = rc["min_inliers"]

    frag_map = {f.id: f for f in frags}

    for frag in frags:
        frag.pairwise_transforms = {}
        if not frag.match_candidates:
            continue

        match_pts = getattr(frag, "_match_points", {})

        for cand_id, score in frag.match_candidates:
            if cand_id not in match_pts:
                continue
            pts_a, pts_b = match_pts[cand_id]
            if len(pts_a) < 2:
                continue

            result = _ransac_se2(pts_a, pts_b, n_iter, thr, min_inl)
            if result is not None:
                tx, ty, theta, n_inliers = result
                frag.pairwise_transforms[cand_id] = (tx, ty, theta, n_inliers)

    n_edges = sum(len(f.pairwise_transforms) for f in frags)
    print(f"[ransac] found {n_edges} valid pairwise transforms")
    return frags


# ── ransac core ───────────────────────────────────────────────────────────────

def _ransac_se2(
    pts_a:   np.ndarray,   # (k, 2) source points
    pts_b:   np.ndarray,   # (k, 2) target points (should map to pts_a after transform)
    n_iter:  int,
    thr:     float,
    min_inl: int,
    rng:     np.random.Generator = np.random.default_rng(0),
) -> Optional[Tuple[float, float, float, int]]:
    """ransac for se(2) rigid transform aligning pts_b onto pts_a.

    the minimal sample is 2 point pairs — sufficient to uniquely determine
    (tx, ty, theta) in 2d rigid motion.

    returns (tx, ty, theta_rad, n_inliers) or None if not enough inliers.
    """
    k = len(pts_a)
    if k < 2:
        return None

    best_inliers = 0
    best_tx, best_ty, best_theta = 0.0, 0.0, 0.0

    for _ in range(n_iter):
        # sample 2 random correspondence pairs (minimal sample for se2)
        idx = rng.choice(k, size=2, replace=False)
        tx, ty, theta = _solve_se2_minimal(pts_b[idx], pts_a[idx])

        # count inliers: points whose residual is below threshold
        pts_b_transformed = _apply_se2(pts_b, tx, ty, theta)
        residuals         = np.linalg.norm(pts_b_transformed - pts_a, axis=1)
        inlier_mask       = residuals < thr
        n_inliers         = inlier_mask.sum()

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_tx, best_ty, best_theta = tx, ty, theta

    if best_inliers < min_inl:
        return None

    # ── refinement: least-squares on all inliers ──────────────────────────────
    pts_b_t  = _apply_se2(pts_b, best_tx, best_ty, best_theta)
    residuals = np.linalg.norm(pts_b_t - pts_a, axis=1)
    inliers   = residuals < thr

    if inliers.sum() < 2:
        return None

    tx_r, ty_r, theta_r = _solve_se2_lstsq(pts_b[inliers], pts_a[inliers])
    return float(tx_r), float(ty_r), float(theta_r), int(inliers.sum())


def _solve_se2_minimal(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float, float]:
    """solve se(2) from exactly 2 point correspondences.
    closed-form: theta from the angle between displacement vectors."""
    d_src = src[1] - src[0]
    d_dst = dst[1] - dst[0]

    theta = np.arctan2(
        d_dst[0] * d_src[1] - d_dst[1] * d_src[0],   # cross product
        d_dst[0] * d_src[0] + d_dst[1] * d_src[1],   # dot product
    )

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # translation: t = mean(dst) - R @ mean(src)
    t = dst.mean(axis=0) - R @ src.mean(axis=0)
    return float(t[0]), float(t[1]), float(theta)


def _solve_se2_lstsq(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float, float]:
    """least-squares se(2) from n >= 2 correspondences via svd (kabsch algorithm)."""
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)

    H = (src - c_src).T @ (dst - c_dst)   # 2×2 cross-covariance
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # handle reflection case (det should be +1 for proper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    theta = np.arctan2(R[1, 0], R[0, 0])
    t     = c_dst - R @ c_src
    return float(t[0]), float(t[1]), float(theta)


def _apply_se2(pts: np.ndarray, tx: float, ty: float, theta: float) -> np.ndarray:
    """apply rigid transform (tx, ty, theta) to point array (n, 2)."""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return (R @ pts.T).T + np.array([tx, ty])