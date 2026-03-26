# src/registration/ransac.py
# ransac-based rigid pose estimation in SE(2).
#
# for each high-scoring candidate pair (a, b):
#   1. read matched point correspondences from fragment.matched_points (dataclass field)
#   2. ransac: repeatedly sample 2 point pairs → solve minimal SE(2) transform
#   3. count inliers (points within threshold of the proposed transform)
#   4. refine on all inliers via least-squares (Kabsch SVD)
#
# rng fix: the old signature had `rng = np.random.default_rng(0)` as a default
# argument. python evaluates default arguments ONCE at function definition time,
# so the same rng object (with continuously advancing state) was shared across
# every call in the entire run. pairs processed later in the run saw a completely
# different random sequence than pairs processed first — deterministic but biased.
# fix: create a fresh rng inside the function body on every call.

import numpy as np
from typing import Tuple, Optional, List
from src.data_models import fragment


# ── public api ────────────────────────────────────────────────────────────────

def estimate_pairwise_transforms(frags: List[fragment], cfg: dict) -> List[fragment]:
    """run ransac for every scored candidate pair.
    reads point correspondences from fragment.matched_points (declared dataclass field).
    populates fragment.pairwise_transforms = {other_id: (tx, ty, theta_rad, n_inliers)}.
    """
    rc      = cfg["registration"]
    n_iter  = rc["ransac_iters"]
    thr     = rc["ransac_inlier_thr"]
    min_inl = rc["min_inliers"]

    for frag in frags:
        frag.pairwise_transforms = {}

        if not frag.match_candidates:
            continue

        # matched_points is the declared dataclass field populated by cross_attention
        match_pts = frag.matched_points or {}

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
    pts_a:   np.ndarray,   # (k, 2) target points
    pts_b:   np.ndarray,   # (k, 2) source points (mapped onto pts_a)
    n_iter:  int,
    thr:     float,
    min_inl: int,
) -> Optional[Tuple[float, float, float, int]]:
    """ransac for SE(2) rigid transform aligning pts_b → pts_a.

    minimal sample = 2 point pairs (sufficient to solve tx, ty, theta in 2D).

    rng is created fresh on every call — no shared state between pairs.

    returns (tx, ty, theta_rad, n_inliers) or None if consensus is too small.
    """
    k = len(pts_a)
    if k < 2:
        return None

    # fresh rng per call — no shared state, no sequence bias across pairs
    rng = np.random.default_rng()

    best_inliers = 0
    best_tx, best_ty, best_theta = 0.0, 0.0, 0.0

    for _ in range(n_iter):
        idx = rng.choice(k, size=2, replace=False)
        tx, ty, theta = _solve_se2_minimal(pts_b[idx], pts_a[idx])

        pts_b_t   = _apply_se2(pts_b, tx, ty, theta)
        residuals = np.linalg.norm(pts_b_t - pts_a, axis=1)
        n_inliers = int((residuals < thr).sum())

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_tx, best_ty, best_theta = tx, ty, theta

    if best_inliers < min_inl:
        return None

    # ── refinement: least-squares on all inliers ──────────────────────────────
    pts_b_t   = _apply_se2(pts_b, best_tx, best_ty, best_theta)
    inliers   = np.linalg.norm(pts_b_t - pts_a, axis=1) < thr

    if inliers.sum() < 2:
        return None

    tx_r, ty_r, theta_r = _solve_se2_lstsq(pts_b[inliers], pts_a[inliers])
    return float(tx_r), float(ty_r), float(theta_r), int(inliers.sum())


# ── SE(2) solvers ─────────────────────────────────────────────────────────────

def _solve_se2_minimal(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float, float]:
    """solve SE(2) from exactly 2 point correspondences (closed-form)."""
    d_src = src[1] - src[0]
    d_dst = dst[1] - dst[0]

    theta = np.arctan2(
        d_src[0] * d_dst[1] - d_src[1] * d_dst[0],   # cross: src × dst
        d_src[0] * d_dst[0] + d_src[1] * d_dst[1],   # dot
    )

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    t = dst.mean(axis=0) - R @ src.mean(axis=0)
    return float(t[0]), float(t[1]), float(theta)


def _solve_se2_lstsq(src: np.ndarray, dst: np.ndarray) -> Tuple[float, float, float]:
    """least-squares SE(2) from n ≥ 2 correspondences via SVD (Kabsch algorithm)."""
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)

    H = (src - c_src).T @ (dst - c_dst)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:   # handle reflection
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