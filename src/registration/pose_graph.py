# src/registration/pose_graph.py
# global pose graph optimization (pgo) — the key upgrade over greedy placement.
#
# critique fix: greedy placement propagates errors — one bad early match corrupts
# the entire layout. pgo solves ALL poses simultaneously, allowing loop closures
# to self-correct local errors.
#
# graph:
#   nodes = fragments, each with pose variable [tx, ty, theta]
#   edges = ransac-verified pairwise transforms with inlier count as confidence weight
#
# optimization:
#   minimize Σ_{(i,j) ∈ edges} w_ij * || T_i^{-1} ⊕ T_j - T_{ij} ||²
#   where T_ij is the observed relative transform from ransac
#   and w_ij = n_inliers (more inliers = higher confidence = higher weight)
#
# solved via scipy.optimize.minimize (L-BFGS-B) — no heavy graph-slam library needed.
# for >100 fragments, swap for g2o or gtsam.

import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple
from src.data_models import fragment


def solve_global_layout(frags: List[fragment], cfg: dict) -> List[fragment]:
    """assign global (tx, ty, theta) to every fragment via pose graph optimization."""
    sc = cfg["solver"]
    method = sc.get("method", "pgo")

    if method == "greedy":
        _greedy_fallback(frags)
        return frags

    frag_map = {f.id: f for f in frags}
    ids      = [f.id for f in frags]
    n        = len(frags)

    # build edge list from pairwise_transforms
    edges: List[Tuple[int, int, float, float, float, float]] = []
    for frag in frags:
        i = ids.index(frag.id)
        for other_id, (tx, ty, theta, n_inl) in (frag.pairwise_transforms or {}).items():
            if other_id not in frag_map:
                continue
            j      = ids.index(other_id)
            weight = float(n_inl)   # inlier count as confidence
            edges.append((i, j, tx, ty, theta, weight))

    if not edges:
        print("[pgo] no edges — falling back to greedy grid layout")
        _greedy_fallback(frags)
        return frags

    # initial state: all poses at origin
    x0 = np.zeros(n * 3, dtype=np.float64)

    # anchor fragment 0 at origin — fixes gauge freedom (absolute position)
    # we achieve this by zeroing its gradient (fixed in the cost function)

    result = minimize(
        _pgo_cost,
        x0,
        args=(edges, n),
        method="L-BFGS-B",
        options={"maxiter": sc["pgo_max_iter"], "ftol": sc["pgo_tol"], "disp": False},
    )

    poses = result.x.reshape(n, 3)
    for i, frag in enumerate(frags):
        frag.global_pose = (float(poses[i, 0]), float(poses[i, 1]), float(poses[i, 2]))

    # fallback: any fragment without a pose (disconnected component) gets a grid slot
    unplaced = [f for f in frags if f.global_pose is None]
    if unplaced:
        _greedy_fallback(unplaced, offset=n)

    print(f"[pgo] optimized {n} poses — residual {result.fun:.4f}, "
          f"converged={result.success}")
    return frags


# ── pgo cost function ─────────────────────────────────────────────────────────

def _pgo_cost(x: np.ndarray, edges: list, n: int) -> float:
    """sum of weighted squared se(2) pose composition errors.

    for edge (i, j) with observed relative transform T_ij = (tx, ty, theta):
    predicted relative = compose(inv(pose_i), pose_j)
    error = euclidean distance between predicted and observed
    """
    poses = x.reshape(n, 3)   # (n, 3) — [tx, ty, theta]
    cost  = 0.0

    for i, j, tx_obs, ty_obs, theta_obs, weight in edges:
        pi = poses[i]   # [tx_i, ty_i, theta_i]
        pj = poses[j]

        # relative transform predicted by current poses: T_i^{-1} ∘ T_j
        dtheta = pj[2] - pi[2]
        cos_i  = np.cos(-pi[2])
        sin_i  = np.sin(-pi[2])
        dtx    = cos_i * (pj[0] - pi[0]) - sin_i * (pj[1] - pi[1])
        dty    = sin_i * (pj[0] - pi[0]) + cos_i * (pj[1] - pi[1])

        # se(2) error: translation error + angular error
        err_t  = (dtx - tx_obs)**2 + (dty - ty_obs)**2
        err_r  = (_wrap_angle(dtheta - theta_obs))**2

        cost  += weight * (err_t + err_r)

    # anchor: strongly penalize pose 0 moving from origin
    cost += 1e6 * (poses[0, 0]**2 + poses[0, 1]**2 + poses[0, 2]**2)

    return cost


def _wrap_angle(a: float) -> float:
    """wrap angle to [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── greedy fallback ───────────────────────────────────────────────────────────

def _greedy_fallback(frags: List[fragment], offset: int = 0):
    """place fragments in a grid — used when pgo has no edges or as fallback."""
    placed = set()
    frag_map = {f.id: f for f in frags}

    # try to chain via highest-confidence pairwise transforms
    if frags and frags[0].global_pose is None:
        frags[0].global_pose = (0.0, 0.0, 0.0)
    placed.add(frags[0].id)

    for frag in frags[1:]:
        best = None
        for other_id, (tx, ty, theta, n_inl) in (frag.pairwise_transforms or {}).items():
            if other_id in placed:
                anchor = frag_map[other_id]
                ax, ay, at = anchor.global_pose
                # compose transforms: global_pose of frag = anchor_pose ∘ relative
                new_theta = at + theta
                cos_a, sin_a = np.cos(at), np.sin(at)
                new_tx = ax + cos_a * tx - sin_a * ty
                new_ty = ay + sin_a * tx + cos_a * ty
                if best is None or n_inl > best[3]:
                    best = (new_tx, new_ty, new_theta, n_inl)

        if best:
            frag.global_pose = (float(best[0]), float(best[1]), float(best[2]))
        else:
            i = len(placed) + offset
            frag.global_pose = (float((i % 5) * 900), float((i // 5) * 900), 0.0)
        placed.add(frag.id)