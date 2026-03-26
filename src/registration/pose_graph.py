# src/registration/pose_graph.py
# global pose graph optimization (PGO).
#
# graph:  nodes = fragments (pose variables [tx, ty, theta])
#         edges = ransac-verified pairwise transforms, weighted by inlier count
#
# cost:   Σ_{(i,j)} w_ij * [ ||Δt_ij - t_obs||² + 2*(1 - cos(Δθ_ij - θ_obs)) ]
#
# rotation cost fix:
#   the old cost used (_wrap_angle(dtheta - theta_obs))² which applies a modulo
#   operation with a discontinuous derivative at ±π. L-BFGS-B computes wrong
#   numerical gradients at these kinks and stalls on fragments near 180° rotation.
#   fix: replace with 2*(1 - cos(error)), which is smooth everywhere, has the
#   same minimum at zero error, and correct gradients throughout [-π, π].
#
# pgo_min_inliers fix:
#   the old code read cfg["registration"]["pgo_min_inliers"] (default 12) but
#   config.yaml only defined cfg["registration"]["min_inliers"] = 6 — a
#   different key with a different default. edges were silently filtered twice.
#   fix: read the single key "min_inliers" from config for both RANSAC and PGO.

import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple
from src.data_models import fragment


def solve_global_layout(frags: List[fragment], cfg: dict) -> List[fragment]:
    """assign global (tx, ty, theta) to every fragment via pose graph optimization."""
    sc     = cfg["solver"]
    method = sc.get("method", "pgo")

    if method == "greedy":
        _greedy_fallback(frags)
        return frags

    frag_map = {f.id: f for f in frags}
    ids      = [f.id for f in frags]
    n        = len(frags)

    # use the same min_inliers threshold as ransac — no diverging defaults
    pgo_min_inl = cfg["registration"]["min_inliers"]

    # build edge list from pairwise_transforms
    edges: List[Tuple] = []
    for frag in frags:
        i = ids.index(frag.id)
        for other_id, (tx, ty, theta, n_inl) in (frag.pairwise_transforms or {}).items():
            if n_inl < pgo_min_inl or other_id not in frag_map:
                continue
            j = ids.index(other_id)
            edges.append((i, j, tx, ty, theta, float(n_inl)))

    if not edges:
        print("[pgo] no edges — falling back to greedy layout")
        _greedy_fallback(frags)
        return frags

    x0 = np.zeros(n * 3, dtype=np.float64)

    result = minimize(
        _pgo_cost,
        x0,
        args=(edges, n),
        method="L-BFGS-B",
        options={"maxiter": sc["pgo_max_iter"], "ftol": sc["pgo_tol"]},
    )

    poses = result.x.reshape(n, 3)
    for i, frag in enumerate(frags):
        frag.global_pose = (float(poses[i, 0]), float(poses[i, 1]), float(poses[i, 2]))

    # any disconnected fragment (no edges) gets a grid fallback slot
    unplaced = [f for f in frags if f.global_pose is None]
    if unplaced:
        _greedy_fallback(unplaced, offset=n)

    print(f"[pgo] optimized {n} poses over {len(edges)} edges — "
          f"residual {result.fun:.4f}, converged={result.success}")
    return frags


# ── PGO cost function ─────────────────────────────────────────────────────────

def _pgo_cost(x: np.ndarray, edges: list, n: int) -> float:
    """smooth pose graph cost.

    translation error: squared Euclidean distance in the relative frame.
    rotation error:    2*(1 - cos(Δθ - θ_obs)) — smooth everywhere, no kinks.

    anchor term: pins fragment 0 to the origin to remove the gauge freedom
    (the optimization is translation/rotation invariant without it).
    """
    poses = x.reshape(n, 3)
    cost  = 0.0

    for i, j, tx_obs, ty_obs, theta_obs, weight in edges:
        pi = poses[i]
        pj = poses[j]

        # relative transform of j in i's frame
        cos_i = np.cos(-pi[2])
        sin_i = np.sin(-pi[2])
        dtx   = cos_i * (pj[0] - pi[0]) - sin_i * (pj[1] - pi[1])
        dty   = sin_i * (pj[0] - pi[0]) + cos_i * (pj[1] - pi[1])
        dtheta = pj[2] - pi[2]

        err_t = (dtx - tx_obs) ** 2 + (dty - ty_obs) ** 2
        # smooth rotation cost — no discontinuous derivative at ±π
        err_r = 2.0 * (1.0 - np.cos(dtheta - theta_obs))

        cost += weight * (err_t + err_r)

    # anchor: keep fragment 0 at origin (removes gauge freedom)
    cost += 1e6 * (poses[0, 0] ** 2 + poses[0, 1] ** 2 + poses[0, 2] ** 2)
    return cost


# ── greedy fallback ───────────────────────────────────────────────────────────

def _greedy_fallback(frags: List[fragment], offset: int = 0):
    """place fragments greedily by propagating from already-placed neighbors.
    unconnected fragments land in a readable grid."""
    placed   = set()
    frag_map = {f.id: f for f in frags}

    if frags and frags[0].global_pose is None:
        frags[0].global_pose = (0.0, 0.0, 0.0)
    placed.add(frags[0].id)

    for frag in frags[1:]:
        best = None
        for other_id, (tx, ty, theta, n_inl) in (frag.pairwise_transforms or {}).items():
            if other_id in placed:
                anchor       = frag_map[other_id]
                ax, ay, at   = anchor.global_pose
                new_theta    = at + theta
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


def convert_poses_to_image_space(frags: list) -> list:
    """convert PGO poses (local contour space) to canvas image-space poses.

    the problem:
      RANSAC receives contour points in each fragment's LOCAL crop coordinate
      system (origin = top-left of the fragment's bounding box crop).
      so the (tx, ty) it finds is the displacement between two LOCAL systems.
      PGO refines these into globally-consistent poses, still in local space.

    the fix:
      anchor fragment 0 at its natural scan position (crop_offset).
      then for every other fragment, chain the pairwise transform from its
      best-placed neighbour to compute its scan-space position.

    result: global_pose (tx, ty, theta) is now the top-left corner of each
    fragment's crop image on the final canvas, with correct rotation.
    """
    frag_map = {f.id: f for f in frags}
    placed   = {}   # {frag_id: (canvas_tx, canvas_ty, theta)}

    # anchor: fragment 0 sits at (0, 0) on the canvas.
    # the renderer builds the canvas bounding box around all fragment poses,
    # so absolute position of the anchor doesn't matter — only relative
    # positions between fragments matter.
    f0 = frags[0]
    placed[f0.id] = (0.0, 0.0,
                     float(f0.global_pose[2]) if f0.global_pose else 0.0)
    f0.global_pose = placed[f0.id]

    # propagate: for each unplaced fragment, find a placed neighbour and
    # apply the pairwise transform to get canvas position
    max_passes = len(frags)
    for _ in range(max_passes):
        progress = False
        for frag in frags[1:]:
            if frag.id in placed:
                continue
            for nb_id, (tx, ty, theta, n_inl) in (frag.pairwise_transforms or {}).items():
                if nb_id not in placed:
                    continue
                nb_cx, nb_cy, nb_theta = placed[nb_id]
                nb_frag = frag_map[nb_id]

                # coordinate system explanation:
                #
                # RANSAC solved: R @ pt_frag_local + t = pt_nb_local
                # where pt_*_local are contour points in each fragment's
                # LOCAL crop coordinate system (origin = crop top-left).
                #
                # to place frag on the canvas we need frag's crop top-left
                # in SCAN (canvas) coordinates.
                #
                # scan_pt = crop_offset + local_pt  for any fragment.
                #
                # from RANSAC: local_pt_nb = R @ local_pt_frag + t
                # in scan space: scan_pt_nb - nb_offset = R @ (scan_pt_frag - f_offset) + t
                # solving for scan_pt_frag (= frag's crop origin, local_pt_frag = 0):
                #   scan_origin_frag = R^T @ (scan_pt_nb - nb_offset - t) + f_offset
                # but we want where frag's crop top-left lands on canvas:
                #   canvas_tx_frag = nb_offset_x + tx - f_offset_x  (for theta~0)
                # generalised with rotation:
                #   canvas_origin = nb_canvas + R(nb_theta) @ (tx, ty)
                #                 + nb_offset - R(nb_theta) @ f_offset
                # where nb_canvas already includes nb_offset (set at anchor).

                nb_ox, nb_oy = nb_frag.crop_offset
                f_ox,  f_oy  = frag.crop_offset

                cos_t = np.cos(nb_theta)
                sin_t = np.sin(nb_theta)

                # rotate frag's crop offset into nb's canvas frame
                f_ox_rot = cos_t * f_ox - sin_t * f_oy
                f_oy_rot = sin_t * f_ox + cos_t * f_oy

                new_tx    = nb_cx + cos_t * tx - sin_t * ty - f_ox_rot + nb_ox
                new_ty    = nb_cy + sin_t * tx + cos_t * ty - f_oy_rot + nb_oy
                new_theta = nb_theta + theta

                placed[frag.id] = (float(new_tx), float(new_ty), float(new_theta))
                frag.global_pose = placed[frag.id]
                progress = True
                break

        if not progress:
            break

    # any fragment still unplaced (disconnected): grid fallback
    for i, frag in enumerate(frags):
        if frag.id not in placed:
            frag.global_pose = (float((i % 5) * 900), float((i // 5) * 900), 0.0)

    return frags