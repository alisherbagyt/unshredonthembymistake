# src/solver/matcher.py
# phase 4 + 5: global solver.
# key fix over v1: alignment is now content-aware — fragments are placed by
# matching their INTERIOR visual content (embedding similarity) not just edge shape.
# for rectangular/clean cuts, content continuity is the only reliable signal.
# upgrade path: set solver.method = "gnn" in config.

import numpy as np
from typing import List, Dict, Tuple, Optional

from src.data_models import fragment


# -- public api ----------------------------------------------------------------

def solve_layout(frags: List[fragment], cfg: dict) -> List[fragment]:
    """assign global (x, y, theta) pose to every fragment."""
    sol_cfg = cfg["solver"]
    method = sol_cfg.get("method", "greedy")
    frag_map: Dict[str, fragment] = {f.id: f for f in frags}

    if method == "greedy":
        _solve_greedy(frag_map, sol_cfg)
    elif method == "gnn":
        raise NotImplementedError("gnn solver: implement gat_network.py and sinkhorn.py")
    else:
        raise ValueError(f"unknown solver: {method}")

    placed = sum(1 for f in frags if f.global_pose)
    print(f"[solver] assigned poses to {placed} / {len(frags)} fragments")
    return frags


# -- greedy solver -------------------------------------------------------------

def _solve_greedy(frag_map: Dict[str, fragment], cfg: dict):
    """greedy placement: anchor first fragment, snap neighbors by best match score.
    unmatched fragments fall back to a readable grid layout."""
    # lower min_match_score for rectangular chunks — content similarity is weaker
    # than tear-shape similarity, so we accept fuzzier matches
    min_score = cfg.get("min_match_score", 0.3)

    frags = list(frag_map.values())
    frags[0].global_pose = (0.0, 0.0, 0.0)
    placed = {frags[0].id}
    queue = list(frags[1:])
    max_iter = cfg.get("max_iterations", 1000)

    for _ in range(max_iter):
        if not queue:
            break
        progress = False

        for frag in queue[:]:
            best = _best_placed_match(frag, placed, min_score)
            if best is None:
                continue
            anchor_id, score = best
            anchor = frag_map[anchor_id]
            tx, ty, theta = _align_by_content(frag, anchor, cfg)
            frag.global_pose = (tx, ty, theta)
            placed.add(frag.id)
            queue.remove(frag)
            progress = True

        if not progress:
            break

    # fallback: any unplaced fragments go into a grid
    if queue:
        print(f"[solver] {len(queue)} fragments unmatched — placing in fallback grid")
        _assign_fallback_grid(queue, frag_map)


def _best_placed_match(
    frag: fragment,
    placed: set,
    min_score: float,
) -> Optional[Tuple[str, float]]:
    """return the highest-scoring already-placed candidate, or None."""
    if not frag.match_candidates:
        return None
    for fid, score in frag.match_candidates:
        if fid in placed and score >= min_score:
            return fid, score
    # if nothing clears min_score, take the best available placed match anyway
    # this prevents fragments being stuck in fallback just from a bad threshold
    placed_candidates = [(fid, s) for fid, s in frag.match_candidates if fid in placed]
    if placed_candidates:
        return max(placed_candidates, key=lambda x: x[1])
    return None


def _align_by_content(frag: fragment, anchor: fragment, cfg: dict) -> Tuple[float, float, float]:
    """place frag adjacent to anchor based on which side their content most resembles.
    strategy: compare mean embeddings of each fragment's four edge strips,
    find the pair of sides (e.g. anchor-right vs frag-left) with highest cosine sim,
    then position frag flush against that side of the anchor."""
    ax, ay, _ = anchor.global_pose
    ah, aw = anchor.image_rgba.shape[:2]
    fh, fw = frag.image_rgba.shape[:2]

    # get mean embedding vector per edge side for both fragments
    anchor_sides = _side_embeddings(anchor)
    frag_sides   = _side_embeddings(frag)

    # side pairing: anchor side -> frag side that would be flush against it
    # e.g. if anchor's right edge matches frag's left edge, frag goes to the right
    pairings = [
        ("right",  "left",   ax + aw,        ay + (ah - fh) / 2,  0.0),
        ("left",   "right",  ax - fw,         ay + (ah - fh) / 2,  0.0),
        ("bottom", "top",    ax + (aw - fw) / 2, ay + ah,          0.0),
        ("top",    "bottom", ax + (aw - fw) / 2, ay - fh,          0.0),
    ]

    best_score = -1.0
    best_pose  = (ax + aw, ay, 0.0)   # default: place to the right

    for a_side, f_side, tx, ty, theta in pairings:
        a_vec = anchor_sides.get(a_side)
        f_vec = frag_sides.get(f_side)
        if a_vec is None or f_vec is None:
            continue
        score = float(_cosine_sim(a_vec, f_vec))
        if score > best_score:
            best_score = score
            best_pose  = (float(tx), float(ty), float(theta))

    return best_pose


def _side_embeddings(frag: fragment) -> Dict[str, np.ndarray]:
    """compute mean embedding vector for each of the 4 edge strips of a fragment.
    uses the spatial position of each patch's contour point to assign it to a side."""
    if frag.edge_embeddings is None or len(frag.edge_embeddings) == 0:
        return {}

    contour = frag.contour_coords   # (n, 2) — same order as embeddings were sampled
    h, w = frag.image_rgba.shape[:2]

    # assign each contour point to nearest side by which boundary it's closest to
    sides: Dict[str, List[np.ndarray]] = {"top": [], "bottom": [], "left": [], "right": []}

    # stride used during patch sampling (must match encoder stride)
    # we re-index: embeddings were sampled at contour[::stride], same length
    n_emb = len(frag.edge_embeddings)
    stride = max(1, len(contour) // n_emb)
    sampled_contour = contour[::stride][:n_emb]

    for pt, emb in zip(sampled_contour, frag.edge_embeddings):
        x, y = pt
        # distance to each of the four borders
        dist = {
            "top":    y,
            "bottom": h - y,
            "left":   x,
            "right":  w - x,
        }
        nearest = min(dist, key=dist.__getitem__)
        sides[nearest].append(emb)

    # mean embedding per side; skip sides with no samples
    return {
        side: np.mean(np.stack(vecs), axis=0)
        for side, vecs in sides.items()
        if vecs
    }


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """cosine similarity between two 1-d vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _assign_fallback_grid(queue: list, frag_map: dict):
    """place unmatched fragments in a tidy grid so output is still readable."""
    placed_count = len(frag_map) - len(queue)
    for i, frag in enumerate(queue):
        col = (placed_count + i) % 5
        row = (placed_count + i) // 5
        frag.global_pose = (float(col * 800), float(row * 800), 0.0)