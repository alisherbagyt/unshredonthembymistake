# src/matching/cross_attention.py
# pairwise compatibility scorer using cross-attention.
#
# scores every (frag_a, frag_b) candidate pair by:
#   1. computing a full cosine-similarity matrix between their contour embeddings
#   2. finding mutual nearest-neighbor point correspondences
#   3. returning a scalar score + the matched contour point coordinates for ransac
#
# the matched point coordinates are stored in fragment.matched_points (the
# declared dataclass field) — NOT as a dynamic _match_points attribute.
# this is the critical plumbing fix: ransac reads fragment.matched_points
# directly from the data contract, so nothing is ever silently lost.

import numpy as np
from typing import Tuple, List
from src.data_models import fragment


def score_pair(
    frag_a: fragment,
    frag_b: fragment,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """cross-attention compatibility score between two fragments.

    computes a full (na, nb) cosine similarity matrix between all contour
    point embeddings, then extracts mutual nearest-neighbor correspondences.

    score = mean similarity of mutual-NN pairs.
    this is honest and directly comparable across all candidate pairs —
    it does not inflate scores through self-weighting.

    returns:
        score:  float — mean mutual-NN cosine similarity, higher = better match
        pts_a:  (k, 2) float32 — matched contour coords from frag_a
        pts_b:  (k, 2) float32 — matched contour coords from frag_b
    """
    emb_a = frag_a.edge_embeddings   # (na, d)
    emb_b = frag_b.edge_embeddings   # (nb, d)

    if emb_a is None or emb_b is None:
        return 0.0, np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    # full cosine similarity matrix — embeddings are l2-normalised by eac-net,
    # so dot product == cosine similarity directly.
    sim = emb_a @ emb_b.T   # (na, nb)

    # mutual nearest neighbours: i↔j is a match iff they are each other's best hit.
    # vectorised: no python loop needed.
    nn_a = sim.argmax(axis=1)   # (na,) best b-index for each a-point
    nn_b = sim.argmax(axis=0)   # (nb,) best a-index for each b-point

    a_all       = np.arange(len(nn_a))
    mutual_mask = nn_b[nn_a] == a_all   # True where nn_b[nn_a[i]] == i
    a_indices   = a_all[mutual_mask]
    b_indices   = nn_a[a_indices]

    # fallback: if fewer than 2 mutual matches, take the top-20 by raw similarity.
    # this happens with random/untrained embeddings — still produces point pairs
    # for ransac to attempt, even if they are noisy.
    if len(a_indices) < 2:
        top_k    = min(20, len(nn_a))
        a_indices = np.argsort(sim.max(axis=1))[-top_k:]
        b_indices = nn_a[a_indices]

    # score = mean cosine similarity of the matched pairs.
    # plain mean is honest: every matched pair contributes equally.
    # this is directly comparable across candidate pairs for ranking.
    matched_sims = sim[a_indices, b_indices].astype(np.float32)
    score        = float(matched_sims.mean())

    pts_a = frag_a.contour_coords[a_indices]   # (k, 2) float32
    pts_b = frag_b.contour_coords[b_indices]   # (k, 2) float32

    return score, pts_a, pts_b


def score_all_candidates(
    frags: List[fragment],
    cfg:   dict,
) -> List[fragment]:
    """score every (frag, candidate) pair with cross-attention.

    - replaces faiss cosine scores with more accurate cross-attention scores
    - stores matched point coordinate pairs in fragment.matched_points
      (the declared dataclass field) so ransac can consume them directly
    """
    min_score = cfg["registration"]["min_score"]
    frag_map  = {f.id: f for f in frags}

    for frag in frags:
        if not frag.match_candidates:
            frag.matched_points = {}
            continue

        rescored      = []
        matched_pts   = {}   # {cand_id: (pts_a, pts_b)} — for ransac

        for cand_id, _ in frag.match_candidates:
            cand = frag_map.get(cand_id)
            if cand is None:
                continue

            score, pts_a, pts_b = score_pair(frag, cand)

            if score >= min_score:
                rescored.append((cand_id, score))
                matched_pts[cand_id] = (pts_a, pts_b)

        # sort candidates by score descending
        rescored.sort(key=lambda x: -x[1])
        frag.match_candidates = rescored

        # store in the declared dataclass field — visible to all downstream stages
        frag.matched_points = matched_pts

    n_with_matches = sum(1 for f in frags if f.matched_points)
    print(f"[cross-attn] rescored {len(frags)} fragments — "
          f"{n_with_matches} have point correspondences for ransac")
    return frags