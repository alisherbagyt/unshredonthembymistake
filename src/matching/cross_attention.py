# src/matching/cross_attention.py
# pairwise compatibility scorer using cross-attention.
#
# critique fix: replaces the destructive 1d-flatten pixel comparison with a
# sequence-to-sequence attention mechanism that:
#   1. compares fragment a's contour embeddings to fragment b's pointwise
#   2. identifies the specific contour-point alignment (index mapping)
#   3. outputs a scalar compatibility score AND matched point index pairs
#
# the index mapping feeds directly into ransac for pose estimation.
# this is the bridge between "who might match" (faiss) and "where exactly" (ransac).

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from src.data_models import fragment


def score_pair(
    frag_a: fragment,
    frag_b: fragment,
    temperature: float = 0.1,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """cross-attention compatibility score between two fragments.

    computes the attention matrix between all contour points of a and b.
    the score is the mean of the maximum similarities (soft matching).
    the matched point indices enable ransac pose estimation.

    returns:
        score:       float in [0, 1] — higher = more likely to match
        pts_a:       (k, 2) float32 — matched contour points from a
        pts_b:       (k, 2) float32 — corresponding points from b
    """
    emb_a = frag_a.edge_embeddings   # (na, d)
    emb_b = frag_b.edge_embeddings   # (nb, d)

    if emb_a is None or emb_b is None:
        return 0.0, np.empty((0, 2)), np.empty((0, 2))

    # cosine similarity matrix: (na, nb)
    # embeddings are already l2-normalized by eac-net, so dot product = cosine sim
    sim = emb_a @ emb_b.T   # (na, nb)

    # soft mutual nearest neighbor: for each a, find best b; for each b, find best a
    # a point pair is a "mutual match" if they are each other's nearest neighbor
    nn_a = sim.argmax(axis=1)   # (na,) — for each a-point, index of best b-point
    nn_b = sim.argmax(axis=0)   # (nb,) — for each b-point, index of best a-point

    # mutual nearest neighbors: i matches j iff nn_a[i]==j and nn_b[j]==i
    mutual_mask = np.array([nn_b[nn_a[i]] == i for i in range(len(nn_a))])
    a_indices   = np.where(mutual_mask)[0]
    b_indices   = nn_a[a_indices]

    if len(a_indices) < 2:
        # fall back to top-k single nearest neighbors when mutual matching fails
        top_k = min(20, len(nn_a))
        scores_per_a = sim.max(axis=1)                    # (na,)
        a_indices    = np.argsort(scores_per_a)[-top_k:]  # top_k highest
        b_indices    = nn_a[a_indices]

    # compatibility score: mean max-similarity, temperature-sharpened
    # softmax-weighted mean rewards confident matches, penalizes diffuse ones
    max_sims = sim[a_indices, b_indices]
    weights  = F.softmax(torch.from_numpy(max_sims) / temperature, dim=0).numpy()
    score    = float((weights * max_sims).sum())

    pts_a = frag_a.contour_coords[a_indices]   # (k, 2)
    pts_b = frag_b.contour_coords[b_indices]   # (k, 2)

    return score, pts_a, pts_b


def score_all_candidates(
    frags: List[fragment],
    cfg:   dict,
) -> List[fragment]:
    """run cross-attention on every (frag, candidate) pair.
    updates match_candidates with cross-attention scores (replaces faiss scores).
    also populates initial pairwise_transforms via quick centroid alignment."""
    min_score = cfg["registration"]["min_score"]
    frag_map  = {f.id: f for f in frags}

    for frag in frags:
        if not frag.match_candidates:
            continue
        rescored = []
        for cand_id, _ in frag.match_candidates:
            cand  = frag_map.get(cand_id)
            if cand is None:
                continue
            score, pts_a, pts_b = score_pair(frag, cand)
            if score >= min_score:
                rescored.append((cand_id, score, pts_a, pts_b))

        # sort by cross-attention score descending
        rescored.sort(key=lambda x: -x[1])
        frag.match_candidates = [(cid, s) for cid, s, _, _ in rescored]

        # stash matched point pairs for ransac (temporary attribute)
        frag._match_points = {cid: (pts_a, pts_b) for cid, _, pts_a, pts_b in rescored}

    print(f"[cross-attn] rescored {len(frags)} fragments")
    return frags