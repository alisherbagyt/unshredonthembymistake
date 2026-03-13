# src/matching/faiss_index.py
# phase 3: sub-quadratic candidate retrieval using fragment-level mean embeddings.
# same faiss structure as before, but now queries against eac-net embeddings
# (trained on contour geometry + texture) instead of frozen imagenet features.
#
# the mean embedding over all contour points is a global signature of the
# fragment's boundary. top-k retrieval gives candidates for the pairwise
# cross-attention scorer to evaluate.

import numpy as np
import faiss
from typing import List, Dict, Tuple
from src.data_models import fragment


def retrieve_candidates(frags: List[fragment], cfg: dict) -> List[fragment]:
    """find top-k nearest fragments per fragment by mean boundary embedding."""
    rc      = cfg["retrieval"]
    top_k   = rc["top_k"]
    itype   = rc.get("index_type", "flat")

    # aggregate per-fragment: mean over contour points → one vector per fragment
    ids  = [f.id for f in frags]
    vecs = np.stack([f.edge_embeddings.mean(axis=0) for f in frags]).astype(np.float32)
    faiss.normalize_L2(vecs)   # unit length → inner product = cosine sim

    dim   = vecs.shape[1]
    index = _build_index(itype, dim, rc)
    index.add(vecs)

    # query each fragment against the full index
    k_query = min(top_k + 1, len(frags))   # +1 to absorb self-match
    scores_mat, idx_mat = index.search(vecs, k_query)

    for i, frag in enumerate(frags):
        candidates = []
        for score, j in zip(scores_mat[i], idx_mat[i]):
            if j < 0 or ids[j] == frag.id:
                continue
            candidates.append((ids[j], float(score)))
        frag.match_candidates = candidates[:top_k]

    print(f"[retrieval] {itype} index — {len(frags)} fragments, top_k={top_k}")
    return frags


def _build_index(itype: str, dim: int, cfg: dict) -> faiss.Index:
    if itype == "flat":
        return faiss.IndexFlatIP(dim)
    if itype == "hnsw":
        m     = cfg.get("hnsw_m", 32)
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch       = 64
        return index
    raise ValueError(f"unknown index type: {itype}")