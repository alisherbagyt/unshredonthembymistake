# src/matching/faiss_index.py
# phase 3: candidate retrieval via faiss nearest-neighbor search.
#
# windows avx2 fix: faiss-cpu on windows ships an avx2-optimized build that
# aborts when vectors are not c-contiguous or when the query batch is too small.
# three fixes applied:
#   1. np.ascontiguousarray() forces c-order memory layout before every faiss call
#   2. KMP_DUPLICATE_LIB_OK must be set before importing faiss — done here at
#      module level so it fires even when pytest imports this module directly
#   3. minimum query batch padded to 8 vectors (avx2 kernel minimum on windows)

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # must precede faiss import

import numpy as np
import faiss
from typing import List
from src.data_models import fragment


def retrieve_candidates(frags: List[fragment], cfg: dict) -> List[fragment]:
    """find top-k nearest fragments per fragment by mean boundary embedding."""
    rc    = cfg["retrieval"]
    top_k = rc["top_k"]
    itype = rc.get("index_type", "flat")

    ids = [f.id for f in frags]

    # mean pool contour embeddings → one vector per fragment
    # ascontiguousarray: faiss avx2 requires c-contiguous float32 — this is non-negotiable
    vecs = np.ascontiguousarray(
        np.stack([f.edge_embeddings.mean(axis=0) for f in frags]),
        dtype=np.float32,
    )   # (n_frags, embed_dim)

    # l2-normalize in-place: after this, inner product == cosine similarity
    faiss.normalize_L2(vecs)

    dim   = vecs.shape[1]
    index = _build_index(itype, dim, rc)
    index.add(vecs)

    # avx2 batch minimum: windows faiss aborts on batches smaller than 8.
    # pad query matrix with zero-rows if needed, then discard padded results.
    n         = len(vecs)
    pad_to    = max(n, 8)
    k_query   = min(top_k + 1, n)   # +1 absorbs self-match

    if n < pad_to:
        query = np.zeros((pad_to, dim), dtype=np.float32)
        query[:n] = vecs
    else:
        query = vecs

    scores_mat, idx_mat = index.search(query, k_query)
    # only use results for the real (non-padded) rows
    scores_mat = scores_mat[:n]
    idx_mat    = idx_mat[:n]

    for i, frag in enumerate(frags):
        candidates = []
        for score, j in zip(scores_mat[i], idx_mat[i]):
            if j < 0 or j >= n or ids[j] == frag.id:
                continue
            candidates.append((ids[j], float(score)))
        frag.match_candidates = candidates[:top_k]

    print(f"[retrieval] {itype} index — {n} fragments, top_k={top_k}")
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