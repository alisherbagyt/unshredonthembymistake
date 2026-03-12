# src/retrieval/faiss_index.py
# phase 3: sub-quadratic candidate retrieval.
# flat index = exact search (mvp, fine for <500 fragments).
# hnsw index = O(N log N) approximate search (production upgrade, set in config).
# upgrade path: set retrieval.index_type = "hnsw" in config.yaml — zero code changes.

import numpy as np
import faiss
from typing import List, Tuple, Dict

from src.data_models import fragment


# ── public api ────────────────────────────────────────────────────────────────

def retrieve_candidates(frags: List[fragment], cfg: dict) -> List[fragment]:
    """for each fragment, find top-k nearest neighbor fragments by embedding similarity.
    stores results in frag.match_candidates = [(other_id, score), ...]."""
    ret_cfg = cfg["retrieval"]
    top_k = ret_cfg["top_k"]
    index_type = ret_cfg.get("index_type", "flat")

    # flatten all edge embeddings into one matrix, track which rows belong to which fragment
    all_vecs, row_to_frag = _build_embedding_matrix(frags)

    # l2-normalize so inner product = cosine similarity
    faiss.normalize_L2(all_vecs)

    dim = all_vecs.shape[1]
    index = _build_index(index_type, dim, ret_cfg)
    index.add(all_vecs)

    # per-fragment: aggregate patch-level scores → fragment-level score
    for frag in frags:
        frag.match_candidates = _query_for_fragment(
            frag, all_vecs, row_to_frag, index, top_k
        )

    print(f"[retrieval] built {index_type} index over {len(all_vecs)} patch embeddings")
    return frags


# ── internal helpers ───────────────────────────────────────────────────────────

def _build_embedding_matrix(frags: List[fragment]) -> Tuple[np.ndarray, Dict[int, str]]:
    """concatenate all fragment embeddings into one matrix.
    returns:
        all_vecs: (total_patches, d) float32
        row_to_frag: {row_index: fragment_id}
    """
    all_vecs = []
    row_to_frag: Dict[int, str] = {}
    current_row = 0

    for frag in frags:
        n_patches = len(frag.edge_embeddings)
        for i in range(n_patches):
            row_to_frag[current_row + i] = frag.id
        all_vecs.append(frag.edge_embeddings)
        current_row += n_patches

    return np.vstack(all_vecs).astype(np.float32), row_to_frag


def _build_index(index_type: str, dim: int, cfg: dict) -> faiss.Index:
    """construct faiss index based on config."""
    if index_type == "flat":
        # exact inner product search — perfect recall, O(N) query
        return faiss.IndexFlatIP(dim)
    elif index_type == "hnsw":
        # approximate nearest neighbor — O(log N) query, slight recall tradeoff
        m = cfg.get("hnsw_m", 32)   # graph connectivity; higher = better recall, more memory
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200   # build quality; reduce for speed
        index.hnsw.efSearch = 64          # search quality
        return index
    else:
        raise ValueError(f"unsupported index type: {index_type}")


def _query_for_fragment(
    frag: fragment,
    all_vecs: np.ndarray,
    row_to_frag: Dict[int, str],
    index: faiss.Index,
    top_k: int,
) -> List[Tuple[str, float]]:
    """query index with every patch from this fragment, aggregate scores per target fragment."""
    query_vecs = frag.edge_embeddings.astype(np.float32).copy()
    faiss.normalize_L2(query_vecs)

    # k+1 because the fragment's own patches are in the index
    scores_raw, indices_raw = index.search(query_vecs, top_k + 1)

    # aggregate: mean cosine similarity from all patch-to-patch comparisons,
    # grouped by target fragment id
    candidate_scores: Dict[str, List[float]] = {}
    for patch_scores, patch_indices in zip(scores_raw, indices_raw):
        for score, idx in zip(patch_scores, patch_indices):
            target_id = row_to_frag.get(int(idx))
            if target_id is None or target_id == frag.id:
                continue    # skip self-matches
            candidate_scores.setdefault(target_id, []).append(float(score))

    # reduce patch scores → mean score per candidate fragment
    aggregated = [
        (fid, float(np.mean(scores)))
        for fid, scores in candidate_scores.items()
    ]

    # sort descending, keep top_k
    aggregated.sort(key=lambda x: -x[1])
    return aggregated[:top_k]
