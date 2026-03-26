# src/matching/geometry_matcher.py
# implements the contour matching algorithm from:
#   Stieber et al., "A Contour Matching Algorithm to Reconstruct Ruptured
#   Documents", DAGM 2010, LNCS 6376, pp. 121-130.
#
# pipeline (follows the paper exactly):
#
#   1. CONTOUR SEGMENTATION (section 2.1)
#      detect corners as high-curvature peaks that separate the outline into
#      segments.  classify each segment as:
#        - border segment: near-straight original paper edge  → discard
#        - contour segment: irregular tear edge               → keep for matching
#
#   2. CURVATURE FEATURE STRINGS (section 2.2, wolfson method)
#      represent each contour segment as a sequence of averaged turning-angle
#      differences: φ_k = (1/q) Σ Δθ(p^arc_k + l·δ)
#      parameters from paper table 1: se=1.9, q=2, Δs=1.8, δ=3.1, m=2.6
#
#   3. SMITH-WATERMAN LOCAL ALIGNMENT (section 2.3)
#      find the optimal LOCAL alignment of two feature strings — handles
#      partial overlaps where tear edges differ in length.
#      scoring scheme (equations 6-7):
#        wij = +2    if |f1_i - f2_j| <= ε1
#        wij = -0.1  if ε1 < |f1_i - f2_j| <= ε2
#        wij = -2    if |f1_i - f2_j| > ε2
#        w_gap = -1
#
#   4. THREE-COMPONENT SIMILARITY SCORE (section 2.3)
#      s_total = s_area + s_len + s_cor  (lower = better match, range [0,3])
#      converted to [0,1] where 1 = best for compatibility with the pipeline.
#
#   5. POINT CORRESPONDENCES FOR RANSAC
#      the SW alignment backtrack gives exact index pairs (i,j) where
#      contour segment A[i] corresponds to contour segment B[j].
#      these are returned as (pts_a, pts_b) in image pixel coordinates.

import numpy as np
from typing import List, Tuple, Optional
from src.data_models import fragment


# ── paper parameters (table 1) ────────────────────────────────────────────────
# these were optimised by the authors on a representative fragment set.
# se: step size of the equally spaced turning function graph
# q:  number of differences to be averaged
# ds: distance Δs used to compute turning function difference
# delta: distance δ between arc-length values for averaging
# m:  global scale factor applied to curvature values
_SE    = 1.9
_Q     = 2
_DS    = 1.8
_DELTA = 3.1
_M     = 2.6

# smith-waterman scoring (equations 6-7)
_EPS1   = 1.0    # tolerance for match reward
_EPS2   = 2.0    # tolerance for small penalty
_W_MATCH      =  2.0
_W_NEAR_MISS  = -0.1
_W_MISMATCH   = -2.0
_W_GAP        = -1.0


# ── public api ────────────────────────────────────────────────────────────────

def match_fragments(frags: List[fragment], cfg: dict) -> List[fragment]:
    """match all fragment pairs using the Stieber et al. pipeline."""
    mc        = cfg["matching"]
    top_k     = mc["top_k"]
    min_score = cfg["registration"]["min_score"]

    n   = len(frags)
    ids = [f.id for f in frags]

    # step 1+2: segment contour and build feature strings per fragment
    segments = []   # list of lists of (feature_string, contour_indices) per fragment
    for frag in frags:
        segs = _segment_and_featurize(frag)
        segments.append(segs)
        print(f"[geo-match] {frag.id}: {len(segs)} tear segment(s) found")

    # step 3+4: score every unique pair with smith-waterman
    scores      = {f.id: {} for f in frags}   # {id: {other_id: score}}
    corresp     = {f.id: {} for f in frags}   # {id: {other_id: (idx_a, idx_b)}}

    for i in range(n):
        for j in range(i + 1, n):
            score, idx_a, idx_b = _best_segment_pair(
                segments[i], frags[i].contour_coords,
                segments[j], frags[j].contour_coords,
            )
            scores[ids[i]][ids[j]] = score
            scores[ids[j]][ids[i]] = score
            corresp[ids[i]][ids[j]] = (idx_a, idx_b)
            corresp[ids[j]][ids[i]] = (idx_b, idx_a)

    print(f"[geo-match] scored {n*(n-1)//2} pairs across {n} fragments")

    # step 5: assign candidates and point pairs
    for fi, frag in enumerate(frags):
        fid    = frag.id
        ranked = sorted(scores[fid].items(), key=lambda x: -x[1])

        candidates  = []
        matched_pts = {}

        for other_id, score in ranked:
            if score < min_score:
                continue
            if len(candidates) >= top_k:
                break

            candidates.append((other_id, score))

            fj          = ids.index(other_id)
            idx_a, idx_b = corresp[fid][other_id]

            if len(idx_a) >= 2:
                pts_a = frag.contour_coords[idx_a].astype(np.float32)
                pts_b = frags[fj].contour_coords[idx_b].astype(np.float32)
                matched_pts[other_id] = (pts_a, pts_b)

        frag.match_candidates = candidates
        frag.matched_points   = matched_pts

    n_matched = sum(1 for f in frags if f.matched_points)
    print(f"[geo-match] {n_matched}/{n} fragments have point correspondences for ransac")
    return frags


# ── step 1: contour segmentation ─────────────────────────────────────────────

def _segment_and_featurize(
    frag: fragment,
    corner_thresh:  float = 0.35,   # curvature peak must exceed this (normalised)
    straight_thresh: float = 0.08,  # mean |curvature| below this = border segment
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """segment contour into tear segments and compute feature strings.

    returns list of (feature_string, contour_indices) for each tear segment.
    border segments (straight edges) are discarded.
    if no corners found, the full contour is treated as one tear segment.
    """
    contour = frag.contour_coords
    n       = len(contour)
    curv    = _compute_raw_curvature(contour)   # unnormalised, for corner detection

    # smooth curvature slightly before corner detection
    kernel   = np.ones(7) / 7
    curv_s   = np.convolve(np.tile(curv, 3), kernel, mode='same')[n:2*n]

    # find corners: local maxima of |curvature| above threshold
    abs_curv = np.abs(curv_s)
    c_max    = abs_curv.max()
    if c_max < 1e-6:
        # degenerate contour — return full contour as single segment
        return [(_build_feature_string(curv, np.arange(n)), np.arange(n))]

    threshold = corner_thresh * c_max
    corners   = _find_peaks(abs_curv, threshold, min_dist=n // 12)

    if len(corners) < 2:
        # no clear corners — treat full contour as tear segment
        feat = _build_feature_string(curv, np.arange(n))
        return [(feat, np.arange(n))]

    # split contour at corners into segments
    corners_sorted = np.sort(corners)
    result = []

    for k in range(len(corners_sorted)):
        start = corners_sorted[k]
        end   = corners_sorted[(k + 1) % len(corners_sorted)]

        if start < end:
            idx = np.arange(start, end)
        else:
            idx = np.concatenate([np.arange(start, n), np.arange(0, end)])

        if len(idx) < 8:
            continue   # too short to be meaningful

        seg_curv = curv[idx % n]

        # classify: border (straight) or tear (irregular)
        if np.abs(seg_curv).mean() < straight_thresh:
            continue   # border segment — discard

        feat = _build_feature_string(curv, idx)
        result.append((feat, idx))

    if not result:
        # all segments were classified as border — fall back to full contour
        feat = _build_feature_string(curv, np.arange(n))
        return [(feat, np.arange(n))]

    return result


def _find_peaks(signal: np.ndarray, threshold: float, min_dist: int) -> np.ndarray:
    """find local maxima above threshold with minimum separation."""
    n       = len(signal)
    peaks   = []
    in_peak = False
    peak_val = 0.0
    peak_idx = 0

    for i in range(n):
        if signal[i] >= threshold:
            if signal[i] > peak_val:
                peak_val = signal[i]
                peak_idx = i
            in_peak = True
        else:
            if in_peak:
                peaks.append(peak_idx)
                peak_val = 0.0
                in_peak  = False

    if in_peak:
        peaks.append(peak_idx)

    # enforce minimum distance between peaks
    if not peaks:
        return np.array([], dtype=int)

    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p - filtered[-1] >= min_dist:
            filtered.append(p)

    return np.array(filtered, dtype=int)


# ── step 2: wolfson curvature feature strings ─────────────────────────────────

def _compute_raw_curvature(contour: np.ndarray, window: int = 5) -> np.ndarray:
    """compute signed curvature via turning angle differences (unnormalised)."""
    w        = window
    prev_pts = np.roll(contour,  w, axis=0)
    next_pts = np.roll(contour, -w, axis=0)
    tangents = next_pts - prev_pts
    t_len    = np.linalg.norm(tangents, axis=1, keepdims=True).clip(1e-8)
    tangents = tangents / t_len
    t_prev   = np.roll(tangents,  1, axis=0)
    t_next   = np.roll(tangents, -1, axis=0)
    cross    = t_prev[:, 0] * t_next[:, 1] - t_prev[:, 1] * t_next[:, 0]
    return np.arcsin(cross.clip(-1, 1)).astype(np.float64)


def _turning_function(curv: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """compute the turning function θ(p^arc) for a contour segment.

    θ at each point = cumulative sum of curvature values = total turning angle
    from the start of the segment to that point.
    """
    seg = curv[idx % len(curv)]
    return np.cumsum(seg)


def _build_feature_string(
    curv:  np.ndarray,
    idx:   np.ndarray,
) -> np.ndarray:
    """build the wolfson feature string for a contour segment (equation 1-2).

    φ_k = (1/q) Σ_{l=0}^{q-1} Δθ(p^arc_k + l·δ)
    where Δθ(p^arc_k) = θ(p^arc_k + Δs) - θ(p^arc_k)

    applied to equally-spaced samples with step se.
    result is scaled by global factor m.

    returns 1d float64 array of length ~ len(idx) / se
    """
    theta = _turning_function(curv, idx)
    m_len = len(theta)

    if m_len < 4:
        return _M * theta

    # equally spaced sampling at step se
    se_int = max(1, int(round(_SE)))
    sample_idx = np.arange(0, m_len, se_int)
    ns = len(sample_idx)

    if ns < 2:
        return _M * theta[sample_idx]

    ds_int    = max(1, int(round(_DS)))
    delta_int = max(1, int(round(_DELTA)))
    q         = int(_Q)

    phi = np.zeros(ns, dtype=np.float64)
    for k in range(ns):
        pk = sample_idx[k]
        acc = 0.0
        for l in range(q):
            base      = pk + l * delta_int
            base_ds   = base + ds_int
            t_base    = theta[min(base,    m_len - 1)]
            t_base_ds = theta[min(base_ds, m_len - 1)]
            acc      += (t_base_ds - t_base)
        phi[k] = acc / q

    return (_M * phi).astype(np.float64)


# ── step 3: smith-waterman local alignment ────────────────────────────────────

def _smith_waterman(
    f1: np.ndarray,
    f2: np.ndarray,
) -> Tuple[float, List[Tuple[int, int]]]:
    """smith-waterman local alignment for real-valued curvature strings.

    scoring scheme from equations 6-7 of the paper:
      wij = +2    if |f1_i - f2_j| <= eps1
      wij = -0.1  if eps1 < |f1_i - f2_j| <= eps2
      wij = -2    if |f1_i - f2_j| > eps2
      w_gap = -1

    returns (best_score, alignment) where alignment is list of (i, j) pairs.
    """
    m = len(f1)
    n = len(f2)

    # build scoring matrix
    M = np.zeros((m + 1, n + 1), dtype=np.float64)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diff = abs(f1[i-1] - f2[j-1])
            if diff <= _EPS1:
                wij = _W_MATCH
            elif diff <= _EPS2:
                wij = _W_NEAR_MISS
            else:
                wij = _W_MISMATCH

            M[i, j] = max(
                M[i-1, j-1] + wij,
                M[i-1, j  ] + _W_GAP,
                M[i,   j-1] + _W_GAP,
                0.0,
            )

    # find best score and backtrack
    best_score = float(M.max())
    if best_score <= 0:
        return 0.0, []

    # backtrack from maximum entry to a zero entry
    bi, bj = np.unravel_index(M.argmax(), M.shape)
    alignment = []
    i, j = int(bi), int(bj)

    while i > 0 and j > 0 and M[i, j] > 0:
        alignment.append((i - 1, j - 1))   # convert to 0-indexed
        diag = M[i-1, j-1]
        up   = M[i-1, j  ]
        left = M[i,   j-1]
        best = max(diag, up, left)
        if best == diag:
            i -= 1; j -= 1
        elif best == up:
            i -= 1
        else:
            j -= 1

    alignment.reverse()
    return best_score, alignment


# ── step 4: three-component similarity score ──────────────────────────────────

def _similarity_score(
    sw_score:   float,
    alignment:  List[Tuple[int, int]],
    seg_a_len:  int,
    seg_b_len:  int,
    f1:         np.ndarray,
    f2:         np.ndarray,
) -> float:
    """compute three-component similarity score (section 2.3).

    s_total = s_area + s_len + s_cor  (paper range [0,3], lower = better)
    we invert to [0,1] where 1 = best match, for pipeline compatibility.
    """
    if not alignment or sw_score <= 0:
        return 0.0

    matched_len = len(alignment)

    # length score: how much of the shorter segment is covered
    shorter = min(seg_a_len, seg_b_len)
    s_len   = 1.0 - min(1.0, matched_len / max(shorter, 1))

    # correlation score: mean normalised similarity of matched pairs
    diffs  = np.array([abs(f1[ia] - f2[ib]) for ia, ib in alignment])
    s_cor  = 1.0 - np.clip(1.0 - diffs.mean() / max(_EPS2, 1e-8), 0, 1)

    # area score: approximate — ratio of mismatched curvature area to arc length
    # (full geometric area computation requires placing contours together;
    # we approximate with the mean absolute difference of matched values)
    s_area = float(np.clip(diffs.mean() / (2 * _EPS2), 0, 1))

    s_total = s_area + s_len + s_cor   # [0, 3], lower = better

    # invert and normalise to [0, 1] where 1 = perfect match
    return float(np.clip(1.0 - s_total / 3.0, 0.0, 1.0))


# ── best segment pair across all segment combinations ─────────────────────────

def _best_segment_pair(
    segs_a: List[Tuple[np.ndarray, np.ndarray]],
    contour_a: np.ndarray,
    segs_b: List[Tuple[np.ndarray, np.ndarray]],
    contour_b: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """find the best-matching pair of contour segments across two fragments.

    tries all combinations of tear segments from a and b.
    matching is asymmetric per the paper — also tries (b_seg, a_seg).
    returns (best_score, contour_idx_a, contour_idx_b).
    """
    best_score = 0.0
    best_idx_a = np.array([], dtype=int)
    best_idx_b = np.array([], dtype=int)

    for feat_a, idx_a in segs_a:
        for feat_b, idx_b in segs_b:
            # try both orderings (asymmetric per paper section 3)
            for fa, ia, fb, ib in [(feat_a, idx_a, feat_b, idx_b),
                                    (feat_b, idx_b, feat_a, idx_a)]:
                if len(fa) < 2 or len(fb) < 2:
                    continue

                # matching tear edges: one is the reverse of the other
                # (traverse from opposite ends of the same tear)
                fb_rev = fb[::-1].copy()

                sw_score, alignment = _smith_waterman(fa, fb_rev)
                score = _similarity_score(
                    sw_score, alignment,
                    len(ia), len(ib), fa, fb_rev,
                )

                if score > best_score:
                    best_score = score
                    # map SW alignment indices back to contour indices
                    if alignment:
                        ali_a = np.array([ia[min(p[0], len(ia)-1)]
                                          for p in alignment], dtype=int)
                        # reverse mapping for b (since we reversed fb)
                        ali_b = np.array([ib[len(ib)-1 - min(p[1], len(ib)-1)]
                                          for p in alignment], dtype=int)
                        best_idx_a = ali_a
                        best_idx_b = ali_b

    return best_score, best_idx_a, best_idx_b