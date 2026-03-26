# src/features/rotation.py
# stage 2: canonical rotation estimation.
#
# problem: fragments arrive from the segmenter at arbitrary orientations.
# ransac must then search the full SE(2) space — (tx, ty, theta) simultaneously.
# with no rotation hint, the 2-point minimal sample almost never lands near the
# true rotation, so inlier counts stay below min_inliers and transforms are
# discarded. the result is zero assembly.
#
# solution: before matching, rotate each fragment to its canonical "upright"
# orientation so ransac only needs to find a 2D translation (theta ≈ 0).
# PGO then refines the small residual rotation errors.
#
# strategy (in priority order):
#   1. corner pieces  — detected by a single ~270° interior angle in the curvature
#                        sequence. the two straight sides define exact orientation.
#                        corner type (TL/TR/BL/BR) assigned by which quadrant the
#                        corner's inward direction points.
#   2. text / Hough   — dominant line angle from a Hough transform on the alpha
#                        mask edge. snap to nearest 90°.
#   3. PCA fallback   — major axis of the contour point cloud. less accurate but
#                        always available. snap to nearest 90°.
#
# all rotation is stored in fragment.canonical_theta (radians).
# fragment.piece_type is set to "corner", "edge", or "interior".
# the image_rgba and contour_coords are rotated IN PLACE so every downstream
# stage (feature extraction, matching, ransac) works in canonical space.

import cv2
import numpy as np
from typing import List, Tuple, Optional
from src.data_models import fragment


# ── public api ────────────────────────────────────────────────────────────────

def canonicalize_fragments(frags: List[fragment], cfg: dict) -> List[fragment]:
    """estimate and apply canonical rotation to every fragment.

    modifies each fragment in-place:
      - fragment.canonical_theta  set to the estimated upright correction (rad)
      - fragment.piece_type       set to "corner" | "edge" | "interior"
      - fragment.image_rgba       rotated to canonical orientation
      - fragment.contour_coords   rotated to match

    returns the same list (modified in-place).
    """
    rc = cfg.get("rotation", {})

    n_corner = n_hough = n_pca = 0

    for frag in frags:
        theta, ptype, method = _estimate_canonical_theta(frag, rc)
        frag.canonical_theta = theta
        frag.piece_type      = ptype

        if method == "corner":
            n_corner += 1
        elif method == "hough":
            n_hough += 1
        else:
            n_pca += 1

        if abs(theta) > 0.01:   # skip trivial rotations
            frag.image_rgba    = _rotate_rgba(frag.image_rgba, theta)
            frag.contour_coords = _rotate_contour(frag.contour_coords, theta,
                                                   frag.image_rgba.shape)

    print(f"[rotation] canonical poses: {n_corner} corner, "
          f"{n_hough} hough, {n_pca} pca fallback")
    return frags


# ── estimation ────────────────────────────────────────────────────────────────

def _estimate_canonical_theta(
    frag: fragment,
    cfg: dict,
) -> Tuple[float, str, str]:
    """return (theta_rad, piece_type, method_used).

    theta_rad: angle to rotate fragment to canonical upright.
    piece_type: "corner" | "edge" | "interior"
    method_used: "corner" | "hough" | "pca"
    """
    # ── 1. corner detection ───────────────────────────────────────────────────
    corner_result = _detect_corner(frag.contour_coords, cfg)
    if corner_result is not None:
        theta, corner_label = corner_result
        return theta, "corner", "corner"

    # ── 2. hough line on alpha mask ───────────────────────────────────────────
    hough_result = _hough_theta(frag.image_rgba[:, :, 3], cfg)
    if hough_result is not None:
        return hough_result, "edge", "hough"

    # ── 3. pca fallback ───────────────────────────────────────────────────────
    theta = _pca_theta(frag.contour_coords)
    return theta, "interior", "pca"


# ── corner detection ──────────────────────────────────────────────────────────

def _detect_corner(
    contour: np.ndarray,
    cfg: dict,
    straight_thresh: float = 0.15,   # curvature magnitude below this = straight
    corner_window:   int   = 20,     # points each side of the corner peak
) -> Optional[Tuple[float, str]]:
    """detect a ~270° interior angle (paper corner) in the curvature sequence.

    a paper corner appears as ONE sharp negative curvature spike (the exterior
    angle is ~90°, interior is ~270°) surrounded by two long near-zero stretches
    (the straight paper edges).

    returns (theta_rad, corner_label) or None if no corner found.
    theta_rad rotates the fragment so this corner points toward top-left.
    """
    n = len(contour)

    # curvature via central finite difference (same formula as geometry.py)
    w        = 5
    prev_pts = np.roll(contour,  w, axis=0)
    next_pts = np.roll(contour, -w, axis=0)
    tangents = next_pts - prev_pts
    t_len    = np.linalg.norm(tangents, axis=1, keepdims=True).clip(1e-8)
    tangents = tangents / t_len
    t_prev   = np.roll(tangents,  1, axis=0)
    t_next   = np.roll(tangents, -1, axis=0)
    cross    = t_prev[:, 0] * t_next[:, 1] - t_prev[:, 1] * t_next[:, 0]
    curv     = np.arcsin(cross.clip(-1, 1))

    # smooth curvature slightly to suppress noise
    kernel = np.ones(5) / 5
    curv_s = np.convolve(np.tile(curv, 3), kernel, mode="same")[n:2*n]

    # find the single sharpest negative peak (exterior corner = negative curvature)
    peak_idx = int(np.argmin(curv_s))
    peak_val = curv_s[peak_idx]

    # threshold: must be significantly more negative than average
    if peak_val > -0.3:
        return None   # no clear corner

    # check that flanking regions are indeed straight (low curvature)
    left_flank  = np.roll(curv_s, -peak_idx)[:corner_window]
    right_flank = np.roll(curv_s,  peak_idx)[:corner_window]
    if (np.abs(left_flank).mean() > straight_thresh or
            np.abs(right_flank).mean() > straight_thresh):
        return None   # corner is not flanked by straight edges

    # the corner point on the contour
    corner_pt = contour[peak_idx]   # (2,) — (x, y) in image coords

    # direction from fragment centroid to corner point
    centroid  = contour.mean(axis=0)
    direction = corner_pt - centroid
    angle_to_corner = np.arctan2(direction[1], direction[0])   # radians

    # we want the corner to point toward top-left (angle = -135° = -3π/4).
    # theta = angle that rotates the fragment so angle_to_corner → -3π/4
    target = -3 * np.pi / 4
    theta  = target - angle_to_corner

    # snap to nearest 90° to avoid over-rotation
    theta = _snap_90(theta)

    # corner label based on which quadrant the corner points to after correction
    # (informational — used by future corner-first matching)
    snapped_dir = angle_to_corner + theta
    label = _quadrant_label(snapped_dir)

    return theta, label


def _quadrant_label(angle: float) -> str:
    """map an angle (rad) to the nearest cardinal corner label."""
    a = (angle + np.pi) % (2 * np.pi) - np.pi   # wrap to [-π, π]
    if   a < -np.pi / 2: return "top-left"
    elif a < 0:          return "top-right"
    elif a < np.pi / 2:  return "bottom-right"
    else:                return "bottom-left"


# ── hough line ────────────────────────────────────────────────────────────────

def _hough_theta(
    alpha_mask: np.ndarray,
    cfg: dict,
    min_line_votes: int = 50,
) -> Optional[float]:
    """estimate dominant line angle from the fragment's alpha mask via Hough.

    extracts the edge of the alpha mask, runs probabilistic Hough, takes the
    longest line, and snaps its angle to the nearest 90°.

    returns theta_rad (upright correction) or None if no strong line found.
    """
    # edge of the alpha mask = the fragment boundary
    mask  = (alpha_mask > 128).astype(np.uint8) * 255
    edges = cv2.Canny(mask, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=min_line_votes,
        minLineLength=int(min(alpha_mask.shape) * 0.15),
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return None

    # find the longest line
    best_len   = -1
    best_angle = 0.0
    for x1, y1, x2, y2 in lines[:, 0]:
        dx  = x2 - x1
        dy  = y2 - y1
        length = np.hypot(dx, dy)
        if length > best_len:
            best_len   = length
            best_angle = np.arctan2(dy, dx)   # angle of this line in image coords

    # theta to rotate this line to horizontal (0°), then snap to nearest 90°
    theta = _snap_90(-best_angle)
    return float(theta)


# ── pca fallback ──────────────────────────────────────────────────────────────

def _pca_theta(contour: np.ndarray) -> float:
    """estimate orientation via PCA of contour points, snap to nearest 90°."""
    pts  = contour - contour.mean(axis=0)
    cov  = pts.T @ pts
    _, vecs = np.linalg.eigh(cov)   # eigenvectors sorted ascending by eigenvalue
    major = vecs[:, -1]              # principal axis (largest eigenvalue)
    angle = np.arctan2(major[1], major[0])
    return float(_snap_90(-angle))


# ── helpers ───────────────────────────────────────────────────────────────────

def _snap_90(theta: float) -> float:
    """snap an angle (rad) to the nearest multiple of 90°."""
    deg     = np.degrees(theta)
    snapped = round(deg / 90) * 90
    return float(np.radians(snapped))


def _rotate_rgba(rgba: np.ndarray, theta: float) -> np.ndarray:
    """rotate an RGBA image by theta radians around its center.
    output canvas is expanded to contain the full rotated image — no clipping.
    """
    h, w    = rgba.shape[:2]
    theta_d = np.degrees(theta)
    cx, cy  = w / 2, h / 2

    # compute expanded output size
    cos_a, sin_a = abs(np.cos(theta)), abs(np.sin(theta))
    new_w = int(np.ceil(h * sin_a + w * cos_a))
    new_h = int(np.ceil(h * cos_a + w * sin_a))

    # rotation matrix centered on original image, shifted to new center
    M = cv2.getRotationMatrix2D((cx, cy), -theta_d, 1.0)   # cv2 uses CW degrees
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(
        rgba, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _rotate_contour(
    contour: np.ndarray,
    theta: float,
    new_shape: Tuple[int, ...],
) -> np.ndarray:
    """rotate contour coordinates to match the rotated image.

    new_shape is the shape of the already-rotated image_rgba so we can compute
    the correct center shift that was applied in _rotate_rgba.
    """
    # original image center — we need the OLD shape to compute the shift.
    # since _rotate_rgba was already applied, we back-compute old size from new.
    # easier: re-derive the shift from theta directly.
    # contour coords are in old-image space.  after rotation by theta around
    # the old center (cx_old, cy_old), they land at rotated positions, then
    # shifted by ((new_w - old_w)/2, (new_h - old_h)/2).
    # we don't store old shape here, so we use the inverse: unrotate new center.
    new_h, new_w = new_shape[:2]
    new_cx, new_cy = new_w / 2, new_h / 2

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # old center = R^{-1} @ new_center  (since new_center = R @ old_center + shift
    # and shift = new_center - R @ old_center, which we solve for old_center)
    # actually: old_center = R^T @ new_center  (R is orthogonal, R^{-1} = R^T)
    old_cx = R[0, 0] * new_cx + R[1, 0] * new_cy
    old_cy = R[0, 1] * new_cx + R[1, 1] * new_cy

    # rotate each contour point around old_center, then shift to new_center
    pts = contour - np.array([old_cx, old_cy])
    rotated = (R @ pts.T).T + np.array([new_cx, new_cy])
    return rotated.astype(np.float32)