# src/rendering/renderer.py
# phase 6: composite all fragments at their optimized global poses.
# poses are continuous (tx, ty, theta_rad) — not snapped to a grid.
# poisson seamless clone available as upgrade — set rendering.method = "poisson".
#
# rotation center fix:
#   the global_pose (tx, ty, theta) means "rotate fragment around its own
#   origin (0,0), then translate by (tx, ty)".  the old code rotated around
#   (w/2, h/2), which drifted every fragment by half its size at any non-zero
#   angle.  the fix uses a single warpAffine matrix that rotates around (0,0)
#   and translates to the canvas position in one step — no drift possible.

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from src.data_models import fragment


def render_reconstruction(frags: List[fragment], cfg: dict) -> np.ndarray:
    rc     = cfg["rendering"]
    pad    = rc.get("canvas_padding", 60)
    bg     = tuple(rc.get("background_color", [255, 255, 255]))
    method = rc.get("method", "simple")

    placed = [f for f in frags if f.global_pose is not None]
    if not placed:
        raise RuntimeError("no fragments have poses — registration failed")

    canvas, (ox, oy) = _allocate_canvas(placed, pad, bg)

    for frag in placed:
        if method == "simple":
            _paste_simple(frag, canvas, ox, oy)
        elif method == "poisson":
            _paste_poisson(frag, canvas, ox, oy)
        else:
            raise ValueError(f"unknown rendering method: {method}")

    print(f"[renderer] composited {len(placed)}/{len(frags)} fragments "
          f"onto {canvas.shape[1]}×{canvas.shape[0]}")
    return canvas


def save_result(canvas: np.ndarray, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, canvas)
    print(f"[renderer] saved → {path}")


# ── canvas allocation ─────────────────────────────────────────────────────────

def _allocate_canvas(
    frags: List[fragment], pad: int, bg: tuple
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """compute canvas size from the bounding box of all rotated fragment corners.

    each fragment's corners are rotated around the fragment origin (0,0) and
    then translated by (tx, ty) — matching exactly what _paste_simple does.
    this guarantees the canvas is never too small.
    """
    all_x, all_y = [], []
    for f in frags:
        tx, ty, theta = f.global_pose
        h, w = f.image_rgba.shape[:2]
        # corners in fragment-local space, origin at (0,0)
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        # rotate around origin, then translate — same as the warpAffine below
        rotated = (R @ corners.T).T + np.array([tx, ty])
        all_x.extend(rotated[:, 0])
        all_y.extend(rotated[:, 1])

    min_x, min_y = min(all_x) - pad, min(all_y) - pad
    max_x, max_y = max(all_x) + pad, max(all_y) + pad

    cw = max(1, int(np.ceil(max_x - min_x)))
    ch = max(1, int(np.ceil(max_y - min_y)))
    canvas = np.full((ch, cw, 3), bg, dtype=np.uint8)
    return canvas, (int(-min_x), int(-min_y))


# ── compositing ───────────────────────────────────────────────────────────────

def _make_warp_matrix(
    tx: float, ty: float, theta: float, ox: int, oy: int, w: int, h: int
) -> Tuple[np.ndarray, int, int]:
    """build a single 2×3 affine matrix that rotates around fragment origin (0,0)
    and translates to the correct canvas position in one step.

    this eliminates the rotation-center drift that occurred when rotating around
    (w/2, h/2) and then naively placing at (tx, ty).

    the output canvas crop is sized to contain the entire rotated fragment,
    with its top-left corner at (dst_x, dst_y) in canvas coordinates.

    returns:
        M:      (2, 3) float64 affine matrix for warpAffine
        dst_x:  canvas x of the output crop's top-left corner
        dst_y:  canvas y of the output crop's top-left corner
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # bounding box of the rotated fragment (corners rotated around origin)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    R2      = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = (R2 @ corners.T).T

    bbox_x0 = rotated[:, 0].min()
    bbox_y0 = rotated[:, 1].min()

    # canvas position of the rotated bounding box top-left
    dst_x = int(round(tx + bbox_x0)) + ox
    dst_y = int(round(ty + bbox_y0)) + oy

    # output image size: ceil of the rotated bounding box
    out_w = int(np.ceil(rotated[:, 0].max() - bbox_x0))
    out_h = int(np.ceil(rotated[:, 1].max() - bbox_y0))
    out_w = max(1, out_w)
    out_h = max(1, out_h)

    # affine matrix: rotate around (0,0), then shift so bbox top-left → (0,0) in output
    M = np.array([
        [cos_t, -sin_t, -bbox_x0],
        [sin_t,  cos_t, -bbox_y0],
    ], dtype=np.float64)

    return M, dst_x, dst_y, out_w, out_h


def _paste_simple(frag: fragment, canvas: np.ndarray, ox: int, oy: int):
    """rotate fragment around its origin, translate to canvas pose, alpha-blend."""
    tx, ty, theta = frag.global_pose
    h, w = frag.image_rgba.shape[:2]

    M, dst_x, dst_y, out_w, out_h = _make_warp_matrix(tx, ty, theta, ox, oy, w, h)

    # warp the full rgba image with the corrected matrix
    rgba = cv2.warpAffine(
        frag.image_rgba, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # clamp source and destination regions to canvas bounds
    sx0 = max(0, -dst_x);    dx0 = max(0, dst_x)
    sy0 = max(0, -dst_y);    dy0 = max(0, dst_y)
    sx1 = min(out_w, canvas.shape[1] - dst_x + sx0)
    sy1 = min(out_h, canvas.shape[0] - dst_y + sy0)
    if sx1 <= sx0 or sy1 <= sy0:
        return

    src = rgba[sy0:sy1, sx0:sx1]
    ch  = src.shape[0]
    cw  = src.shape[1]
    dst = canvas[dy0:dy0 + ch, dx0:dx0 + cw]
    if dst.shape[:2] != src.shape[:2]:
        # safety clamp if rounding pushed us one pixel over edge
        ch = min(ch, dst.shape[0])
        cw = min(cw, dst.shape[1])
        src = src[:ch, :cw]
        dst = canvas[dy0:dy0 + ch, dx0:dx0 + cw]

    alpha   = src[:, :, 3:4].astype(np.float32) / 255.0
    blended = alpha * src[:, :, :3].astype(np.float32) + (1 - alpha) * dst.astype(np.float32)
    canvas[dy0:dy0 + ch, dx0:dx0 + cw] = blended.astype(np.uint8)


def _paste_poisson(frag: fragment, canvas: np.ndarray, ox: int, oy: int):
    """seamless poisson clone — removes lighting seams at fragment boundaries.
    uses the same corrected rotation matrix as _paste_simple.
    falls back to simple alpha-blend if poisson fails (e.g. fragment near border).
    """
    tx, ty, theta = frag.global_pose
    h, w = frag.image_rgba.shape[:2]

    M, dst_x, dst_y, out_w, out_h = _make_warp_matrix(tx, ty, theta, ox, oy, w, h)

    rgba = cv2.warpAffine(
        frag.image_rgba, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    mask = rgba[:, :, 3]
    src  = rgba[:, :, :3]

    # seamlessClone center = center of the rotated fragment on the canvas
    cx = dst_x + out_w // 2
    cy = dst_y + out_h // 2

    if (0 < cx < canvas.shape[1] and 0 < cy < canvas.shape[0] and mask.any()):
        try:
            canvas[:] = cv2.seamlessClone(src, canvas, mask, (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error:
            _paste_simple(frag, canvas, ox, oy)
    else:
        _paste_simple(frag, canvas, ox, oy)