# src/rendering/renderer.py
# phase 6: composite all fragments at their optimized global poses.
# poses are now continuous (tx, ty, theta_rad) not snapped to a grid.
# poisson seamless clone available as upgrade — removes visible seams.

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

def _allocate_canvas(frags: List[fragment], pad: int, bg: tuple) -> Tuple[np.ndarray, Tuple[int,int]]:
    """compute bounding box over all rotated fragment corners."""
    all_x, all_y = [], []
    for f in frags:
        tx, ty, theta = f.global_pose
        h, w = f.image_rgba.shape[:2]
        corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t],[sin_t, cos_t]])
        rotated = (R @ corners.T).T + np.array([tx, ty])
        all_x.extend(rotated[:, 0])
        all_y.extend(rotated[:, 1])

    min_x, min_y = min(all_x) - pad, min(all_y) - pad
    max_x, max_y = max(all_x) + pad, max(all_y) + pad

    cw = max(1, int(max_x - min_x))
    ch = max(1, int(max_y - min_y))
    canvas = np.full((ch, cw, 3), bg, dtype=np.uint8)
    return canvas, (int(-min_x), int(-min_y))


# ── compositing ───────────────────────────────────────────────────────────────

def _paste_simple(frag: fragment, canvas: np.ndarray, ox: int, oy: int):
    """rotate fragment by theta (continuous), alpha-blend onto canvas."""
    tx, ty, theta = frag.global_pose
    h, w  = frag.image_rgba.shape[:2]
    theta_deg = np.degrees(theta)

    if abs(theta_deg) > 0.05:
        rot  = cv2.getRotationMatrix2D((w/2, h/2), -theta_deg, 1.0)  # cv2 is CW
        rgba = cv2.warpAffine(frag.image_rgba, rot, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0,0,0,0))
    else:
        rgba = frag.image_rgba

    dst_x = int(round(tx)) + ox
    dst_y = int(round(ty)) + oy

    # clamp to canvas
    sx0 = max(0, -dst_x);  dx0 = max(0, dst_x)
    sy0 = max(0, -dst_y);  dy0 = max(0, dst_y)
    sx1 = min(w, canvas.shape[1] - dx0 + sx0)
    sy1 = min(h, canvas.shape[0] - dy0 + sy0)
    if sx1 <= sx0 or sy1 <= sy0:
        return

    src   = rgba[sy0:sy1, sx0:sx1]
    dst   = canvas[dy0:dy0+src.shape[0], dx0:dx0+src.shape[1]]
    alpha = src[:,:,3:4].astype(np.float32) / 255.0
    blended = alpha * src[:,:,:3].astype(np.float32) + (1-alpha) * dst.astype(np.float32)
    canvas[dy0:dy0+src.shape[0], dx0:dx0+src.shape[1]] = blended.astype(np.uint8)


def _paste_poisson(frag: fragment, canvas: np.ndarray, ox: int, oy: int):
    """seamless poisson clone — removes lighting seams at fragment boundaries.
    upgrade: set rendering.method = "poisson" in config."""
    tx, ty, theta = frag.global_pose
    h, w  = frag.image_rgba.shape[:2]
    theta_deg = np.degrees(theta)

    if abs(theta_deg) > 0.05:
        rot  = cv2.getRotationMatrix2D((w/2, h/2), -theta_deg, 1.0)
        rgba = cv2.warpAffine(frag.image_rgba, rot, (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0,0,0,0))
    else:
        rgba = frag.image_rgba

    mask = rgba[:,:,3]                 # binary mask from segmentation
    src  = rgba[:,:,:3]
    cx   = int(round(tx)) + ox + w // 2
    cy   = int(round(ty)) + oy + h // 2

    # cv2.seamlessClone requires center point inside canvas
    if (0 < cx < canvas.shape[1] and 0 < cy < canvas.shape[0]
            and mask.any()):
        try:
            canvas[:] = cv2.seamlessClone(src, canvas, mask, (cx, cy),
                                           cv2.NORMAL_CLONE)
        except cv2.error:
            # fallback to simple if poisson fails (e.g. fragment near border)
            _paste_simple(frag, canvas, ox, oy)
    else:
        _paste_simple(frag, canvas, ox, oy)