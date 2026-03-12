# src/rendering/renderer.py
# phase 6: composite all fragments onto a global canvas using solved poses.
# current: simple alpha-blend paste (fast, visible seams are acceptable for mvp).
# upgrade path: set rendering.method = "poisson" in config → seamless cloning.

import cv2
import numpy as np
from typing import List
from pathlib import Path

from src.data_models import fragment


# ── public api ────────────────────────────────────────────────────────────────

def render_reconstruction(frags: List[fragment], cfg: dict) -> np.ndarray:
    """composite all placed fragments into a single canvas image.
    returns: (h, w, 3) uint8 bgr image."""
    render_cfg = cfg["rendering"]
    method = render_cfg.get("method", "simple")
    pad = render_cfg.get("canvas_padding", 50)
    bg = tuple(render_cfg.get("background_color", [255, 255, 255]))

    placed = [f for f in frags if f.global_pose is not None]
    canvas, offset = _allocate_canvas(placed, pad, bg)

    for frag in placed:
        if method == "simple":
            _paste_simple(frag, canvas, offset)
        elif method == "poisson":
            _paste_poisson(frag, canvas, offset)   # future upgrade
        else:
            raise ValueError(f"unknown rendering method: {method}")

    print(f"[renderer] composited {len(placed)} fragments onto {canvas.shape[:2]} canvas")
    return canvas


def save_result(canvas: np.ndarray, output_path: str):
    """write result as tiff (lossless, archival standard)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"[renderer] saved → {output_path}")


# ── internal helpers ───────────────────────────────────────────────────────────

def _allocate_canvas(frags: List[fragment], pad: int, bg: tuple) -> tuple:
    """compute canvas size from all fragment poses, return blank canvas + offset."""
    # collect all fragment corners in global space to find bounding box
    all_x, all_y = [], []
    for frag in frags:
        tx, ty, theta = frag.global_pose
        h, w = frag.image_rgba.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        corners_global = _rotate_points(corners, theta) + np.array([tx, ty])
        all_x.extend(corners_global[:, 0].tolist())
        all_y.extend(corners_global[:, 1].tolist())

    min_x, min_y = min(all_x) - pad, min(all_y) - pad
    max_x, max_y = max(all_x) + pad, max(all_y) + pad

    canvas_w = int(max_x - min_x)
    canvas_h = int(max_y - min_y)
    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)

    offset = (-min_x, -min_y)   # translate so minimum coords land at (pad, pad)
    return canvas, offset


def _paste_simple(frag: fragment, canvas: np.ndarray, offset: tuple):
    """paste rgba fragment onto canvas using alpha mask — fast, shows seams."""
    tx, ty, theta = frag.global_pose
    ox, oy = offset
    h, w = frag.image_rgba.shape[:2]

    # rotate the fragment image around its center
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), theta, 1.0)
    rotated_rgba = cv2.warpAffine(frag.image_rgba, rot_mat, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))

    # destination top-left corner on canvas
    dst_x = int(tx + ox)
    dst_y = int(ty + oy)

    # clamp to canvas bounds
    src_x0, src_y0 = max(0, -dst_x), max(0, -dst_y)
    dst_x0 = max(0, dst_x)
    dst_y0 = max(0, dst_y)
    src_x1 = min(w, canvas.shape[1] - dst_x0 + src_x0)
    src_y1 = min(h, canvas.shape[0] - dst_y0 + src_y0)

    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return   # fragment outside canvas — skip

    src_crop = rotated_rgba[src_y0:src_y1, src_x0:src_x1]
    alpha = src_crop[:, :, 3:4].astype(np.float32) / 255.0
    rgb = src_crop[:, :, :3].astype(np.float32)

    # alpha blend: out = alpha * src + (1-alpha) * dst
    dst_region = canvas[dst_y0:dst_y0 + src_crop.shape[0],
                        dst_x0:dst_x0 + src_crop.shape[1]].astype(np.float32)
    blended = alpha * rgb + (1 - alpha) * dst_region
    canvas[dst_y0:dst_y0 + src_crop.shape[0],
           dst_x0:dst_x0 + src_crop.shape[1]] = blended.astype(np.uint8)


def _paste_poisson(frag: fragment, canvas: np.ndarray, offset: tuple):
    """future: seamless poisson image editing at fragment boundaries."""
    raise NotImplementedError(
        "poisson blend: use cv2.seamlessClone with the fragment mask. "
        "input: (src_rgba, canvas, mask, center_point). wire here."
    )


def _rotate_points(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    """rotate (n, 2) point array around origin by angle_deg."""
    theta = np.radians(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    return (rot @ pts.T).T
