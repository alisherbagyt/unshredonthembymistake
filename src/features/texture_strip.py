# src/features/texture_strip.py
# unrolls the fragment perimeter into a continuous image strip.
#
# critique fix: instead of axis-aligned 64×64 patches (massive overlap, wrong context),
# we sample a thin oriented strip centered on the contour, perpendicular to the edge.
# this captures ink that crosses the tear boundary — the primary matching signal.
#
# for each contour point i:
#   - compute inward normal n_i from tangent
#   - sample `strip_width` pixels along n_i centered at the contour point
#   - stack all samples into an (n, strip_width, 3) rgb strip
#
# the strip is then the input to the texture branch of eac-net.

import cv2
import numpy as np


def extract_texture_strip(
    rgba: np.ndarray,
    contour: np.ndarray,
    strip_width: int = 16,
) -> np.ndarray:
    """extract oriented perpendicular strip along the contour.

    args:
        rgba:        (h, w, 4) uint8 fragment image with alpha mask
        contour:     (n, 2) float32 equidistant contour
        strip_width: number of pixels to sample perpendicular to edge

    returns:
        (n, strip_width, 3) float32 in [0, 1] — normalized rgb strip
    """
    rgb  = rgba[:, :, :3].astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    n    = len(contour)
    half = strip_width // 2

    # tangent via central finite difference (same as geometry.py for consistency)
    prev_pts = np.roll(contour,  1, axis=0)
    next_pts = np.roll(contour, -1, axis=0)
    tangents = next_pts - prev_pts
    t_len    = np.linalg.norm(tangents, axis=1, keepdims=True).clip(1e-8)
    tangents = tangents / t_len                  # (n, 2) unit tangents

    # inward normal: rotate tangent 90° clockwise — points into fragment interior
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)  # (n, 2)

    strip = np.zeros((n, strip_width, 3), dtype=np.float32)

    for i in range(n):
        cx, cy  = contour[i]
        nx, ny  = normals[i]

        # sample strip_width pixels along the inward normal, centered on contour
        for j in range(strip_width):
            offset = j - half
            px = int(round(cx + offset * nx))
            py = int(round(cy + offset * ny))

            # boundary check — out-of-bounds = black (transparent region)
            if 0 <= px < w and 0 <= py < h:
                strip[i, j] = rgb[py, px]

    return strip   # (n, strip_width, 3)