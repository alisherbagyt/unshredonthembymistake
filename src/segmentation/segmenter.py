# src/segmentation/segmenter.py
# phase 1: extract all fragment blobs from input images.
#
# critique fixes applied:
#   - PCA deskew REMOVED — orientation is now solved jointly by pose graph, not pre-baked.
#     rotating a fragment before matching destroys the correspondence between its
#     pixel coordinates and the ground-truth global frame.
#   - equidistant contour resampling ADDED — uniform spatial resolution is required
#     so that 1d cnn features are position-consistent and cross-attention can compare
#     corresponding points meaningfully.
#   - adaptive threshold ADDED — handles uneven lighting and aged paper better than otsu.

import cv2
import numpy as np
from pathlib import Path
from typing import List
from src.data_models import fragment


# ── public api ────────────────────────────────────────────────────────────────

def load_fragments(input_dir: str, cfg: dict) -> List[fragment]:
    """detect all blobs in every input image, return one fragment per blob."""
    seg = cfg["segmentation"]
    paths = sorted(Path(input_dir).glob("*.png")) + sorted(Path(input_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no images in {input_dir}")

    frags: List[fragment] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        frags.extend(_extract_all(img, p.stem, seg))

    print(f"[segmenter] extracted {len(frags)} fragments from {len(paths)} images")
    return frags


# ── internal ──────────────────────────────────────────────────────────────────

def _extract_all(img: np.ndarray, stem: str, cfg: dict) -> List[fragment]:
    min_area = cfg["min_fragment_area"]
    mask     = _mask(img, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]

    result = []
    for i, cnt in enumerate(valid):
        frag = _build(img, cnt, f"{stem}_{i:03d}", cfg)
        if frag is not None:
            result.append(frag)
    return result


def _build(img: np.ndarray, contour: np.ndarray, fid: str, cfg: dict) -> fragment | None:
    """crop the blob, build rgba, resample contour to uniform density."""
    pad = 12
    x, y, w, h = cv2.boundingRect(contour)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)

    img_crop = img[y0:y1, x0:x1].copy()

    # shift contour to crop-local coordinates
    cnt_local = contour - np.array([[x0, y0]])
    mask_crop = np.zeros(img_crop.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_crop, [cnt_local], -1, 255, cv2.FILLED)

    # extract final contour from local mask (after crop, avoids coord drift)
    contours2, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours2:
        return None
    cnt_clean = max(contours2, key=cv2.contourArea).squeeze(1)  # (n, 2)

    # equidistant resampling: critical for 1d-cnn spatial consistency
    n_pts = cfg["contour_n_points"]
    cnt_uniform = _resample_equidistant(cnt_clean.astype(np.float32), n_pts)
    if cnt_uniform is None:
        return None

    rgba = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask_crop

    return fragment(id=fid, image_rgba=rgba, contour_coords=cnt_uniform)


def _mask(img: np.ndarray, cfg: dict) -> np.ndarray:
    """adaptive threshold mask — handles uneven lighting better than global otsu."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    method = cfg.get("method", "threshold")

    if method == "unet":
        raise NotImplementedError("unet: load weights and implement here")

    # adaptive gaussian threshold: each pixel gets its own threshold based on
    # local neighborhood (block_size × block_size), offset by constant c.
    # THRESH_BINARY_INV: dark ink on light background → foreground = white
    block = cfg.get("adaptive_block", 51)
    C     = cfg.get("adaptive_c",     10)
    mask  = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, C
    )
    # morphological cleanup
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


def _resample_equidistant(pts: np.ndarray, n: int) -> np.ndarray | None:
    """resample a closed contour to exactly n equidistant points via arc-length
    parameterization. this ensures uniform spatial resolution regardless of
    the original opencv contour sampling density, which varies with curvature."""
    if len(pts) < 3:
        return None

    # compute cumulative arc length along the contour
    diffs   = np.diff(pts, axis=0, append=pts[:1])   # wrap around (closed)
    seg_len = np.linalg.norm(diffs, axis=1)           # (m,) segment lengths
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total   = cum_len[-1]

    if total < 1.0:
        return None

    # sample n equally spaced arc-length positions
    target = np.linspace(0.0, total, n, endpoint=False)

    # interpolate x and y independently at target arc-length positions
    xs = np.interp(target, cum_len, np.append(pts[:, 0], pts[0, 0]))
    ys = np.interp(target, cum_len, np.append(pts[:, 1], pts[0, 1]))
    return np.stack([xs, ys], axis=1).astype(np.float32)