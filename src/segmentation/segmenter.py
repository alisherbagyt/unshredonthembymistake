# src/segmentation/segmenter.py
# phase 1: segment fragments from raw scans + deskew via pca on ink pixels.
# handles two input modes automatically:
#   a) one image per fragment (original mode — one blob per file)
#   b) one image containing MULTIPLE shuffled chunks on a background — detects all blobs
# upgrade path: swap _segment_threshold -> _segment_unet, zero other changes.

import cv2
import numpy as np
from pathlib import Path
from typing import List

from src.data_models import fragment


# -- public api ----------------------------------------------------------------

def load_fragments(input_dir: str, cfg: dict) -> List[fragment]:
    """load images from input_dir.
    if an image contains multiple separated blobs -> extract each as its own fragment.
    if an image is already a single cropped fragment -> treat it as one."""
    seg_cfg = cfg["segmentation"]
    paths = sorted(Path(input_dir).glob("*.png")) + sorted(Path(input_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no images found in {input_dir}")

    frags = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        extracted = _extract_all_fragments(img, p.stem, seg_cfg)
        frags.extend(extracted)

    print(f"[segmenter] loaded {len(frags)} fragments from {input_dir}")
    return frags


# -- multi-blob extraction -----------------------------------------------------

def _extract_all_fragments(img: np.ndarray, stem: str, cfg: dict) -> List[fragment]:
    """find all distinct document chunks in one image and return each as a fragment."""
    min_area = cfg.get("min_fragment_area", 500)
    mask = _segment_threshold(img, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not valid:
        return []

    frags = []
    for i, contour in enumerate(valid):
        frag = _build_fragment_from_contour(img, contour, f"{stem}_{i:03d}", cfg)
        if frag is not None:
            frags.append(frag)

    return frags


def _build_fragment_from_contour(
    img: np.ndarray,
    contour: np.ndarray,
    frag_id: str,
    cfg: dict,
) -> fragment | None:
    """crop a single contour region out of the full image, deskew it, build fragment."""
    pad = 10
    x, y, w, h = cv2.boundingRect(contour)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + h + pad)

    img_crop = img[y0:y1, x0:x1].copy()

    # draw filled mask for just this contour in crop coordinates
    contour_local = contour - np.array([x0, y0])
    mask_crop = np.zeros(img_crop.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_crop, [contour_local], -1, 255, thickness=cv2.FILLED)

    img_deskewed, mask_deskewed, _ = _deskew(img_crop, mask_crop)

    contour_final = _extract_contour(mask_deskewed, cfg)
    if contour_final is None:
        return None

    rgba = cv2.cvtColor(img_deskewed, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask_deskewed

    return fragment(
        id=frag_id,
        image_rgba=rgba,
        contour_coords=contour_final.astype(np.float32),
    )


# -- segmentation backends -----------------------------------------------------

def _segment_threshold(img: np.ndarray, cfg: dict) -> np.ndarray:
    """classical otsu threshold -> clean binary mask."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_val = cfg.get("binary_threshold", 0)

    if thresh_val == 0:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _segment_unet(img: np.ndarray, cfg: dict) -> np.ndarray:
    raise NotImplementedError("unet segmentation: load model weights and implement here")


# -- geometry helpers ----------------------------------------------------------

def _deskew(img: np.ndarray, mask: np.ndarray) -> tuple:
    """rotate fragment so dominant text baseline is horizontal via pca."""
    ys, xs = np.where(mask > 127)
    if len(xs) < 10:
        return img, mask, 0.0

    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    dominant = eigvecs[:, np.argmax(eigvals)]
    angle_deg = np.degrees(np.arctan2(dominant[1], dominant[0]))

    h, w = img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    img_rot = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                              borderValue=(255, 255, 255))
    mask_rot = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST)
    return img_rot, mask_rot, angle_deg


def _extract_contour(mask: np.ndarray, cfg: dict) -> np.ndarray | None:
    """extract largest contour from mask as (n, 2) point array."""
    min_area = cfg.get("min_fragment_area", 500)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    return largest.squeeze(axis=1)