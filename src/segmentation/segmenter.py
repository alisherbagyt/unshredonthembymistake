# src/segmentation/segmenter.py
# phase 1: extract all fragment blobs from input images.
#
# key design decisions:
#   - LIGHT-ON-DARK detection: paper fragments are lighter than background.
#     uses THRESH_BINARY (not INV) after background subtraction so white/cream
#     paper on gray/dark background is correctly identified as foreground.
#   - WATERSHED separation: when multiple fragments touch or nearly touch,
#     the distance-transform watershed finds the boundary between them.
#     this is the standard solution for separating touching objects in CV.
#   - two-stage pipeline:
#       stage 1 — coarse mask: global Otsu threshold to find all paper pixels
#       stage 2 — watershed: separate touching fragments using distance peaks
#   - equidistant resampling RETAINED — required for 1d-cnn spatial consistency.
#   - PCA deskew ABSENT (intentional) — orientation solved by pose graph.

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
from src.data_models import fragment


# ── public api ────────────────────────────────────────────────────────────────

def load_fragments(input_dir: str, cfg: dict) -> List[fragment]:
    """detect all blobs in every input image, return one fragment per blob."""
    seg  = cfg["segmentation"]
    exts = ["*.png", "*.jpg", "*.tif", "*.tiff",
            "*.PNG", "*.JPG", "*.TIF", "*.TIFF"]
    paths = []
    for ext in exts:
        paths.extend(sorted(Path(input_dir).glob(ext)))

    # deduplicate (case-insensitive filesystems may double-count)
    seen  = set()
    paths = [p for p in paths
             if not (str(p).lower() in seen or seen.add(str(p).lower()))]

    if not paths:
        raise FileNotFoundError(f"no images in {input_dir}")

    frags: List[fragment] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[segmenter] warning: could not read {p.name} — skipped")
            continue
        new_frags = _extract_all(img, p.stem, seg)
        frags.extend(new_frags)
        print(f"[segmenter]   {p.name}: {len(new_frags)} fragments")

    print(f"[segmenter] total: {len(frags)} fragments from {len(paths)} images")
    return frags


# ── internal ──────────────────────────────────────────────────────────────────

def _extract_all(img: np.ndarray, stem: str, cfg: dict) -> List[fragment]:
    h, w     = img.shape[:2]
    img_area = h * w

    # resolve area thresholds (relative float or legacy absolute int)
    min_area = _resolve_area(cfg["min_fragment_area"],         img_area, "min")
    max_area = _resolve_area(cfg.get("max_fragment_area", 0.95), img_area, "max")

    min_solidity     = float(cfg.get("min_solidity",     0.20))
    max_aspect_ratio = float(cfg.get("max_aspect_ratio", 10.0))

    # ── stage 1: coarse foreground mask ───────────────────────────────────────
    fg_mask = _foreground_mask(img, cfg)

    # ── stage 2: watershed to split touching fragments ────────────────────────
    labels = _watershed_labels(fg_mask, cfg)

    # each unique label > 0 is one separated fragment
    result              = []
    n_rejected_area     = 0
    n_rejected_solidity = 0
    n_rejected_aspect   = 0
    n_rejected_contour  = 0

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]   # skip background (0) and borders (-1)

    for label_id in unique_labels:
        # build binary mask for this single fragment
        single = np.zeros(fg_mask.shape, dtype=np.uint8)
        single[labels == label_id] = 255

        # find contour of this fragment
        contours, _ = cv2.findContours(
            single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            n_rejected_contour += 1
            continue
        cnt  = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        # 1. area filter
        if area < min_area or area > max_area:
            n_rejected_area += 1
            continue

        # 2. solidity filter
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1.0:
            n_rejected_solidity += 1
            continue
        solidity = area / hull_area
        if solidity < min_solidity:
            n_rejected_solidity += 1
            continue

        # 3. aspect ratio filter
        _, _, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > max_aspect_ratio:
            n_rejected_aspect += 1
            continue

        frag = _build(img, cnt, f"{stem}_{label_id:03d}", cfg)
        if frag is not None:
            result.append(frag)

    total_rejected = (n_rejected_area + n_rejected_solidity
                      + n_rejected_aspect + n_rejected_contour)
    if total_rejected:
        print(f"[segmenter]   rejected {total_rejected} blobs "
              f"(area:{n_rejected_area} solidity:{n_rejected_solidity} "
              f"aspect:{n_rejected_aspect})")
    return result


def _resolve_area(cfg_val, img_area: int, kind: str) -> float:
    """convert relative (0-1 float) or absolute area config value to px²."""
    if isinstance(cfg_val, float) and cfg_val <= 1.0:
        return cfg_val * img_area
    val = float(cfg_val)
    if kind == "min" and val < 0.001 * img_area:
        print(f"[segmenter] WARNING: min_fragment_area={val:.0f}px² is very small "
              f"for a {img_area:,}px² image. Consider using a relative value like 0.003.")
    return val


def _foreground_mask(img: np.ndarray, cfg: dict) -> np.ndarray:
    """
    detect paper fragments (light objects) on a dark background.

    strategy:
      1. convert to grayscale
      2. global Otsu threshold — finds the natural light/dark split
         for light paper on dark background, Otsu correctly separates them
      3. morphological closing — fills text/ink holes inside fragments
         so each fragment appears as one solid region
      4. morphological opening — removes isolated noise dots

    why Otsu instead of adaptive threshold here:
      adaptive threshold was designed for ink-on-paper (local contrast).
      for whole-fragment detection (fragment-on-background), the background
      vs paper is a global contrast difference — Otsu handles this correctly.
      adaptive threshold creates a noisy mess on large uniform regions.

    tuning:
      if fragments are not detected (all filtered out):
        lower otsu_offset (try -10 to make threshold more permissive)
      if background leaks into foreground:
        raise otsu_offset (try +10)
      if text holes split fragments:
        increase morph_close_size (try 15 or 21)
      if nearby fragments merge:
        decrease morph_close_size (try 5)
        the watershed stage will handle remaining merges
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # optional brightness equalisation — helps with yellowed/aged paper
    if cfg.get("clahe_equalise", True):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        gray  = clahe.apply(gray)

    # global Otsu threshold — paper (light) becomes foreground (white)
    otsu_offset = int(cfg.get("otsu_offset", 0))
    thresh_val, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if otsu_offset != 0:
        # shift the Otsu threshold up/down manually
        _, mask = cv2.threshold(
            gray, max(0, thresh_val + otsu_offset), 255, cv2.THRESH_BINARY
        )

    # morphological closing — fills text/ink holes inside fragment body
    close_size  = int(cfg.get("morph_close_size",  15))
    close_iters = int(cfg.get("morph_close_iters",  3))
    k_close     = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=close_iters)

    # morphological opening — removes noise dots
    open_size  = int(cfg.get("morph_open_size",  5))
    open_iters = int(cfg.get("morph_open_iters", 2))
    k_open     = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=open_iters)

    return mask


def _watershed_labels(fg_mask: np.ndarray, cfg: dict) -> np.ndarray:
    """
    use the distance-transform watershed to separate touching fragments.

    how it works:
      1. distance transform: every foreground pixel gets a value = distance
         to nearest background pixel. pixels deep inside a fragment score
         high; pixels near the edge score low.
      2. peak finding: local maxima in the distance map = centres of
         individual fragments. each maximum becomes a watershed "seed".
      3. watershed fill: starting from seeds, regions grow outward until
         they meet — the meeting line becomes the boundary between fragments.

    result: a label image where each integer > 0 = one separated fragment,
    0 = background, -1 = watershed boundary between touching fragments.

    tuning:
      too many splits (real fragment cut in half):
        raise watershed_dist_thresh (try 0.5 or 0.6)
        raise watershed_min_dist (try 40 or 60)
      touching fragments not separated:
        lower watershed_dist_thresh (try 0.3)
        lower watershed_min_dist (try 15)
    """
    # distance transform
    dist = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)

    # normalise to [0, 1] for thresholding
    dist_norm = dist / (dist.max() + 1e-8)

    # threshold distance map to find definite fragment centres
    dist_thresh = float(cfg.get("watershed_dist_thresh", 0.4))
    _, sure_fg  = cv2.threshold(dist_norm, dist_thresh, 255, cv2.THRESH_BINARY)
    sure_fg     = sure_fg.astype(np.uint8)

    # minimum distance between peaks — suppresses over-splitting
    min_dist = int(cfg.get("watershed_min_dist", 30))
    # erode sure_fg to separate close seeds
    k_sep    = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (min_dist, min_dist)
    )
    sure_fg  = cv2.morphologyEx(sure_fg, cv2.MORPH_ERODE, k_sep, iterations=1)

    # sure background: dilate foreground mask slightly
    k_bg     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg  = cv2.dilate(fg_mask, k_bg, iterations=3)

    # unknown region: between sure_fg and sure_bg
    unknown  = cv2.subtract(sure_bg, sure_fg)

    # label seeds for watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers   += 1                    # background = 1, fragments start at 2
    markers[unknown == 255] = 0       # unknown region = 0 (watershed will fill)

    # watershed needs a 3-channel uint8 image
    img_3ch  = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    markers  = cv2.watershed(img_3ch, markers)
    # after watershed: -1 = boundaries, 1 = background, ≥ 2 = fragments
    # remap so background = 0, fragments start at 1
    markers[markers == 1]  = 0
    markers[markers == -1] = 0
    markers[markers >= 2] -= 1

    return markers


def _build(img: np.ndarray, contour: np.ndarray,
           fid: str, cfg: dict) -> Optional[fragment]:
    """crop the blob, build rgba, resample contour to uniform density."""
    pad        = 12
    x, y, w, h = cv2.boundingRect(contour)
    x0, y0     = max(0, x - pad), max(0, y - pad)
    x1, y1     = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    img_crop   = img[y0:y1, x0:x1].copy()

    # shift contour to crop-local coordinates
    cnt_local  = contour - np.array([[x0, y0]])
    mask_crop  = np.zeros(img_crop.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_crop, [cnt_local], -1, 255, cv2.FILLED)

    # re-extract contour from local mask (avoids coord drift after crop)
    contours2, _ = cv2.findContours(
        mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours2:
        return None
    cnt_clean = max(contours2, key=cv2.contourArea).squeeze(1)   # (n, 2)

    # equidistant resampling — required for 1d-cnn spatial consistency
    n_pts       = cfg["contour_n_points"]
    cnt_uniform = _resample_equidistant(cnt_clean.astype(np.float32), n_pts)
    if cnt_uniform is None:
        return None

    rgba          = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask_crop
    return fragment(id=fid, image_rgba=rgba, contour_coords=cnt_uniform,
                    crop_offset=(int(x0), int(y0)))


def _resample_equidistant(pts: np.ndarray, n: int) -> Optional[np.ndarray]:
    """
    resample a closed contour to exactly n equidistant points via arc-length
    parameterization. ensures uniform spatial resolution regardless of opencv
    contour sampling density (which varies with curvature).
    """
    if len(pts) < 3:
        return None

    diffs   = np.diff(pts, axis=0, append=pts[:1])   # wrap around (closed)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total   = cum_len[-1]

    if total < 1.0:
        return None

    target = np.linspace(0.0, total, n, endpoint=False)
    xs     = np.interp(target, cum_len, np.append(pts[:, 0], pts[0, 0]))
    ys     = np.interp(target, cum_len, np.append(pts[:, 1], pts[0, 1]))
    return np.stack([xs, ys], axis=1).astype(np.float32)