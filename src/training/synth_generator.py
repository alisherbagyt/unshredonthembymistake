# src/training/synth_generator.py
# synthetic training data generator.
#
# takes intact document images and procedurally cuts them into fragments using
# random bezier curves, then applies random (x, y, theta) perturbations.
# produces ground-truth fragment pairs where matching contour points are known.
#
# usage: python train.py --generate-data  (see train.py)

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


def generate_dataset(
    docs_dir:  str,
    out_dir:   str,
    n_frags:   int   = 6,
    noise_std: float = 5.0,
    seed:      int   = 42,
) -> List[dict]:
    """cut every document in docs_dir into n_frags fragments, save to out_dir.

    returns list of metadata dicts, each describing one fragment pair with
    ground-truth matching point indices.
    """
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = list(Path(docs_dir).glob("*.png")) + list(Path(docs_dir).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no documents found in {docs_dir}")

    metadata = []
    for doc_i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        pairs = _cut_document(img, doc_i, n_frags, noise_std, rng, out)
        metadata.extend(pairs)

    print(f"[synth] generated {len(metadata)} matching pairs from {len(paths)} documents")
    return metadata


def _cut_document(
    img:       np.ndarray,
    doc_idx:   int,
    n_cuts:    int,
    noise_std: float,
    rng:       np.random.Generator,
    out:       Path,
) -> List[dict]:
    """make n_cuts random bezier cuts through the document, save each piece."""
    h, w   = img.shape[:2]
    pieces = [(img, np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32))]
    pairs  = []

    for cut_i in range(n_cuts - 1):
        # pick a random piece to cut
        idx = rng.integers(len(pieces))
        piece_img, piece_bbox = pieces.pop(idx)

        ph, pw = piece_img.shape[:2]
        if pw < 50 or ph < 50:
            pieces.append((piece_img, piece_bbox))
            continue

        # generate bezier cut path through the piece
        cut_pts = _bezier_cut(pw, ph, rng, noise_std)

        left_img, right_img = _apply_cut(piece_img, cut_pts)

        # ground truth: cut_pts gives the shared boundary —
        # resample to 512 points so indices are aligned between both sides
        cut_resampled = _resample(cut_pts, 512)

        # save fragments
        left_path  = out / f"doc{doc_idx:04d}_cut{cut_i:02d}_L.png"
        right_path = out / f"doc{doc_idx:04d}_cut{cut_i:02d}_R.png"
        cv2.imwrite(str(left_path),  left_img)
        cv2.imwrite(str(right_path), right_img)

        pairs.append({
            "left":         str(left_path),
            "right":        str(right_path),
            "cut_pts":      cut_resampled.tolist(),   # shared boundary in left-frag coords
            "n_match_pts":  len(cut_resampled),
        })

        pieces.append((left_img,  piece_bbox))
        pieces.append((right_img, piece_bbox))

    return pairs


def _bezier_cut(w: int, h: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    """random quadratic bezier curve crossing the piece horizontally."""
    # start at left edge, end at right edge, random control point in the middle
    y0 = rng.uniform(0.1, 0.9) * h
    y1 = rng.uniform(0.1, 0.9) * h
    yc = rng.uniform(0.2, 0.8) * h

    t   = np.linspace(0, 1, max(w, 64))
    # quadratic bezier: B(t) = (1-t)² P0 + 2t(1-t) Pc + t² P1
    x   = t * w
    y   = (1 - t)**2 * y0 + 2*t*(1-t)*yc + t**2*y1

    # add gaussian noise to simulate irregular tear
    y += rng.normal(0, noise, size=len(t))
    y  = y.clip(1, h - 1)

    return np.stack([x, y], axis=1).astype(np.float32)


def _apply_cut(img: np.ndarray, cut_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """split image along cut_pts into left (above) and right (below) masks."""
    h, w   = img.shape[:2]
    mask_l = np.zeros((h, w), dtype=np.uint8)
    mask_r = np.zeros((h, w), dtype=np.uint8)

    # build polygon for left side: cut path + top edge
    top_edge = np.array([[0, 0], [w, 0]], dtype=np.float32)
    poly_l   = np.vstack([top_edge, cut_pts[::-1]]).astype(np.int32)
    cv2.fillPoly(mask_l, [poly_l], 255)

    # right side = inverse
    mask_r = 255 - mask_l

    left  = cv2.bitwise_and(img, img, mask=mask_l)
    right = cv2.bitwise_and(img, img, mask=mask_r)
    return left, right


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    """arc-length resample pts to n equidistant points."""
    diffs   = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    target  = np.linspace(0, cum_len[-1], n)
    xs      = np.interp(target, cum_len, pts[:, 0])
    ys      = np.interp(target, cum_len, pts[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)