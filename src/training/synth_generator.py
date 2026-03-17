# src/training/synth_generator.py
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict


# ── public api ────────────────────────────────────────────────────────────────

def generate_dataset(
    docs_dir:  str,
    out_dir:   str,
    n_frags:   int   = 6,
    noise_std: float = 5.0,
    seed:      int   = 42,
) -> List[dict]:
    """backwards-compatible entry point — flat image directory, single split."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = _collect_images(docs_dir)
    if not paths:
        raise FileNotFoundError(f"no images found in {docs_dir}")

    metadata = []
    for doc_i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        pairs = _cut_document(img, f"doc{doc_i:05d}", n_frags, noise_std, rng, out)
        metadata.extend(pairs)

    print(f"[synth] generated {len(metadata)} pairs from {len(paths)} documents")
    return metadata


def generate_splits(
    doclayet_root: str,
    out_dir:       str,
    split_budgets: Dict[str, int] = None,
    n_frags:       int   = 6,
    noise_std:     float = 5.0,
    seed:          int   = 42,
) -> Dict[str, List[dict]]:
    """generate train / val / test splits from a DocLayNet directory tree."""
    if split_budgets is None:
        split_budgets = {"train": 4000, "val": 1500, "test": 1500}

    split_img_subdirs = {
        "train": "train_img",
        "val":   "val_img",
        "test":  "test_img",
    }

    root         = Path(doclayet_root)
    out          = Path(out_dir)
    all_metadata = {}

    for split, budget in split_budgets.items():
        img_subdir = split_img_subdirs.get(split, f"{split}_img")
        img_dir    = root / split / img_subdir

        if not img_dir.exists():
            print(f"[synth] WARNING: {img_dir} not found — skipping {split}")
            all_metadata[split] = []
            continue

        paths = _collect_images(str(img_dir))
        if not paths:
            print(f"[synth] WARNING: no images in {img_dir} — skipping {split}")
            all_metadata[split] = []
            continue

        split_out = out / split
        split_out.mkdir(parents=True, exist_ok=True)

        rng   = np.random.default_rng(seed + abs(hash(split)) % (2**31))
        paths = list(paths)
        rng.shuffle(paths)

        metadata  = []
        docs_used = 0

        for doc_i, p in enumerate(paths):
            if len(metadata) >= budget:
                break
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            doc_id = f"{split}_{doc_i:05d}"
            pairs  = _cut_document(img, doc_id, n_frags, noise_std, rng, split_out)
            metadata.extend(pairs)
            docs_used += 1

        meta_path = split_out / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        all_metadata[split] = metadata
        print(
            f"[synth] {split}: {len(metadata)} pairs from {docs_used} documents"
            f" → {meta_path}"
        )

    return all_metadata


# ── internal helpers ──────────────────────────────────────────────────────────

def _collect_images(directory: str) -> List[Path]:
    d     = Path(directory)
    exts  = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths = []
    for pattern in exts:
        paths.extend(d.glob(pattern))
    return sorted(set(paths))


def _cut_document(
    img:       np.ndarray,
    doc_id:    str,
    n_frags:   int,
    noise_std: float,
    rng:       np.random.Generator,
    out:       Path,
) -> List[dict]:
    pieces = [(img, f"{doc_id}_p00")]
    pairs  = []
    cut_i  = 0

    for _ in range(n_frags - 1):
        cuttable = [
            (i, p) for i, p in enumerate(pieces)
            if p[0].shape[0] >= 150 and p[0].shape[1] >= 150
        ]
        if not cuttable:
            break

        idx, (piece_img, piece_id) = cuttable[rng.integers(len(cuttable))]
        pieces.pop(idx)

        ph, pw    = piece_img.shape[:2]
        direction = rng.choice(["horizontal", "vertical", "diagonal"])
        cut_pts   = _bezier_cut(pw, ph, direction, rng, noise_std)

        if cut_pts is None:
            pieces.append((piece_img, piece_id))
            continue

        left_img, right_img = _apply_cut(piece_img, cut_pts, direction)

        if _is_blank(left_img) or _is_blank(right_img):
            pieces.append((piece_img, piece_id))
            continue

        left_id  = f"{piece_id}_L{cut_i}"
        right_id = f"{piece_id}_R{cut_i}"

        cut_resampled = _resample(cut_pts, 512)

        left_path  = out / f"{left_id}.png"
        right_path = out / f"{right_id}.png"
        cv2.imwrite(str(left_path),  left_img)
        cv2.imwrite(str(right_path), right_img)

        pairs.append({
            "left":          str(left_path),
            "right":         str(right_path),
            "cut_pts":       cut_resampled.tolist(),
            "n_match_pts":   512,
            "cut_direction": direction,
            "doc_id":        doc_id,
        })

        pieces.append((left_img,  left_id))
        pieces.append((right_img, right_id))
        cut_i += 1

    return pairs


def _is_blank(img: np.ndarray, threshold: float = 0.02) -> bool:
    total   = img.shape[0] * img.shape[1]
    nonzero = int(np.count_nonzero(img.max(axis=2)))
    return (nonzero / total) < threshold


def _bezier_cut(
    w:         int,
    h:         int,
    direction: str,
    rng:       np.random.Generator,
    noise:     float,
) -> "np.ndarray | None":
    if direction == "horizontal":
        if w < 50:
            return None
        n  = max(w, 64)
        t  = np.linspace(0, 1, n)
        y0 = rng.uniform(0.2, 0.8) * h
        y1 = rng.uniform(0.2, 0.8) * h
        yc = rng.uniform(0.2, 0.8) * h
        x  = t * w
        y  = (1-t)**2 * y0 + 2*t*(1-t)*yc + t**2 * y1
        y += rng.normal(0, noise, n)
        y  = y.clip(2, h - 2)

    elif direction == "vertical":
        if h < 50:
            return None
        n  = max(h, 64)
        t  = np.linspace(0, 1, n)
        x0 = rng.uniform(0.2, 0.8) * w
        x1 = rng.uniform(0.2, 0.8) * w
        xc = rng.uniform(0.2, 0.8) * w
        y  = t * h
        x  = (1-t)**2 * x0 + 2*t*(1-t)*xc + t**2 * x1
        x += rng.normal(0, noise, n)
        x  = x.clip(2, w - 2)

    else:  # diagonal
        if w < 100 or h < 100:
            return None
        n  = max(w + h, 128)
        t  = np.linspace(0, 1, n)
        if rng.random() < 0.5:
            # top edge → right edge
            x0, y0 = rng.uniform(0.1, 0.9) * w, 0.0
            x1, y1 = float(w),                   rng.uniform(0.1, 0.9) * h
        else:
            # left edge → bottom edge
            x0, y0 = 0.0,                         rng.uniform(0.1, 0.9) * h
            x1, y1 = rng.uniform(0.1, 0.9) * w,  float(h)
        xc = (x0 + x1) / 2 + rng.uniform(-0.2, 0.2) * w
        yc = (y0 + y1) / 2 + rng.uniform(-0.2, 0.2) * h
        x  = (1-t)**2 * x0 + 2*t*(1-t)*xc + t**2 * x1
        y  = (1-t)**2 * y0 + 2*t*(1-t)*yc + t**2 * y1
        x += rng.normal(0, noise * 0.5, n)
        y += rng.normal(0, noise * 0.5, n)
        x  = x.clip(0, w)
        y  = y.clip(0, h)

    return np.stack([x, y], axis=1).astype(np.float32)


def _apply_cut(
    img:       np.ndarray,
    cut_pts:   np.ndarray,
    direction: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """split img along cut_pts into two halves.

    ipts[0] is shape (2,) — a 1-D array [x, y].
    access x as first[0], y as first[1].  NOT first[0, 1].

    horizontal  →  mask_a = ABOVE the cut
        poly: TL → TR → cut reversed
    vertical    →  mask_a = LEFT of the cut
        poly: TL → cut → BL
    diagonal top→right   →  mask_a = top-left region
        poly: TL → cut → TR
    diagonal left→bottom →  mask_a = bottom-left region
        poly: TL → BL → cut reversed
    """
    h, w   = img.shape[:2]
    mask_a = np.zeros((h, w), dtype=np.uint8)
    ipts   = cut_pts.astype(np.int32)   # shape (n, 2)

    TL = np.array([[0,     0    ]], dtype=np.int32)
    TR = np.array([[w - 1, 0    ]], dtype=np.int32)
    BL = np.array([[0,     h - 1]], dtype=np.int32)

    if direction == "horizontal":
        poly_a = np.vstack([TL, TR, ipts[::-1]])

    elif direction == "vertical":
        poly_a = np.vstack([TL, ipts, BL])

    else:
        # ipts[0] is 1-D [x, y]  →  y-coordinate is index 1
        first_y = int(ipts[0, 1])        # safe: ipts is 2-D (n,2), so [0,1] is fine
        if first_y < h * 0.3:
            # started on top edge → top-left triangle
            poly_a = np.vstack([TL, ipts, TR])
        else:
            # started on left edge → bottom-left triangle
            poly_a = np.vstack([TL, BL, ipts[::-1]])

    cv2.fillPoly(mask_a, [poly_a], 255)
    mask_b = 255 - mask_a

    part_a = cv2.bitwise_and(img, img, mask=mask_a)
    part_b = cv2.bitwise_and(img, img, mask=mask_b)
    return part_a, part_b


def _resample(pts: np.ndarray, n: int) -> np.ndarray:
    diffs   = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total   = cum_len[-1]
    if total < 1e-6:
        return np.tile(pts[0], (n, 1)).astype(np.float32)
    target  = np.linspace(0, total, n)
    xs      = np.interp(target, cum_len, pts[:, 0])
    ys      = np.interp(target, cum_len, pts[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)