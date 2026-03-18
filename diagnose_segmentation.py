#!/usr/bin/env python3
"""
diagnose_segmentation.py — visualise what the watershed segmenter sees.

usage:
    python diagnose_segmentation.py path/to/image.tif
    python diagnose_segmentation.py path/to/image.tif --dist-thresh 0.3
    python diagnose_segmentation.py path/to/image.tif --min-dist 50

output:
    debug_blobs.png  — each watershed-separated fragment outlined + numbered
    debug_mask.png   — the foreground mask before watershed
    debug_dist.png   — the distance transform heatmap
"""
import sys
import argparse
import cv2
import numpy as np


def foreground_mask(img, close_size=15, close_iters=3,
                    open_size=5, open_iters=2, otsu_offset=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    gray  = clahe.apply(gray)

    thresh_val, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"  Otsu threshold: {thresh_val:.1f}  (offset: {otsu_offset:+d})")
    if otsu_offset != 0:
        _, mask = cv2.threshold(
            gray, max(0, thresh_val + otsu_offset), 255, cv2.THRESH_BINARY
        )

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iters)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=open_iters)
    return mask


def watershed_labels(fg_mask, dist_thresh=0.4, min_dist=30):
    dist      = cv2.distanceTransform(fg_mask, cv2.DIST_L2, 5)
    dist_norm = dist / (dist.max() + 1e-8)

    _, sure_fg = cv2.threshold(dist_norm, dist_thresh, 255, cv2.THRESH_BINARY)
    sure_fg    = sure_fg.astype(np.uint8)

    k_sep   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_dist, min_dist))
    sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_ERODE, k_sep, iterations=1)

    k_bg    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(fg_mask, k_bg, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers   += 1
    markers[unknown == 255] = 0

    img_3ch = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_3ch, markers)
    markers[markers == 1]  = 0
    markers[markers == -1] = 0
    markers[markers >= 2] -= 1
    return markers, dist_norm


def diagnose(image_path, dist_thresh=0.4, min_dist=30, otsu_offset=0):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"ERROR: could not read {image_path}")
        sys.exit(1)

    h, w   = img.shape[:2]
    area   = h * w
    print(f"\nImage: {image_path}")
    print(f"Resolution: {w} x {h} = {area:,} px²")
    print(f"Settings: dist_thresh={dist_thresh}  min_dist={min_dist}px  "
          f"otsu_offset={otsu_offset:+d}")

    # step 1: foreground mask
    fg = foreground_mask(img, otsu_offset=otsu_offset)
    cv2.imwrite("debug_mask.png", fg)
    print("  Mask saved -> debug_mask.png")

    # step 2: watershed
    labels, dist_norm = watershed_labels(fg, dist_thresh, min_dist)

    # save distance heatmap
    dist_vis = (dist_norm * 255).astype(np.uint8)
    dist_col  = cv2.applyColorMap(dist_vis, cv2.COLORMAP_JET)
    cv2.imwrite("debug_dist.png", dist_col)
    print("  Distance map saved -> debug_dist.png")

    # step 3: report each label
    unique = np.unique(labels)
    unique = unique[unique > 0]
    print(f"\nFound {len(unique)} separated regions after watershed:")
    print(f"{'#':>3}  {'area px²':>12}  {'% image':>8}  "
          f"{'W':>6}  {'H':>6}  {'solidity':>9}  status")
    print("-" * 65)

    blobs = []
    for lbl in unique:
        single = np.zeros(fg.shape, dtype=np.uint8)
        single[labels == lbl] = 255
        cnts, _ = cv2.findContours(single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt       = max(cnts, key=cv2.contourArea)
        blob_area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        hull      = cv2.convexHull(cnt)
        ha        = cv2.contourArea(hull)
        solidity  = blob_area / ha if ha > 0 else 0
        frac      = blob_area / area
        is_real   = frac > 0.005 and solidity > 0.15
        tag       = "REAL ?" if is_real else "noise?"
        blobs.append({"lbl": lbl, "area": blob_area, "frac": frac,
                      "w": bw, "h": bh, "solidity": solidity,
                      "cnt": cnt, "x": x, "y": y, "real": is_real})
        print(f"{lbl:>3}  {blob_area:>12,.0f}  {frac:>7.3%}  "
              f"{bw:>6}  {bh:>6}  {solidity:>9.3f}  {tag}")

    real_count = sum(1 for b in blobs if b["real"])
    print(f"\nEstimated real fragments: {real_count}")
    print()
    print("Tuning hints:")
    print("  Fragments still merged  -> lower  watershed_dist_thresh "
          f"(current: {dist_thresh}) or lower watershed_min_dist (current: {min_dist})")
    print("  Fragments over-split    -> raise  watershed_dist_thresh "
          f"(current: {dist_thresh}) or raise  watershed_min_dist (current: {min_dist})")
    print("  Background in foreground-> raise  otsu_offset "
          f"(current: {otsu_offset:+d}), try +10")
    print("  Fragments missing       -> lower  otsu_offset "
          f"(current: {otsu_offset:+d}), try -10")

    # step 4: visualise
    vis = img.copy()
    rng    = np.random.default_rng(42)
    colors = [tuple(int(c) for c in rng.integers(80, 255, 3)) for _ in blobs]

    for i, b in enumerate(blobs):
        color = colors[i % len(colors)]
        cv2.drawContours(vis, [b["cnt"]], -1, color, 4)
        cx    = b["x"] + b["w"] // 2
        cy    = b["y"] + b["h"] // 2
        label = f"{b['lbl']} ({b['frac']:.1%})"
        cv2.putText(vis, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 0), 5)
        cv2.putText(vis, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 2)

    # scale down for viewing
    max_dim = 2000
    scale   = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
    cv2.imwrite("debug_blobs.png", vis)
    print("\nVisual saved -> debug_blobs.png")
    print("Each numbered outline = one fragment after watershed separation.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--dist-thresh", type=float, default=0.4,
                    help="watershed distance threshold (default 0.4)")
    ap.add_argument("--min-dist",    type=int,   default=30,
                    help="minimum px between watershed seeds (default 30)")
    ap.add_argument("--otsu-offset", type=int,   default=0,
                    help="shift Otsu threshold up(+) or down(-)")
    args = ap.parse_args()
    diagnose(args.image, args.dist_thresh, args.min_dist, args.otsu_offset)