# src/utils/debug_viz.py — diagnostic visualizations, only loaded with --debug.

import cv2
import numpy as np
from pathlib import Path
from typing import List
from src.data_models import fragment


def save_segmentation_debug(frags: List[fragment], out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for frag in frags:
        bgr = cv2.cvtColor(frag.image_rgba, cv2.COLOR_RGBA2BGR)
        pts = frag.contour_coords.astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(bgr, [pts], -1, (0, 0, 255), 1)
        # mark every 32nd resampled point to show uniform spacing
        for pt in frag.contour_coords[::32].astype(int):
            cv2.circle(bgr, tuple(pt), 3, (0, 255, 0), -1)
        cv2.imwrite(f"{out_dir}/{frag.id}_seg.png", bgr)
    print(f"[debug] segmentation crops → {out_dir}/")


def save_match_debug(frags: List[fragment], out_dir: str, top_n: int = 3):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    lines = []
    for frag in frags:
        cands = frag.match_candidates or []
        line  = f"{frag.id:30s} → " + ", ".join(
            f"{fid}({s:.3f})" for fid, s in cands[:top_n]
        )
        lines.append(line)
        print(f"[debug] {line}")
    # Fix: Added encoding="utf-8" to support the → character on Windows
    with open(f"{out_dir}/match_scores.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[debug] match scores → {out_dir}/match_scores.txt")


def save_pose_debug(frags: List[fragment], out_dir: str):
    """print pose graph stats: nodes, edges, inlier counts."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    lines = []
    for frag in frags:
        pose = frag.global_pose
        xforms = frag.pairwise_transforms or {}
        edge_summary = "; ".join(
            f"{oid}({n_inl}inl)" for oid, (*_, n_inl) in list(xforms.items())[:3]
        )
        lines.append(f"{frag.id:30s} pose={pose}  edges=[{edge_summary}]")
        print(f"[debug] {lines[-1]}")
    # Fix: Added encoding="utf-8"
    with open(f"{out_dir}/pose_graph.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[debug] pose graph → {out_dir}/pose_graph.txt")


def save_layout_debug(frags: List[fragment], out_dir: str):
    """thumbnail schematic of solved layout."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    placed = [f for f in frags if f.global_pose]
    if not placed:
        return

    xs = [f.global_pose[0] for f in placed]
    ys = [f.global_pose[1] for f in placed]
    ws = [f.image_rgba.shape[1] for f in placed]
    hs = [f.image_rgba.shape[0] for f in placed]
    min_x, min_y = min(xs), min(ys)
    max_x = max(x + w for x, w in zip(xs, ws))
    max_y = max(y + h for y, h in zip(ys, hs))

    scale = 0.08
    cw = max(1, int((max_x - min_x) * scale) + 40)
    ch = max(1, int((max_y - min_y) * scale) + 40)
    canvas = np.full((ch, cw, 3), 240, dtype=np.uint8)
    palette = [(220,80,80),(80,180,80),(80,80,220),(180,180,40),(40,180,180),(180,40,180)]

    for i, frag in enumerate(placed):
        tx, ty, theta = frag.global_pose
        fh, fw = frag.image_rgba.shape[:2]
        x0 = int((tx - min_x) * scale) + 20
        y0 = int((ty - min_y) * scale) + 20
        x1 = x0 + max(2, int(fw * scale))
        y1 = y0 + max(2, int(fh * scale))
        color = palette[i % len(palette)]
        cv2.rectangle(canvas, (x0,y0), (x1,y1), color, -1)
        cv2.rectangle(canvas, (x0,y0), (x1,y1), (0,0,0), 1)
        label = frag.id.split("_")[-1]
        cv2.putText(canvas, label, (x0+2, y0+12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (255,255,255), 1)

    cv2.imwrite(f"{out_dir}/layout_map.png", canvas)
    print(f"[debug] layout map → {out_dir}/layout_map.png")