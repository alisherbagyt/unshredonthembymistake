# src/utils/debug_viz.py
# debug visualization stub.
# called by pipeline.py when --debug flag is passed.
# saves per-stage debug images to debug_dir for inspection.

import cv2
import numpy as np
from pathlib import Path
from typing import List
from src.data_models import fragment


def save_debug_outputs(frags: List[fragment], debug_dir: str):
    """save all available debug visualizations to debug_dir."""
    out = Path(debug_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_segmentation_debug(frags, debug_dir)
    save_rotation_debug(frags, debug_dir)
    save_match_debug(frags, debug_dir)
    save_pose_debug(frags, debug_dir)


def save_segmentation_debug(frags: List[fragment], debug_dir: str):
    """save each fragment's cropped RGBA image with its contour overlaid."""
    out = Path(debug_dir) / "segmentation"
    out.mkdir(parents=True, exist_ok=True)

    for frag in frags:
        img = frag.image_rgba[:, :, :3].copy()
        pts = frag.contour_coords.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imwrite(str(out / f"{frag.id}_seg.png"), img)

    print(f"[debug] segmentation images → {out}")


def save_rotation_debug(frags: List[fragment], debug_dir: str):
    """save each fragment after canonical rotation with piece_type label."""
    out = Path(debug_dir) / "rotation"
    out.mkdir(parents=True, exist_ok=True)

    for frag in frags:
        img = frag.image_rgba[:, :, :3].copy()
        label = f"{frag.piece_type or '?'} θ={np.degrees(frag.canonical_theta or 0):.1f}°"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / f"{frag.id}_rot.png"), img)

    print(f"[debug] rotation images → {out}")


def save_match_debug(frags: List[fragment], debug_dir: str):
    """print match candidate summary per fragment."""
    out = Path(debug_dir) / "matches.txt"
    lines = []
    for frag in frags:
        cands = frag.match_candidates or []
        lines.append(f"{frag.id}: {len(cands)} candidates")
        for cid, score in cands[:5]:
            lines.append(f"  → {cid}  score={score:.4f}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[debug] match summary → {out}")


def save_pose_debug(frags: List[fragment], debug_dir: str):
    """print global pose summary per fragment."""
    out = Path(debug_dir) / "poses.txt"
    lines = []
    for frag in frags:
        pose = frag.global_pose
        if pose:
            tx, ty, theta = pose
            lines.append(f"{frag.id}: tx={tx:.1f} ty={ty:.1f} θ={np.degrees(theta):.2f}°")
        else:
            lines.append(f"{frag.id}: NO POSE")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[debug] pose summary → {out}")