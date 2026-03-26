# src/features/eac_net.py
# edge-aware contrastive network (eac-net).
#
# critique fix: replaces frozen imagenet resnet (trained to ignore micro-textures)
# with a purpose-built siamese architecture trained on torn-paper data.
#
# architecture — dual-branch:
#   branch 1 (geometry): 1d cnn on curvature/tangent sequence → geo_dim features per point
#   branch 2 (texture):  2d cnn on oriented strip images       → tex_dim features per point
#   fusion: mlp that concatenates both → embed_dim output per contour point
#
# training objective: infonce contrastive loss (see training/trainer.py)
# this model is UNTRAINED at init — run train.py first.
# pipeline works without training (random embeddings) but matching will be random.

import torch
import torch.nn as nn
import numpy as np
from typing import List
from src.data_models import fragment


# ── model definition ──────────────────────────────────────────────────────────

class _geo_branch(nn.Module):
    """1d cnn over the sequence of geometric features (n, geo_in) → (n, geo_dim).
    processes each point in full sequence context via dilated convolutions,
    giving each point a receptive field that spans its local contour neighborhood."""

    def __init__(self, geo_in: int = 5, geo_dim: int = 64):
        super().__init__()
        # dilated 1d convolutions grow the receptive field exponentially
        # without increasing parameter count — critical for sequence modeling
        self.net = nn.Sequential(
            nn.Conv1d(geo_in,   32,  kernel_size=5, padding=2,  dilation=1),
            nn.GELU(),
            nn.Conv1d(32,       64,  kernel_size=5, padding=4,  dilation=2),
            nn.GELU(),
            nn.Conv1d(64,       geo_dim, kernel_size=5, padding=8, dilation=4),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n, geo_in) → transpose to (batch, geo_in, n) for conv1d
        return self.net(x.transpose(1, 2)).transpose(1, 2)  # (batch, n, geo_dim)


class _tex_branch(nn.Module):
    """2d cnn on the oriented strip (n, strip_w, 3) → (n, tex_dim).
    processes each strip patch independently — effectively a shared-weight cnn
    applied at every contour point in parallel."""

    def __init__(self, strip_width: int = 16, tex_dim: int = 128):
        super().__init__()
        # treat each (strip_width, 3) strip as a (3, 1, strip_w) image
        self.net = nn.Sequential(
            nn.Conv1d(3,      32,  kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(32,     64,  kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(4),   # compress spatial dim to 4 fixed slots
            nn.Flatten(1),             # → 64 * 4 = 256
            nn.Linear(256, tex_dim),   nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch * n, strip_w, 3) — pre-flattened batch+contour dims
        # transpose to (batch*n, 3, strip_w) for conv1d
        return self.net(x.permute(0, 2, 1))  # (batch*n, tex_dim)


class eac_net(nn.Module):
    """edge-aware contrastive network.
    inputs:  geo_seq (batch, n, 5) + strip (batch, n, strip_w, 3)
    output:  embeddings (batch, n, embed_dim) — l2-normalized per point"""

    def __init__(self, geo_dim: int = 64, tex_dim: int = 128,
                 embed_dim: int = 256, strip_width: int = 16):
        super().__init__()
        self.geo = _geo_branch(geo_in=5, geo_dim=geo_dim)
        self.tex = _tex_branch(strip_width=strip_width, tex_dim=tex_dim)
        # fusion mlp: concatenated features → final embedding
        self.fuse = nn.Sequential(
            nn.Linear(geo_dim + tex_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, geo: torch.Tensor, strip: torch.Tensor) -> torch.Tensor:
        """
        geo:   (b, n, 5)           — geometric feature sequence
        strip: (b, n, strip_w, 3)  — texture strip per contour point
        returns: (b, n, embed_dim) — l2-normalized embeddings
        """
        b, n = geo.shape[:2]

        geo_feat  = self.geo(geo)                      # (b, n, geo_dim)

        # flatten batch+contour for parallel tex processing, then restore
        strip_flat = strip.reshape(b * n, strip.shape[2], 3)  # (b*n, sw, 3)
        tex_feat   = self.tex(strip_flat).reshape(b, n, -1)   # (b, n, tex_dim)

        fused = self.fuse(torch.cat([geo_feat, tex_feat], dim=-1))  # (b, n, embed_dim)

        # l2-normalize so cosine similarity = dot product — required for infonce
        return nn.functional.normalize(fused, dim=-1)


# ── public api ────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> eac_net:
    """construct eac-net from config."""
    fc = cfg["features"]
    return eac_net(
        geo_dim=fc["geo_dim"],
        tex_dim=fc["tex_dim"],
        embed_dim=fc["embed_dim"],
        strip_width=fc["strip_width"],
    )


def load_model(cfg: dict, device: torch.device) -> eac_net:
    """build model and optionally load trained weights from config path."""
    model = build_model(cfg).to(device).eval()
    path  = cfg["features"].get("weights_path")

    if path:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"[eac-net] loaded weights from {path}")
    else:
        print("[eac-net] no weights_path set — using random init. run train.py first.")

    return model


@torch.no_grad()
def embed_fragments(frags: List[fragment], cfg: dict) -> List[fragment]:
    """run eac-net on every fragment, populate edge_embeddings."""
    from src.features.geometry     import compute_geometric_features
    from src.features.texture_strip import extract_texture_strip

    fc     = cfg["features"]
    device = torch.device(fc.get("device", "cpu"))
    model  = load_model(cfg, device)

    for frag in frags:
        # geometry branch input
        geo_np = compute_geometric_features(frag.contour_coords)  # (n, 5)
        frag.geo_features = geo_np

        # texture branch input
        tex_np = extract_texture_strip(
            frag.image_rgba, frag.contour_coords, fc["strip_width"]
        )   # (n, sw, 3)
        frag.tex_features = tex_np

        # forward pass — add batch dim, remove after
        geo_t   = torch.from_numpy(geo_np).unsqueeze(0).to(device)      # (1, n, 5)
        strip_t = torch.from_numpy(tex_np).unsqueeze(0).to(device)      # (1, n, sw, 3)
        emb     = model(geo_t, strip_t).squeeze(0).cpu().numpy()         # (n, embed_dim)

        frag.edge_embeddings = emb

    print(f"[eac-net] embedded {len(frags)} fragments — shape {frags[0].edge_embeddings.shape}")
    return frags