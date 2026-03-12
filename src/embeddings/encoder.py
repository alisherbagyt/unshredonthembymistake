# src/embeddings/encoder.py
# phase 2: extract dense feature embeddings from edge patches.
# uses pretrained resnet18 (no training data needed for mvp).
# upgrade path: change backbone in config.yaml to "vit_b_16" — zero code changes.
# for production: fine-tune with contrastive (nt-xent) loss on matched pairs.

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from typing import List

from src.data_models import fragment


# imagenet normalization — required for pretrained torchvision backbones
_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((64, 64)),           # all patches normalized to 64×64
    T.ToTensor(),
    _normalize,
])


# ── public api ────────────────────────────────────────────────────────────────

def embed_fragments(frags: List[fragment], cfg: dict) -> List[fragment]:
    """extract edge patch embeddings for every fragment in-place."""
    emb_cfg = cfg["embeddings"]
    device = torch.device(emb_cfg.get("device", "cpu"))
    backbone_name = emb_cfg.get("backbone", "resnet18")

    model = _build_backbone(backbone_name, device)

    for frag in frags:
        patches = _sample_edge_patches(
            frag.image_rgba,
            frag.contour_coords,
            patch_size=emb_cfg["patch_size"],
            stride=emb_cfg["patch_stride"],
        )
        if len(patches) == 0:
            # degenerate fragment: assign zero vector so pipeline continues
            frag.edge_embeddings = np.zeros((1, emb_cfg["embedding_dim"]), dtype=np.float32)
            continue

        frag.edge_embeddings = _encode_patches(patches, model, device)

    print(f"[encoder] embedded {len(frags)} fragments")
    return frags


# ── internal helpers ───────────────────────────────────────────────────────────

def _build_backbone(name: str, device: torch.device) -> nn.Module:
    """load pretrained backbone, strip classification head, freeze weights."""
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()           # remove classifier → output is 512-d embedding
    elif name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        model.heads = nn.Identity()        # strip head → 768-d embedding
    else:
        raise ValueError(f"unsupported backbone: {name}")

    model.eval()
    model.to(device)

    # freeze all params — mvp uses as feature extractor only, no fine-tuning
    for p in model.parameters():
        p.requires_grad = False

    return model


def _sample_edge_patches(
    rgba: np.ndarray,
    contour: np.ndarray,
    patch_size: int,
    stride: int,
) -> List[np.ndarray]:
    """extract rgb patches centered on contour points at given stride."""
    h, w = rgba.shape[:2]
    rgb = rgba[:, :, :3]          # drop alpha for encoder input
    half = patch_size // 2
    patches = []

    # sample every `stride` points along the contour to avoid redundancy
    sampled = contour[::stride]

    for x, y in sampled.astype(int):
        # bounds check — skip patches that would go outside image
        if x - half < 0 or y - half < 0 or x + half >= w or y + half >= h:
            continue
        patch = rgb[y - half:y + half, x - half:x + half]
        patches.append(patch)

    return patches


@torch.no_grad()
def _encode_patches(patches: List[np.ndarray], model: nn.Module, device: torch.device) -> np.ndarray:
    """batch-encode patches → (n_patches, embedding_dim) float32 array."""
    # stack into (n, c, h, w) tensor in one shot — avoids python loop overhead
    tensors = torch.stack([_transform(p) for p in patches]).to(device)
    embeddings = model(tensors)                    # (n, d)
    return embeddings.cpu().numpy().astype(np.float32)
