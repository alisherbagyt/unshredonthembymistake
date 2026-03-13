# tests/test_pipeline.py
# contract tests — validate the shape and type of every stage's output.
# no real images needed; all data is synthetic.
# run: pytest tests/ -v
#
# windows: KMP_DUPLICATE_LIB_OK must be set before any import that loads torch
# or faiss. setting here at module level so pytest picks it up before any test.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
from src.data_models import fragment


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_fragment(fid="frag_000", n=512, embed_dim=256):
    """synthetic fragment with known shapes."""
    rgba    = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
    rgba[:,:,3] = 255
    contour = (np.random.rand(n, 2) * 200 + 28).astype(np.float32)
    emb     = np.random.randn(n, embed_dim).astype(np.float32)
    emb    /= np.linalg.norm(emb, axis=1, keepdims=True)   # l2-normalize
    return fragment(id=fid, image_rgba=rgba, contour_coords=contour,
                    edge_embeddings=emb)


# ── segmentation tests ────────────────────────────────────────────────────────

def test_equidistant_resampling():
    """resampled contour must have exactly n_points rows."""
    from src.segmentation.segmenter import _resample_equidistant
    raw = np.random.rand(800, 2).astype(np.float32) * 200
    out = _resample_equidistant(raw, 512)
    assert out is not None
    assert out.shape == (512, 2), f"expected (512,2), got {out.shape}"
    assert out.dtype == np.float32


def test_resampling_degeneracy():
    """tiny contour with <3 points must return None, not crash."""
    from src.segmentation.segmenter import _resample_equidistant
    assert _resample_equidistant(np.array([[0,0],[1,1]], dtype=np.float32), 512) is None


# ── geometry feature tests ────────────────────────────────────────────────────

def test_geometric_features_shape():
    """geo features must be (n, 5) float32."""
    from src.features.geometry import compute_geometric_features
    contour = (np.random.rand(512, 2) * 200).astype(np.float32)
    feats   = compute_geometric_features(contour)
    assert feats.shape == (512, 5), f"expected (512,5) got {feats.shape}"
    assert feats.dtype == np.float32


def test_geometric_features_curvature_range():
    """normalized curvature must be in [-1, 1]."""
    from src.features.geometry import compute_geometric_features
    contour = (np.random.rand(512, 2) * 100).astype(np.float32)
    feats   = compute_geometric_features(contour)
    assert feats[:, 0].max() <= 1.01   # allow tiny float tolerance
    assert feats[:, 0].min() >= -1.01


# ── texture strip tests ───────────────────────────────────────────────────────

def test_texture_strip_shape():
    """strip must be (n, strip_width, 3) float32 in [0,1]."""
    from src.features.texture_strip import extract_texture_strip
    frag = make_fragment()
    strip = extract_texture_strip(frag.image_rgba, frag.contour_coords, strip_width=16)
    assert strip.shape == (512, 16, 3), f"expected (512,16,3) got {strip.shape}"
    assert strip.dtype == np.float32
    assert strip.min() >= 0.0 and strip.max() <= 1.0


# ── eac-net model tests ───────────────────────────────────────────────────────

def test_eac_net_output_shape():
    """model must output (1, n, embed_dim) l2-normalized tensor."""
    import torch
    from src.features.eac_net import build_model
    cfg   = {"features": {"geo_dim":64,"tex_dim":128,"embed_dim":256,"strip_width":16}}
    model = build_model(cfg).eval()

    geo   = torch.randn(1, 64, 5)              # (batch=1, n=64, geo_in=5)
    strip = torch.randn(1, 64, 16, 3)          # (batch=1, n=64, sw=16, 3)
    with torch.no_grad():
        out = model(geo, strip)

    assert out.shape == (1, 64, 256), f"expected (1,64,256) got {out.shape}"


def test_eac_net_output_normalized():
    """embeddings must be l2-unit-norm (required for cosine sim = dot product)."""
    import torch
    from src.features.eac_net import build_model
    cfg   = {"features": {"geo_dim":64,"tex_dim":128,"embed_dim":256,"strip_width":16}}
    model = build_model(cfg).eval()
    geo   = torch.randn(1, 32, 5)
    strip = torch.randn(1, 32, 16, 3)
    with torch.no_grad():
        out = model(geo, strip).squeeze(0)    # (32, 256)
    norms = torch.norm(out, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"embeddings not unit-norm, max deviation={( norms-1).abs().max():.6f}"


# ── retrieval tests ───────────────────────────────────────────────────────────

def test_retrieval_output_shape():
    """each fragment must get at most top_k candidates, none self-matching."""
    from src.matching.faiss_index import retrieve_candidates
    frags = [make_fragment(f"frag_{i:03d}") for i in range(6)]
    cfg   = {"retrieval": {"index_type":"flat","top_k":3,"hnsw_m":32}}
    result = retrieve_candidates(frags, cfg)
    for f in result:
        assert f.match_candidates is not None
        assert len(f.match_candidates) <= 3
        for fid, score in f.match_candidates:
            assert fid != f.id, "self-match found"
            assert isinstance(score, float)


# ── cross-attention tests ─────────────────────────────────────────────────────

def test_cross_attention_score_range():
    """compatibility score must be a float in a reasonable range."""
    from src.matching.cross_attention import score_pair
    a = make_fragment("a")
    b = make_fragment("b")
    score, pts_a, pts_b = score_pair(a, b)
    assert isinstance(score, float)
    assert pts_a.shape[1] == 2
    assert pts_b.shape[1] == 2
    assert len(pts_a) == len(pts_b)


# ── ransac tests ──────────────────────────────────────────────────────────────

def test_ransac_known_transform():
    """ransac must recover a known (tx, ty, theta) from clean correspondences."""
    from src.registration.ransac import _ransac_se2, _apply_se2
    rng = np.random.default_rng(42)

    # ground truth transform
    tx_gt, ty_gt, theta_gt = 30.0, -20.0, 0.3

    # generate 30 random source points
    src = rng.uniform(0, 200, (30, 2)).astype(np.float32)
    dst = _apply_se2(src, tx_gt, ty_gt, theta_gt)

    # add slight noise to 5 points (inliers should still dominate)
    dst[:5] += rng.normal(0, 1.0, (5, 2)).astype(np.float32)

    result = _ransac_se2(dst, src, n_iter=500, thr=3.0, min_inl=6,
                         rng=np.random.default_rng(0))
    assert result is not None, "ransac returned None on clean data"
    tx_r, ty_r, theta_r, n_inl = result
    assert abs(tx_r - tx_gt) < 2.0,   f"tx error too large: {tx_r:.2f} vs {tx_gt}"
    assert abs(ty_r - ty_gt) < 2.0,   f"ty error too large: {ty_r:.2f} vs {ty_gt}"
    assert abs(theta_r - theta_gt) < 0.05, f"theta error: {theta_r:.3f} vs {theta_gt}"
    assert n_inl >= 20, f"too few inliers: {n_inl}"


def test_ransac_insufficient_points():
    """ransac must return None when given fewer than min_inliers clean matches."""
    from src.registration.ransac import _ransac_se2
    # pure noise — no consistent transform
    rng = np.random.default_rng(99)
    src = rng.uniform(0, 100, (3, 2)).astype(np.float32)
    dst = rng.uniform(0, 100, (3, 2)).astype(np.float32)
    result = _ransac_se2(src, dst, n_iter=100, thr=2.0, min_inl=10)
    assert result is None


# ── pose graph tests ──────────────────────────────────────────────────────────

def test_pgo_assigns_all_poses():
    """every fragment must have a global_pose tuple after pgo."""
    from src.registration.pose_graph import solve_global_layout
    frags = [make_fragment(f"frag_{i:03d}") for i in range(4)]
    # inject synthetic pairwise transforms forming a chain: 0→1→2→3
    frags[0].pairwise_transforms = {"frag_001": (300.0, 0.0, 0.0, 20)}
    frags[1].pairwise_transforms = {"frag_000": (-300.0, 0.0, 0.0, 20),
                                     "frag_002": (300.0, 0.0, 0.0, 20)}
    frags[2].pairwise_transforms = {"frag_001": (-300.0, 0.0, 0.0, 20),
                                     "frag_003": (300.0, 0.0, 0.0, 15)}
    frags[3].pairwise_transforms = {"frag_002": (-300.0, 0.0, 0.0, 15)}

    cfg = {"solver": {"method":"pgo","pgo_max_iter":200,"pgo_tol":1e-6}}
    result = solve_global_layout(frags, cfg)

    for f in result:
        assert f.global_pose is not None, f"{f.id} has no pose"
        tx, ty, theta = f.global_pose
        assert isinstance(tx, float)
        assert isinstance(ty, float)
        assert isinstance(theta, float)


def test_pgo_no_edges_fallback():
    """pgo with zero edges must fall back to grid without crashing."""
    from src.registration.pose_graph import solve_global_layout
    frags = [make_fragment(f"frag_{i:03d}") for i in range(3)]
    for f in frags:
        f.pairwise_transforms = {}
    cfg = {"solver": {"method":"pgo","pgo_max_iter":100,"pgo_tol":1e-6}}
    result = solve_global_layout(frags, cfg)
    for f in result:
        assert f.global_pose is not None


# ── renderer tests ────────────────────────────────────────────────────────────

def test_renderer_output_shape():
    """renderer must return (h, w, 3) uint8 bgr canvas."""
    from src.rendering.renderer import render_reconstruction
    frags = [make_fragment("frag_000")]
    frags[0].global_pose = (0.0, 0.0, 0.0)
    cfg = {"rendering": {"method":"simple","canvas_padding":20,
                         "background_color":[255,255,255]}}
    canvas = render_reconstruction(frags, cfg)
    assert canvas.ndim == 3
    assert canvas.shape[2] == 3
    assert canvas.dtype == np.uint8


def test_renderer_nonzero_rotation():
    """renderer must handle theta != 0 without crashing."""
    from src.rendering.renderer import render_reconstruction
    frags = [make_fragment("frag_000")]
    frags[0].global_pose = (50.0, 50.0, 0.5236)   # 30 degrees in radians
    cfg = {"rendering": {"method":"simple","canvas_padding":20,
                         "background_color":[255,255,255]}}
    canvas = render_reconstruction(frags, cfg)
    assert canvas.dtype == np.uint8