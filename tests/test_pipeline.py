# tests/test_pipeline.py
# tests validate contracts between stages, not implementation details.
# each test: given valid input, output shape/type must match the next stage's expectation.
# run: pytest tests/ -v

import numpy as np
import pytest
from src.data_models import fragment


# ── fixtures ──────────────────────────────────────────────────────────────────

def make_fragment(fid="test_001", n_contour=200, n_patches=10, emb_dim=512):
    """build a synthetic fragment with known shapes for testing."""
    rgba = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
    contour = np.random.rand(n_contour, 2).astype(np.float32) * 255
    embeddings = np.random.rand(n_patches, emb_dim).astype(np.float32)
    return fragment(
        id=fid,
        image_rgba=rgba,
        contour_coords=contour,
        edge_embeddings=embeddings,
    )


# ── data model tests ──────────────────────────────────────────────────────────

def test_fragment_repr_no_crash():
    f = make_fragment()
    assert "test_001" in repr(f)


def test_fragment_default_none_fields():
    f = make_fragment()
    assert f.global_pose is None
    assert f.match_candidates is None


# ── retrieval tests ───────────────────────────────────────────────────────────

def test_faiss_retrieval_output_shape():
    """retrieval must return top_k candidates per fragment."""
    from src.retrieval.faiss_index import retrieve_candidates

    frags = [make_fragment(f"frag_{i:03d}") for i in range(5)]
    cfg = {
        "retrieval": {
            "index_type": "flat",
            "top_k": 3,
        }
    }
    result = retrieve_candidates(frags, cfg)

    for f in result:
        assert f.match_candidates is not None, "match_candidates must be set after retrieval"
        assert len(f.match_candidates) <= 3, "must return at most top_k candidates"
        for fid, score in f.match_candidates:
            assert isinstance(fid, str), "candidate id must be string"
            assert isinstance(score, float), "candidate score must be float"
            assert fid != f.id, "self-matches must be excluded"


# ── solver tests ──────────────────────────────────────────────────────────────

def test_solver_assigns_poses_to_all_fragments():
    """every fragment must have a global_pose after solving."""
    from src.solver.matcher import solve_layout

    frags = [make_fragment(f"frag_{i:03d}") for i in range(4)]
    # inject synthetic candidate links so greedy solver can chain them
    frags[0].match_candidates = [("frag_001", 0.95)]
    frags[1].match_candidates = [("frag_000", 0.95), ("frag_002", 0.90)]
    frags[2].match_candidates = [("frag_001", 0.85), ("frag_003", 0.80)]
    frags[3].match_candidates = [("frag_002", 0.75)]

    cfg = {"solver": {"method": "greedy", "min_match_score": 0.7, "max_iterations": 100}}
    result = solve_layout(frags, cfg)

    for f in result:
        assert f.global_pose is not None, f"fragment {f.id} missing global_pose"
        assert len(f.global_pose) == 3, "global_pose must be (x, y, theta)"


def test_solver_pose_types():
    """poses must be numeric floats, not integers or tensors."""
    from src.solver.matcher import solve_layout

    frags = [make_fragment("frag_000"), make_fragment("frag_001")]
    frags[0].match_candidates = [("frag_001", 0.9)]
    frags[1].match_candidates = [("frag_000", 0.9)]

    cfg = {"solver": {"method": "greedy", "min_match_score": 0.7, "max_iterations": 10}}
    result = solve_layout(frags, cfg)

    for f in result:
        x, y, theta = f.global_pose
        assert isinstance(x, float), "x must be float"
        assert isinstance(y, float), "y must be float"
        assert isinstance(theta, float), "theta must be float"


# ── rendering tests ───────────────────────────────────────────────────────────

def test_renderer_returns_bgr_image():
    """renderer must return (h, w, 3) uint8 array."""
    from src.rendering.renderer import render_reconstruction

    frags = [make_fragment("frag_000")]
    frags[0].global_pose = (0.0, 0.0, 0.0)

    cfg = {
        "rendering": {
            "method": "simple",
            "canvas_padding": 10,
            "background_color": [255, 255, 255],
        }
    }
    canvas = render_reconstruction(frags, cfg)
    assert canvas.ndim == 3, "canvas must be 3d"
    assert canvas.shape[2] == 3, "canvas must be bgr (3 channels)"
    assert canvas.dtype == np.uint8, "canvas must be uint8"
