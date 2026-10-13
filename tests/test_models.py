"""Model components: shapes, invariants, and the properties the method claims."""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from ggssvt.config import MODEL
from ggssvt.models.attention import CrossViewGeometricAttention
from ggssvt.models.backbones import (
    BackboneError,
    CnnBackbone,
    backbone_is_available,
    build_backbone,
)
from ggssvt.models.embedding import FourierFeatures, normalise_world, patch_centroids
from ggssvt.models.ggssvt import GGSSVT
from ggssvt.models.head import BiomassHead

TINY = dataclasses.replace(
    MODEL,
    embed_dim=48,
    encoder_depth=1,
    fusion_depth=1,
    num_heads=2,
    decoder_hidden=32,
    decoder_depth=2,
    head_hidden=16,
    use_checkpointing=False,
)


def _inputs(batch=1, views=4, height=64, width=64, queries=128, seed=0):
    generator = torch.Generator().manual_seed(seed)
    rgb = torch.rand(batch, views, 3, height, width, generator=generator)
    depth = torch.rand(batch, views, 1, height, width, generator=generator) + 0.6
    points = torch.rand(batch, views, 3, height, width, generator=generator) * 0.6 - 0.3
    subject = (
        torch.rand(batch, views, 1, height, width, generator=generator) > 0.5
    ).float()
    query = torch.rand(batch, queries, 3, generator=generator) * 0.6 - 0.3
    return rgb, depth, points, subject, query


def test_fourier_features_output_dimension_and_range():
    fourier = FourierFeatures(n_bands=4, max_freq=3.0)
    encoded = fourier(torch.rand(7, 3))

    assert encoded.shape == (7, fourier.out_dim)
    assert fourier.out_dim == 3 * (2 * 4 + 1)
    # Everything past the raw coordinates is a sine or cosine.
    assert encoded[:, 3:].abs().max() <= 1.0 + 1e-6


def test_normalise_world_maps_the_working_volume_onto_the_unit_cube():
    from ggssvt.config import VOLUME_EXTENT_M

    half = VOLUME_EXTENT_M / 2
    corners = torch.tensor(
        [[-half, -half, 0.0], [half, half, VOLUME_EXTENT_M], [0.0, 0.0, half]]
    )
    normalised = normalise_world(corners)

    assert torch.allclose(normalised[0], torch.tensor([-1.0, -1.0, -1.0]), atol=1e-6)
    assert torch.allclose(normalised[1], torch.tensor([1.0, 1.0, 1.0]), atol=1e-6)
    assert torch.allclose(normalised[2], torch.zeros(3), atol=1e-6)


def test_patch_centroids_average_only_the_valid_pixels():
    points = torch.zeros(1, 1, 3, 4, 4)
    points[..., 0, 0] = 5.0          # one pixel at (5, 5, 5)
    points[:, :, :, 0, 0] = 5.0
    valid = torch.zeros(1, 1, 1, 4, 4, dtype=torch.bool)
    valid[..., 0, 0] = True

    centroids, patch_valid = patch_centroids(points, valid, (1, 1))

    assert patch_valid.all()
    assert torch.allclose(centroids[0, 0, 0], torch.full((3,), 5.0))


def test_patch_with_no_valid_pixels_is_flagged_invalid():
    points = torch.zeros(1, 1, 3, 4, 4)
    valid = torch.zeros(1, 1, 1, 4, 4, dtype=torch.bool)

    _, patch_valid = patch_centroids(points, valid, (1, 1))
    assert not patch_valid.any()


def test_distance_bias_makes_near_tokens_attend_more_strongly():
    """The central claim of the attention design, stated as a test."""
    torch.manual_seed(0)
    attention = CrossViewGeometricAttention(embed_dim=16, num_heads=1, bias_scale=50.0)

    tokens = torch.zeros(1, 3, 16)      # identical appearance everywhere
    anchors = torch.tensor([[[0.0, 0, 0], [0.02, 0, 0], [1.0, 0, 0]]])

    with torch.no_grad():
        query = attention.qkv(tokens)[..., :16].reshape(1, 3, 1, 16).transpose(1, 2)
        keys = query
        logits = (query @ keys.transpose(-2, -1)) / 4.0
        squared = torch.cdist(anchors, anchors).pow(2)
        biased = logits - attention.distance_scale.view(1, 1, 1, 1) * squared.unsqueeze(1)
        weights = biased.softmax(dim=-1)[0, 0, 0]

    # Token 0 should attend to its 2 cm neighbour far more than to the 1 m one.
    assert weights[1] > weights[2] * 100


def test_distance_scale_stays_positive_after_a_large_negative_step():
    attention = CrossViewGeometricAttention(embed_dim=16, num_heads=2, bias_scale=4.0)
    with torch.no_grad():
        attention.log_distance_scale -= 50.0
    assert (attention.distance_scale > 0).all()


def test_forward_shapes():
    model = GGSSVT(config=TINY, tokens_per_view=8).eval()
    rgb, depth, points, subject, query = _inputs()

    with torch.no_grad():
        output = model(rgb, depth, points, subject, query, predict_biomass=True)

    assert output.occupancy_logits.shape == (1, 128)
    assert output.tokens.shape == (1, 4 * 8, TINY.embed_dim)
    assert output.anchors.shape == (1, 4 * 8, 3)
    assert output.mass_kg.shape == (1,)
    assert (output.occupancy >= 0).all() and (output.occupancy <= 1).all()


def test_chunked_decoding_equals_a_single_pass():
    """Queries must not attend to each other, or chunking would change results."""
    model = GGSSVT(config=TINY, tokens_per_view=8).eval()
    rgb, depth, points, subject, query = _inputs(queries=256)

    with torch.no_grad():
        whole = model(rgb, depth, points, subject, query, chunk=1024).occupancy_logits
        pieces = model(rgb, depth, points, subject, query, chunk=17).occupancy_logits

    assert torch.allclose(whole, pieces, atol=1e-5)


def test_gradients_reach_every_trainable_parameter():
    model = GGSSVT(config=TINY, tokens_per_view=8).train()
    rgb, depth, points, subject, query = _inputs()

    output = model(rgb, depth, points, subject, query, predict_biomass=True)
    loss = output.occupancy_logits.square().mean() + output.mass_kg.square().mean()
    loss.backward()

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []


def test_geometry_ablation_removes_the_distance_bias():
    ablated = GGSSVT(config=TINY, tokens_per_view=8, geometry_grounded=False)
    assert torch.all(ablated.fusion.distance_scales() == 0)
    assert ablated.n_parameters() < GGSSVT(config=TINY, tokens_per_view=8).n_parameters()


def test_biomass_head_is_a_pure_physical_model_at_initialisation():
    """Mass must start as exactly density times volume, with no learned offset."""
    head = BiomassHead(embed_dim=8, hidden=8, density_prior_kg_m3=250.0)

    occupancy = torch.zeros(1, 100)
    occupancy[0, :10] = 1.0
    points = torch.zeros(1, 100, 3)
    points[..., 2] = 1.0                       # all above the pot rim
    voxel_volume = 1e-4

    out = head(
        occupancy, points, voxel_volume, torch.zeros(1, 4, 8), torch.ones(1, 4, dtype=bool)
    )

    assert out["volume_m3"].item() == pytest.approx(10 * voxel_volume)
    assert out["density_kg_m3"].item() == pytest.approx(250.0)
    assert out["residual_kg"].item() == pytest.approx(0.0)
    assert out["mass_kg"].item() == pytest.approx(250.0 * 10 * voxel_volume)


def test_biomass_head_ignores_material_below_the_pot_rim():
    from ggssvt.config import POT_HEIGHT_M

    head = BiomassHead(embed_dim=8, hidden=8)
    occupancy = torch.ones(1, 20)
    points = torch.zeros(1, 20, 3)
    points[0, :10, 2] = POT_HEIGHT_M - 0.05    # in the pot
    points[0, 10:, 2] = POT_HEIGHT_M + 0.05    # above it

    out = head(
        occupancy, points, 1e-4, torch.zeros(1, 2, 8), torch.ones(1, 2, dtype=bool)
    )
    assert out["volume_m3"].item() == pytest.approx(10 * 1e-4)


def test_density_modulation_stays_within_its_physical_bounds():
    head = BiomassHead(embed_dim=8, hidden=8, density_prior_kg_m3=200.0, max_density_ratio=4.0)
    torch.nn.init.normal_(head.density_modulation.weight, std=20.0)
    torch.nn.init.normal_(head.density_modulation.bias, std=20.0)

    occupancy = torch.rand(6, 50)
    points = torch.rand(6, 50, 3)
    points[..., 2] += 1.0

    out = head(
        occupancy, points, 1e-4, torch.randn(6, 4, 8), torch.ones(6, 4, dtype=bool)
    )
    assert (out["density_kg_m3"] >= 200.0 / 4.0 - 1e-3).all()
    assert (out["density_kg_m3"] <= 200.0 * 4.0 + 1e-3).all()


def test_parameter_groups_cover_the_stages():
    model = GGSSVT(config=TINY, tokens_per_view=8)
    assert len(model.parameter_groups("pretrain")) == 1
    assert len(model.parameter_groups("finetune")) == 3
    with pytest.raises(ValueError):
        model.parameter_groups("nonsense")


def test_backbone_registry_rejects_unknown_kinds():
    with pytest.raises(BackboneError):
        build_backbone("resnet50")


def test_cnn_backbone_is_the_no_dino_control_and_sees_depth():
    """The control condition must still receive depth and validity."""
    backbone = build_backbone("cnn", embed_dim=32, patch_size=8)
    assert isinstance(backbone, CnnBackbone)
    assert backbone.grid_size(64, 32) == (8, 4)

    rgb = torch.rand(2, 3, 64, 32)
    depth = torch.rand(2, 1, 64, 32)
    valid = torch.ones(2, 1, 64, 32)

    out = backbone(rgb, depth, valid)
    assert out.shape == (2, 32, 8, 4)

    # Changing depth alone must change the output, or the control is RGB-only.
    other = backbone(rgb, depth + 1.0, valid)
    assert not torch.allclose(out, other)


def test_cnn_control_is_fully_trainable():
    backbone = build_backbone("cnn", embed_dim=16, patch_size=8)
    assert backbone.n_frozen_parameters == 0


def test_dinov3_reports_its_access_requirement_rather_than_failing_obscurely():
    available, reason = backbone_is_available("dinov3")
    if not available:
        assert "gated" in reason.lower() or "unavailable" in reason.lower()
        assert "huggingface" in reason.lower()


def test_patch_centroids_align_with_an_arbitrary_token_grid():
    """DINOv2 uses 14-pixel patches and DINOv3 uses 16; anchors must follow."""
    points = torch.rand(1, 1, 3, 28, 28)
    valid = torch.ones(1, 1, 1, 28, 28, dtype=torch.bool)

    for grid in ((2, 2), (7, 7), (4, 14)):
        centroids, patch_valid = patch_centroids(points, valid, grid)
        assert centroids.shape == (1, 1, grid[0] * grid[1], 3)
        assert patch_valid.shape == (1, 1, grid[0] * grid[1])
        assert patch_valid.all()
