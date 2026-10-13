"""The upstream API surface the pose-free adapters depend on.

These backends are third-party research repositories pinned to nothing, cloned
from a moving `main`. The adapters were written against documented interfaces and
could not be executed for a long time, so the failure mode to guard against is
not a wrong number -- it is a signature that quietly changed while nobody was
looking, discovered at the start of a booked GPU session.

Each test states one call the adapters make and checks the real function still
accepts it. They skip when the repositories are not installed, so they cost
nothing in the normal environment and fail loudly in the one where it matters.

One drift has already been caught this way: Fast3R's entry point is
``inference_multiview.inference(views, model, device, dtype)``, not
``inference_multiview(model, views, device=...)``, and camera poses come from a
separate ``estimate_camera_poses`` call rather than riding along inside each
prediction.
"""

from __future__ import annotations

import inspect

import pytest


def _accepts(fn, *args, **kwargs) -> None:
    """Assert a call would bind, without executing it."""
    inspect.signature(fn).bind(*args, **kwargs)


dust3r = pytest.importorskip("dust3r", reason="DUSt3R repository not on sys.path")


class TestDust3r:
    def test_load_images_takes_paths_and_a_size(self):
        from dust3r.utils.image import load_images

        _accepts(load_images, ["a.png", "b.png"], size=512)

    def test_make_pairs_takes_a_complete_scene_graph(self):
        from dust3r.image_pairs import make_pairs

        _accepts(
            make_pairs, [], scene_graph="complete", prefilter=None, symmetrize=True
        )

    def test_inference_takes_pairs_first_then_the_model(self):
        from dust3r.inference import inference

        _accepts(inference, [], object(), "cuda", batch_size=1)

    def test_global_aligner_takes_the_output_and_a_mode(self):
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner

        _accepts(
            global_aligner,
            {},
            device="cuda",
            mode=GlobalAlignerMode.PointCloudOptimizer,
        )

    def test_point_cloud_optimizer_exposes_what_the_normaliser_reads(self):
        from dust3r.cloud_opt.optimizer import PointCloudOptimizer

        for name in ("get_im_poses", "get_pts3d", "get_conf"):
            assert hasattr(PointCloudOptimizer, name), name

    def test_compute_global_alignment_accepts_an_init_strategy(self):
        from dust3r.cloud_opt.optimizer import PointCloudOptimizer

        _accepts(
            PointCloudOptimizer.compute_global_alignment,
            None,
            init="mst",
            niter=300,
            schedule="cosine",
            lr=0.01,
        )


class TestMast3r:
    def test_the_metric_model_class_is_importable(self):
        pytest.importorskip("mast3r", reason="MASt3R repository not on sys.path")
        from mast3r.model import AsymmetricMASt3R

        assert hasattr(AsymmetricMASt3R, "from_pretrained")

    def test_mast3r_shares_the_dust3r_inference_path(self):
        """MASt3R vendors dust3r; the adapter reuses the same alignment call."""
        pytest.importorskip("mast3r", reason="MASt3R repository not on sys.path")
        from dust3r.cloud_opt import global_aligner

        assert callable(global_aligner)


class TestFast3r:
    def test_inference_takes_views_first_and_requires_a_dtype(self):
        pytest.importorskip("fast3r", reason="Fast3R not installed")
        import torch
        from fast3r.dust3r.inference_multiview import inference

        _accepts(inference, [], object(), "cuda", dtype=torch.float32, verbose=False)

    def test_there_is_no_function_called_inference_multiview(self):
        """The name the adapter originally used. Kept so the correction sticks."""
        pytest.importorskip("fast3r", reason="Fast3R not installed")
        import fast3r.dust3r.inference_multiview as module

        assert not hasattr(module, "inference_multiview")

    def test_camera_poses_come_from_a_separate_static_call(self):
        pytest.importorskip("open3d", reason="fast3r's lit module imports open3d")
        from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

        _accepts(
            MultiViewDUSt3RLitModule.estimate_camera_poses,
            [],
            niter_PnP=100,
            focal_length_estimation_method="first_view_from_global_head",
        )

    def test_a_pose_count_mismatch_is_refused_rather_than_zipped_short(self):
        """`zip` would silently drop views; the count has to be checked.

        The adapter used to read a pose out of each prediction. It does not
        arrive that way, so poses are now passed alongside -- and a length
        mismatch between the two means the upstream contract moved again.
        """
        pytest.importorskip("fast3r", reason="Fast3R not installed")
        from ggssvt.geometry.pose_free import PoseFreeError
        from ggssvt.geometry.pose_free_backends import _from_multiview

        preds = [{"pts3d_in_other_view": [[0.0, 0.0, 0.0]]}]

        with pytest.raises(PoseFreeError, match="1 predictions but 0 poses"):
            _from_multiview({"preds": preds}, [], "fast3r", ["camA_000"])
