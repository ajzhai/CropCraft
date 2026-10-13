"""Resume must tell a crash apart from a change of ground truth."""

from __future__ import annotations

import json

import pytest

from ggssvt.campaign import PLANS, RunResult, _result_path


def _write(out_dir, name: str, fingerprint: str) -> None:
    result = RunResult(
        name=name,
        question="fixture",
        status="done",
        metrics={"rmse_kg": 0.5},
        dataset=fingerprint,
    )
    _result_path(name, out_dir).write_text(
        json.dumps(result.__dict__), encoding="utf-8"
    )


def test_a_stored_result_round_trips_with_its_fingerprint(tmp_path):
    _write(tmp_path, "baseline_cnn", "abc123")

    stored = json.loads(_result_path("baseline_cnn", tmp_path).read_text(encoding="utf-8"))

    assert RunResult(**stored).dataset == "abc123"


def test_results_written_before_fingerprinting_are_still_loadable(tmp_path):
    """Older result files have no dataset key and must not crash the resume."""
    path = _result_path("baseline_cnn", tmp_path)
    path.write_text(
        json.dumps({"name": "baseline_cnn", "question": "old", "status": "done"}),
        encoding="utf-8",
    )

    stored = json.loads(path.read_text(encoding="utf-8"))
    result = RunResult(**stored)

    assert result.dataset == ""      # unknown, so resume leaves it alone


@pytest.mark.parametrize("plan", sorted(PLANS))
def test_every_planned_run_has_a_distinct_name(plan):
    """Two runs sharing a name would overwrite each other's results file."""
    names = [run.name for run in PLANS[plan]()]

    assert len(names) == len(set(names))


def test_fingerprint_depends_on_targets_not_just_specimen_names(monkeypatch, tmp_path):
    """The whole point: a changed mass must produce a changed fingerprint."""
    from ggssvt import campaign

    class FakeCached:
        def __init__(self, target_kg: float):
            self.target_kg = target_kg

    masses = {"A": 1.0, "B": 2.0}

    from ggssvt.data import preprocess

    monkeypatch.setattr(preprocess, "usable_plant_ids", lambda *_a, **_k: list(masses))
    monkeypatch.setattr(
        preprocess, "load_cached", lambda pid, *_a, **_k: FakeCached(masses[pid])
    )

    before = campaign.dataset_fingerprint(tmp_path)
    masses["B"] = 0.5                      # the V-batch correction, in miniature
    after = campaign.dataset_fingerprint(tmp_path)

    assert before != after


def test_fingerprint_is_stable_when_nothing_changes(monkeypatch, tmp_path):
    from ggssvt import campaign
    from ggssvt.data import preprocess

    class FakeCached:
        target_kg = 1.25

    monkeypatch.setattr(preprocess, "usable_plant_ids", lambda *_a, **_k: ["A", "B"])
    monkeypatch.setattr(preprocess, "load_cached", lambda *_a, **_k: FakeCached())

    assert campaign.dataset_fingerprint(tmp_path) == campaign.dataset_fingerprint(tmp_path)


def test_fingerprint_ignores_specimen_ordering(monkeypatch, tmp_path):
    from ggssvt import campaign
    from ggssvt.data import preprocess

    class FakeCached:
        def __init__(self, target_kg: float):
            self.target_kg = target_kg

    masses = {"A": 1.0, "B": 2.0}
    order = ["A", "B"]

    monkeypatch.setattr(preprocess, "usable_plant_ids", lambda *_a, **_k: list(order))
    monkeypatch.setattr(
        preprocess, "load_cached", lambda pid, *_a, **_k: FakeCached(masses[pid])
    )

    first = campaign.dataset_fingerprint(tmp_path)
    order = ["B", "A"]
    assert campaign.dataset_fingerprint(tmp_path) == first


# --- reproducibility --------------------------------------------------------

def test_seed_everything_makes_initialisation_repeatable():
    """TrainConfig.seed existed for a long time and nothing consumed it.

    Two runs of the same command gave different weights, different shuffling and
    different dropout, which for a project reporting a 0.209 kg difference with
    a bootstrap interval means the headline number cannot be reproduced.
    """
    import pytest

    torch = pytest.importorskip("torch")
    from ggssvt.models.ggssvt import GGSSVT
    from ggssvt.training.trainer import seed_everything

    def first_weights(seed):
        seed_everything(seed)
        return next(iter(GGSSVT().state_dict().values())).flatten()[:8].clone()

    assert torch.allclose(first_weights(0), first_weights(0))
    assert not torch.allclose(first_weights(0), first_weights(1))


def test_the_training_config_seed_is_actually_wired_to_the_cli():
    """The regression that lets this rot again is the flag quietly going away."""
    import argparse

    from ggssvt.cli import _training_config

    args = argparse.Namespace(seed=1234)
    assert _training_config(args).seed == 1234


def test_the_smoke_plan_does_not_write_the_campaign_s_artefact():
    """A five-minute plumbing check must not mark eight hours as done.

    run_all decides the campaign has already run by looking for
    campaign/summary.txt. The smoke plan used to write exactly that, so an
    unattended run launched after a smoke test would skip the campaign
    entirely and report success.
    """
    from ggssvt.campaign import default_out_dir

    assert default_out_dir("smoke") != default_out_dir("core")
    assert default_out_dir("core") == default_out_dir("full")


def test_run_all_watches_the_directory_the_real_plans_write():
    """The two halves of the trap, pinned together.

    If either the runner's artefact path or the campaign's default output
    directory moves, this fails rather than silently skipping a night.
    """
    from pathlib import Path

    from ggssvt.campaign import default_out_dir
    from ggssvt.run_all import programme

    steps = {s.key: s for s in programme(Path("wd"), device="cpu", plan="core",
                                         batch_size=2, workers=4)}
    watched = steps["campaign"].artefact
    assert watched is not None
    assert watched.name == "summary.txt"
    assert watched.parent.name == default_out_dir("core").name
    assert watched.parent.name != default_out_dir("smoke").name


def test_the_pretrained_state_handed_to_loocv_is_on_the_host():
    """The folds must not share the card with the model that pretrained them.

    `model.state_dict()` returns references to live GPU tensors, and the
    pretrained model used to stay alive through all 38 folds, so the card held
    a spare copy of everything while each fold built its own. On a 16 GB card
    that failed with an out-of-memory error *after* pretraining had already
    succeeded, which is the most expensive place to fail.
    """
    import inspect

    from ggssvt import campaign

    source = inspect.getsource(campaign.execute)
    handoff = source.index("pretrained_state = {")
    call = source.index("pretrained_state=pretrained_state")

    assert 'to("cpu"' in source[handoff:call], "the state is not copied to the host"
    assert "del model" in source[handoff:call], "the pretrained model is not released"
    assert "empty_cache" in source[handoff:call], "the cache is not emptied"


def test_evaluation_runs_without_building_a_graph():
    """`.eval()` switches dropout, not autograd.

    Without no_grad, predicting one specimen built a graph over every
    intermediate activation of a full query grid and held it. That is where the
    campaign died: pretraining finished its 120 epochs, the first evaluation
    forward allocated 14 GiB nobody would ever backpropagate through, and the
    next request failed. The tell was that a 19.3M CNN and a 105.9M ViT failed
    at exactly the same number, so it was activations rather than weights.
    """
    import inspect

    from ggssvt.training import trainer

    source = inspect.getsource(trainer.predict)
    forward = source.index("output = model(")

    assert "with torch.no_grad():" in source[:forward], "evaluation tracks gradients"
    assert "chunk=" in source[forward:], "the query grid is not evaluated in pieces"
