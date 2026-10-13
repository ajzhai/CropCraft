"""The unattended runner's failure handling.

The whole point of `ggssvt.run_all` is that it survives a night alone, so the
behaviour worth pinning is not that it runs things, it is what it does when one
of them fails. An optional failure must not cost the remaining hours, and a
required failure must not let later steps score a stale cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ggssvt.run_all import Step, _fresh, _tail, programme


def _fail(key: str, *, required: bool) -> Step:
    return Step(key, key, [sys.executable, "-c", "raise SystemExit(3)"],
                None, required)


def _pass(key: str) -> Step:
    return Step(key, key, [sys.executable, "-c", "pass"], None, False)


def test_an_optional_failure_does_not_stop_the_run(monkeypatch, tmp_path):
    """The mesh arm without scikit-image must not cost the campaign."""
    import ggssvt.run_all as mod

    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(mod, "programme",
                        lambda *a, **k: (_fail("optional_boom", required=False),
                                         _pass("after")))
    monkeypatch.setattr(mod, "_summarise",
                        lambda outcomes, *a: seen.extend(
                            (o.key, o.status) for o in outcomes))

    mod.run(work_dir=tmp_path, device="cpu")
    assert dict(seen) == {"optional_boom": "failed", "after": "done"}


def test_a_required_failure_abandons_the_rest(monkeypatch, tmp_path):
    """Nothing downstream may run against a cache that was never rebuilt."""
    import ggssvt.run_all as mod

    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(mod, "programme",
                        lambda *a, **k: (_fail("required_boom", required=True),
                                         _pass("after")))
    monkeypatch.setattr(mod, "_summarise",
                        lambda outcomes, *a: seen.extend(
                            (o.key, o.status) for o in outcomes))

    code = mod.run(work_dir=tmp_path, device="cpu")
    assert dict(seen) == {"required_boom": "failed", "after": "blocked-by"}
    assert code == 1


def test_a_current_artefact_is_skipped_so_a_night_resumes(tmp_path):
    """Re-running after an interruption must not redo eleven hours."""
    artefact = tmp_path / "done.json"
    artefact.write_text("{}", encoding="utf-8")
    step = Step("k", "k", ["true"], artefact)

    assert _fresh(step, artefact.stat().st_mtime - 1)
    # An artefact older than the ground truth it was fitted to is not current.
    assert not _fresh(step, artefact.stat().st_mtime + 1)


def test_a_missing_artefact_always_runs(tmp_path):
    step = Step("k", "k", ["true"], tmp_path / "never_written.json")
    assert not _fresh(step, 0.0)


def test_every_step_key_is_unique():
    """--skip and --only address steps by key, so duplicates would be silent."""
    keys = [s.key for s in programme(Path("wd"), device="cpu", plan="core",
                                     batch_size=2, workers=4)]
    assert len(keys) == len(set(keys))


def test_the_required_steps_are_the_ones_others_read():
    """A step is required exactly when a later step consumes its output."""
    steps = {s.key: s for s in programme(Path("wd"), device="cpu", plan="core",
                                         batch_size=2, workers=4)}
    for key in ("inspect", "preprocess", "gate", "fuse", "baselines"):
        assert steps[key].required, key
    for key in ("mesh", "dino_probe", "posefree", "campaign"):
        assert not steps[key].required, key


def test_tail_survives_a_missing_log(tmp_path):
    assert _tail(tmp_path / "absent.log") == ""


def test_a_gated_skip_reruns_once_access_is_granted(monkeypatch, tmp_path):
    """The DINOv3 cells write {"skipped": ...} rather than failing.

    That leaves a complete-looking artefact, so without this the step would be
    skipped forever and an approval that finally lands would never be used.
    """
    import ggssvt.run_all as mod

    artefact = tmp_path / "dino_probe.json"
    artefact.write_text(
        '{"conditions": {"dinov2": {"rmse_kg": 0.4}, "dinov3": {"skipped": "gated"}}}',
        encoding="utf-8",
    )
    step = Step("dino_probe", "probe", ["true"], artefact, required=False)
    older = artefact.stat().st_mtime - 1

    monkeypatch.setattr("ggssvt.models.backbones.backbone_is_available",
                        lambda *a, **k: (False, "still gated"))
    assert mod._fresh(step, older), "still gated, so do not redo the probe"

    monkeypatch.setattr("ggssvt.models.backbones.backbone_is_available",
                        lambda *a, **k: (True, ""))
    assert not mod._fresh(step, older), "access granted, so the probe must rerun"


def test_an_artefact_with_no_skip_is_left_alone(monkeypatch, tmp_path):
    import ggssvt.run_all as mod

    artefact = tmp_path / "metrics.json"
    artefact.write_text('{"conditions": {"cnn": {"rmse_kg": 0.5}}}', encoding="utf-8")
    step = Step("baselines", "baselines", ["true"], artefact)

    monkeypatch.setattr("ggssvt.models.backbones.backbone_is_available",
                        lambda *a, **k: (True, ""))
    assert mod._fresh(step, artefact.stat().st_mtime - 1)
