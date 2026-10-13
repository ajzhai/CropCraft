"""The experiment log that the project page is generated from."""

from __future__ import annotations

import json

from ggssvt.eval.progress import summarise, survey


def _work_dir(tmp_path):
    (tmp_path / "cache").mkdir()
    (tmp_path / "reports").mkdir()
    return tmp_path


def test_an_empty_work_directory_reports_everything_pending(tmp_path):
    statuses = survey(tmp_path)

    assert statuses, "the programme should never be empty"
    assert not any(s.done for s in statuses)
    assert summarise(statuses)["done"] == 0


def test_an_artefact_marks_its_experiment_done(tmp_path):
    work = _work_dir(tmp_path)
    (work / "cache" / "quality.json").write_text("[]", encoding="utf-8")

    done = {s.key for s in survey(work) if s.done}

    assert "preprocess" in done
    assert "campaign" not in done


def test_a_result_older_than_the_cache_is_flagged_stale(tmp_path):
    """The failure mode this exists for: numbers that predate their targets."""
    import os
    import time

    work = _work_dir(tmp_path)
    (work / "reports" / "metrics.json").write_text("{}", encoding="utf-8")
    old = time.time() - 3600
    os.utime(work / "reports" / "metrics.json", (old, old))
    (work / "cache" / "quality.json").write_text("[]", encoding="utf-8")

    baselines = next(s for s in survey(work) if s.key == "baselines")

    assert baselines.done
    assert baselines.stale
    assert summarise(survey(work))["stale"] >= 1


def test_the_cache_itself_is_never_stale_against_itself(tmp_path):
    work = _work_dir(tmp_path)
    (work / "cache" / "quality.json").write_text("[]", encoding="utf-8")

    preprocess = next(s for s in survey(work) if s.key == "preprocess")

    assert not preprocess.stale


def test_headline_is_read_from_the_artefact(tmp_path):
    work = _work_dir(tmp_path)
    (work / "reports" / "metrics.json").write_text(
        json.dumps({"a": {"rmse_kg": 0.5}, "b": {"rmse_kg": 0.4}}), encoding="utf-8"
    )

    baselines = next(s for s in survey(work) if s.key == "baselines")

    assert baselines.headline == "best RMSE 0.400 kg"


def test_a_corrupt_artefact_does_not_break_the_survey(tmp_path):
    """A half-written JSON must not take the whole page down with it."""
    work = _work_dir(tmp_path)
    (work / "reports" / "metrics.json").write_text("{not json", encoding="utf-8")

    baselines = next(s for s in survey(work) if s.key == "baselines")

    assert baselines.done          # the file exists
    assert baselines.headline is None


def test_gpu_work_is_identified_so_it_can_be_batched(tmp_path):
    statuses = survey(tmp_path)
    gpu = {s.key for s in statuses if s.needs_gpu}

    assert "campaign" in gpu
    assert "posefree" in gpu
    assert "baselines" not in gpu


def test_every_entry_carries_the_command_that_produces_it(tmp_path):
    for status in survey(tmp_path):
        assert status.command.strip(), status.key
        assert status.why.strip(), status.key


def test_statuses_are_json_serialisable(tmp_path):
    payload = [s.as_dict() for s in survey(tmp_path)]

    assert json.loads(json.dumps(payload)) == payload
