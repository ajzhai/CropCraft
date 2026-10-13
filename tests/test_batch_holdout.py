"""The batch-holdout scorer: does it split the batches, and does it catch a leak?"""

from __future__ import annotations

import numpy as np
import pytest

from ggssvt.eval.batch_holdout import (
    batch_key,
    batch_names,
    batch_only,
    cross_validate,
    mcnemar,
)

IDS = (
    [f"E{i:03d}" for i in range(1, 11)]
    + [f"E{i:03d}" for i in range(11, 21)]
    + [f"M{i:03d}" for i in range(1, 11)]
    + [f"V{i:03d}" for i in range(1, 9)]
)


def test_block_boundary_is_where_the_capture_sessions_were():
    assert batch_key("E010") == ("E", 0)
    assert batch_key("E011") == ("E", 1)
    assert batch_key("M001") == ("M", 0)
    assert batch_key("V008") == ("V", 0)


def test_labels_span_the_members_present_not_the_block_bounds():
    names = batch_names(IDS)
    assert names["V001"] == "V001-V008"      # eight members, not V001-V010
    assert names["E001"] == "E001-E010"
    assert names["E011"] == "E011-E020"
    assert len(set(names.values())) == 4


def test_an_unparseable_id_gets_its_own_batch_rather_than_joining_one():
    names = batch_names(["E001", "E002", "scratch"])
    assert names["scratch"] == "scratch"
    assert names["E001"] == "E001-E002"


def test_lobo_never_trains_on_a_specimen_from_the_held_out_batch():
    # A feature that is the batch mean and nothing else. Leave-one-out can read
    # it straight off the training fold; leave-one-batch-out cannot.
    rng = np.random.default_rng(0)
    names = batch_names(IDS)
    offsets = {name: 1.0 * i for i, name in enumerate(sorted(set(names.values())))}
    targets = np.array([offsets[names[p]] for p in IDS]) + rng.normal(0, 0.01, len(IDS))
    features = targets.reshape(-1, 1) + rng.normal(0, 0.01, (len(IDS), 1))

    loocv, _ = cross_validate(
        features, targets, IDS, condition="leak", scheme="loocv", components=None)
    lobo, _ = cross_validate(
        features, targets, IDS, condition="leak", scheme="lobo", components=None)

    assert loocv.n_folds == len(IDS)
    assert lobo.n_folds == 4
    assert lobo.smallest_train <= len(IDS) - 8
    # The batch signal is available under loocv and absent under lobo, so the
    # gap the scorer exists to measure must be visible.
    assert loocv.rmse_kg < lobo.rmse_kg


def test_batch_only_reproduces_the_confound_under_loocv_and_loses_it_under_lobo():
    names = batch_names(IDS)
    offsets = {name: 1.0 * i for i, name in enumerate(sorted(set(names.values())))}
    targets = np.array([offsets[names[p]] for p in IDS], dtype=float)

    within, _ = batch_only(targets, IDS, scheme="loocv")
    across, _ = batch_only(targets, IDS, scheme="lobo")

    assert within.r2 > 0.99          # batch membership is the whole signal
    assert across.r2 < 0.05          # and the scheme removes it


def test_mcnemar_uses_only_the_specimens_the_two_screens_disagree_on():
    # Eight pass under A, thirty-one under B, on the same thirty-six specimens.
    a = [True] * 8 + [False] * 28
    b = [True] * 5 + [False] * 3 + [True] * 26 + [False] * 2

    result = mcnemar(a, b)
    assert result["n_pairs"] == 36
    assert result["only_a"] == 3
    assert result["only_b"] == 26
    assert result["discordant"] == 29
    assert result["p_value"] < 0.001


def test_mcnemar_says_nothing_when_the_screens_never_disagree():
    same = [True, False, True, False]
    assert mcnemar(same, same)["p_value"] == 1.0


def test_mcnemar_refuses_an_unpaired_comparison():
    with pytest.raises(ValueError, match="same specimens"):
        mcnemar([True, False], [True])
