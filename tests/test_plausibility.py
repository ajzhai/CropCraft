"""The implied-density plausibility check."""

from __future__ import annotations

import math

from ggssvt.eval.plausibility import classify, summarise


def test_a_sensible_volume_for_the_mass_is_plausible():
    # 1 kg in 2 litres -> 500 kg/m^3, squarely fresh plant tissue.
    result = classify("P1", mass_kg=1.0, volume_m3=0.002)

    assert result.plausible
    assert result.density_kg_m3 == 500.0


def test_a_canopy_envelope_is_caught():
    """Mango's carve: 25 L of hull for a 0.74 kg shoot."""
    result = classify("M001", mass_kg=0.74, volume_m3=0.02579)

    assert result.verdict == "envelope"
    assert result.density_kg_m3 < 50


def test_an_unreconstructed_specimen_is_caught():
    """E019: 0.09 L of volume for a 1.95 kg plant."""
    result = classify("E019", mass_kg=1.95, volume_m3=0.00009)

    assert result.verdict == "missing"
    assert result.density_kg_m3 > 10_000


def test_an_empty_volume_is_reported_rather_than_dividing_by_zero():
    result = classify("P0", mass_kg=1.0, volume_m3=0.0)

    assert result.verdict == "no volume"
    assert math.isinf(result.density_kg_m3)
    assert not result.plausible


def test_the_band_edges_are_inclusive():
    assert classify("lo", 1.0, 1.0 / 200.0).plausible
    assert classify("hi", 1.0, 1.0 / 1000.0).plausible


def test_summary_uses_the_median_so_one_outlier_cannot_set_it():
    results = [
        classify("a", 1.0, 0.002),        # 500
        classify("b", 1.0, 0.0025),       # 400
        classify("c", 1.0, 0.00009),      # 11111, the outlier
    ]

    summary = summarise(results)

    assert summary["n"] == 3
    assert summary["n_plausible"] == 2
    assert summary["median_density_kg_m3"] == 500.0
    assert summary["verdicts"]["missing"] == 1


def test_summary_of_nothing_is_not_an_error():
    assert summarise([])["n"] == 0


# --- leverage --------------------------------------------------------------
#
# One specimen with a 190 litre hull took a frozen-feature probe from R2 +0.312
# to -5.2; adding a second similar one brought it back to +0.459. A score can
# therefore look catastrophic for a reason that has nothing to do with the
# method under test, and no summary statistic says so.

def test_one_wild_point_is_flagged_as_dominating():
    import numpy as np

    from ggssvt.eval.metrics import leverage_report

    targets = np.ones(20)
    predictions = targets.copy()
    predictions[7] = 50.0
    report = leverage_report(None, targets, predictions)

    assert report["worst"] == 7
    assert report["dominated"] is True
    assert report["share"] > 0.99


def test_ordinary_scatter_is_not_flagged():
    import numpy as np

    from ggssvt.eval.metrics import leverage_report

    rng = np.random.default_rng(0)
    targets = rng.normal(1.0, 0.4, 40)
    predictions = targets + rng.normal(0, 0.15, 40)
    assert leverage_report(None, targets, predictions)["dominated"] is False


def test_a_perfect_fit_reports_nothing_rather_than_dividing_by_zero():
    import numpy as np

    from ggssvt.eval.metrics import leverage_report

    targets = np.ones(5)
    report = leverage_report(None, targets, targets.copy())
    assert report["worst"] is None
    assert report["dominated"] is False


def test_the_threshold_scales_with_sample_size():
    """At n=8 an even split is 12.5%, so 25% is not yet remarkable."""
    import numpy as np

    from ggssvt.eval.metrics import leverage_report

    targets = np.zeros(8)
    predictions = np.zeros(8)
    predictions[0] = 1.0
    predictions[1:] = 0.5           # the worst point carries 1/(1+7*0.25) = 36%
    report = leverage_report(None, targets, predictions)
    assert report["even_share"] == 0.125
    assert report["dominated"] is False, "4/n = 0.5 at n=8, so 36% is under it"
