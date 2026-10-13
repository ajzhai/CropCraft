"""The view-count ablation's reporting."""

from __future__ import annotations

from ggssvt.eval.view_ablation import (
    VIEW_CACHES,
    ViewCountResult,
    format_table,
    run_ablation,
)


def _result(n_views: int, **overrides) -> ViewCountResult:
    fields = {
        "n_views": n_views,
        "n_usable": 30,
        "n_total": 38,
        "agreement": 0.5,
        "coverage": 0.85,
        "mean_above_ground_l": 100.0,
        "n_plausible": 2,
        "median_density_kg_m3": 12.0,
    }
    fields.update(overrides)
    return ViewCountResult(**fields)


def test_twelve_views_is_the_default_cache():
    """The 12-view carve is the main cache, not a `_v12` directory."""
    assert VIEW_CACHES[12] == "cache"


def test_every_view_count_divides_twelve():
    """A subset that does not divide 12 cannot be evenly spaced around the rig."""
    for n_views in VIEW_CACHES:
        assert 12 % n_views == 0


def test_a_missing_cache_is_reported_not_raised(tmp_path, capsys):
    """Building the caches is a slow separate step; a partial sweep still runs."""
    results = run_ablation(tmp_path, verbose=True)

    assert results == []
    printed = capsys.readouterr().out
    assert "preprocess --views" in printed      # tells the user how to fix it


def test_table_reports_both_counts_as_fractions():
    table = format_table([_result(4, n_usable=25, n_plausible=0)])

    assert "25/38" in table       # usable, out of everything captured
    assert "0/25" in table        # plausible, out of what was usable


def test_table_orders_by_view_count():
    table = format_table([_result(3), _result(4), _result(12)])
    rows = [line for line in table.splitlines() if line.strip()][2:]

    assert [row.split()[0] for row in rows] == ["3", "4", "12"]


def test_as_dict_is_json_safe():
    import json

    payload = json.loads(json.dumps(_result(6).as_dict()))

    assert payload["n_views"] == 6
    assert payload["usable"] == "30/38"
    assert payload["plausible"] == "2/30"
