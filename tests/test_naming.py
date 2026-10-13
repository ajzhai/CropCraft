"""The camB naming conventions and how they resolve to physical azimuths."""

from __future__ import annotations

import pytest

from ggssvt.data.naming import (
    PositionIdError,
    expected_position_ids,
    parse_position,
    resolve_positions,
)


def test_cama_angle_is_taken_literally():
    position = parse_position("camA_030")
    assert position.camera == "camA"
    assert position.azimuth_deg == 30
    assert not position.used_step_angle_convention


def test_camb_protocol_convention_is_taken_literally():
    position = parse_position("camB_210")
    assert position.azimuth_deg == 210
    assert not position.used_step_angle_convention


def test_camb_collector_convention_is_rotated_by_180():
    """``collect_specimen.py`` writes camB files under camA's step angle."""
    position = parse_position("camB_030")
    assert position.azimuth_deg == 210
    assert position.used_step_angle_convention


def test_both_conventions_describe_the_same_twelve_azimuths():
    protocol = {p.azimuth_deg for p in map(parse_position, expected_position_ids("protocol"))}
    collector = {
        p.azimuth_deg for p in map(parse_position, expected_position_ids("collector"))
    }
    assert protocol == collector == set(range(0, 360, 30))


@pytest.mark.parametrize("bad", ["camC_000", "camA_45", "camA_015", "camA_360", "junk"])
def test_malformed_ids_are_rejected(bad):
    with pytest.raises(PositionIdError):
        parse_position(bad)


def test_duplicate_azimuths_are_reported_not_silently_dropped():
    """E001 carries both camB_000 and camB_180, which are the same position."""
    positions, dropped = resolve_positions(["camA_000", "camB_000", "camB_180"])
    assert len(positions) == 2
    assert dropped == ["camB_180"]


def test_positions_come_back_sorted_by_azimuth():
    ids = expected_position_ids("collector")
    positions, _ = resolve_positions(ids)
    azimuths = [p.azimuth_deg for p in positions]
    assert azimuths == sorted(azimuths)
    assert azimuths == list(range(0, 360, 30))
