"""Position-id parsing, and the camB azimuth convention.

Two naming conventions exist in ``dataset/plants``:

``camB_180 .. camB_330``
    The convention documented in ``dataset/README.md``. The filename angle is
    already the physical azimuth of camB.

``camB_000 .. camB_150``
    What ``rig_calibration/collect_specimen.py`` actually writes. It names the
    camB file with *camA's* step angle rather than the opposite angle it
    computes for the console message. The physical azimuth is the filename
    angle plus 180.

Both conventions describe the same six physical camB azimuths (180..330), so a
single rule resolves them: camB occupies the upper half-circle, therefore an
azimuth below 180 in a camB filename means the step-angle convention and must
be rotated by 180 degrees. camA is unambiguous in both conventions.

Getting this wrong places camB on top of camA instead of opposite it, which
silently collapses the 12-view rig into a 6-view one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import CAMB_OFFSET_DEG, CAMERAS

_POSITION_RE = re.compile(r"^(?P<cam>cam[AB])_(?P<angle>\d{3})$")


class PositionIdError(ValueError):
    """Raised when a position id does not match the rig naming scheme."""


@dataclass(frozen=True)
class Position:
    """One rig position: which camera, and where it physically stood."""

    position_id: str
    camera: str
    file_angle_deg: int
    azimuth_deg: int

    @property
    def used_step_angle_convention(self) -> bool:
        """True when the filename carried camA's step angle instead of camB's."""
        return self.azimuth_deg != self.file_angle_deg


def parse_position(position_id: str) -> Position:
    """Resolve a position id such as ``camB_030`` to a physical azimuth.

    Raises:
        PositionIdError: if the id is not ``cam[AB]_NNN`` or the angle is not a
            multiple of 30 degrees in [0, 360).
    """
    match = _POSITION_RE.match(position_id)
    if match is None:
        raise PositionIdError(
            f"{position_id!r} is not a rig position id (expected e.g. 'camA_030')"
        )

    camera = match.group("cam")
    angle = int(match.group("angle"))

    if not 0 <= angle < 360:
        raise PositionIdError(f"{position_id!r}: angle {angle} outside [0, 360)")
    if angle % 30 != 0:
        raise PositionIdError(
            f"{position_id!r}: angle {angle} is not a multiple of the 30 degree step"
        )

    if camera == "camB" and angle < CAMB_OFFSET_DEG:
        azimuth = (angle + CAMB_OFFSET_DEG) % 360
    else:
        azimuth = angle

    return Position(
        position_id=position_id,
        camera=camera,
        file_angle_deg=angle,
        azimuth_deg=azimuth,
    )


def resolve_positions(position_ids: list[str]) -> tuple[list[Position], list[str]]:
    """Parse many ids, dropping any whose azimuth duplicates an earlier one.

    ``E001`` carries both ``camB_000`` and ``camB_180``, which resolve to the
    same physical azimuth. The first occurrence wins and the later duplicate is
    returned separately rather than silently dropped, so callers can warn.

    Returns:
        ``(positions, dropped_ids)``.
    """
    positions: list[Position] = []
    dropped: list[str] = []
    seen: dict[tuple[str, int], str] = {}

    for pid in sorted(position_ids):
        position = parse_position(pid)
        key = (position.camera, position.azimuth_deg)
        if key in seen:
            dropped.append(pid)
            continue
        seen[key] = pid
        positions.append(position)

    positions.sort(key=lambda p: (p.azimuth_deg, p.camera))
    return positions, dropped


def expected_position_ids(convention: str = "protocol") -> list[str]:
    """Return the 12 ids a complete capture should contain.

    Args:
        convention: ``"protocol"`` for the ``dataset/README.md`` naming
            (camB_180..330), or ``"collector"`` for what ``collect_specimen.py``
            writes (camB_000..150).
    """
    if convention not in {"protocol", "collector"}:
        raise ValueError(f"unknown convention {convention!r}")

    ids = [f"camA_{a:03d}" for a in range(0, 180, 30)]
    if convention == "protocol":
        ids += [f"camB_{a:03d}" for a in range(180, 360, 30)]
    else:
        ids += [f"camB_{a:03d}" for a in range(0, 180, 30)]
    return ids


__all__ = [
    "CAMERAS",
    "Position",
    "PositionIdError",
    "expected_position_ids",
    "parse_position",
    "resolve_positions",
]
