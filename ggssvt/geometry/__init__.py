"""Calibration-free geometry: floor fitting, rig registration, segmentation, carving."""

from .carving import (
    OccupancyVolume,
    carve,
    carve_specimen,
    largest_connected_component,
    surface_coverage,
)
from .plane import Plane, PlaneFitError, fit_plane_ransac
from .refine import RefinementResult, ViewCorrection, refine_registration
from .rig import (
    RigEstimationError,
    RigSolution,
    ViewPose,
    estimate_rig,
    estimate_view_pose,
    fit_floor,
    nominal_view_pose,
)
from .segment import (
    ViewSegmentation,
    fused_point_cloud,
    multiview_agreement,
    segment_specimen,
    segment_view,
)

__all__ = [
    "OccupancyVolume",
    "Plane",
    "PlaneFitError",
    "RefinementResult",
    "RigEstimationError",
    "RigSolution",
    "ViewCorrection",
    "ViewPose",
    "ViewSegmentation",
    "carve",
    "carve_specimen",
    "estimate_rig",
    "estimate_view_pose",
    "fit_floor",
    "fit_plane_ransac",
    "fused_point_cloud",
    "largest_connected_component",
    "multiview_agreement",
    "nominal_view_pose",
    "refine_registration",
    "segment_specimen",
    "segment_view",
    "surface_coverage",
]
