"""Frequency analysis of plant occupancy: the measurement H3 needs.

The proposal's third hypothesis concerns frequency grounding "especially as
relates to positional encodings for 3D spectral features". That mechanism is
already in the model -- :class:`~ggssvt.models.embedding.FourierFeatures` builds
a geometric ladder of bands and both the token embedding and the occupancy
decoder position themselves with it -- but nothing has ever *measured* the
spectral content it is supposed to represent. This module supplies that
measurement.

Three quantities, each answering a different part of the hypothesis.

``radial_power_spectrum``
    Where a specimen's occupancy energy sits in 3D spatial frequency. A thin
    branching sapling and a compact canopy of the same volume have very
    different spectra, and that difference is the physical basis for expecting
    a structure-dependent optimum rather than a universal setting.

``spectral_bandwidth``
    The frequency below which a given fraction of the energy lies, in cycles per
    metre. This is directly comparable against what a Fourier encoding of ``B``
    bands at ``max_freq`` can represent, and against the Nyquist limit set by the
    voxel size -- so "is the encoding wide enough for this plant?" becomes a
    number rather than a guess.

``band_error``
    Reconstruction error decomposed by frequency band. The Frequency Principle
    predicts a network fits low frequencies first; for a branching plant the low
    frequencies are the pot and the trunk, and the high frequencies are the
    branches. If the error concentrates in the high bands and stays there, that
    quantifies *why* thin structure is hard, and connects the spectral argument
    directly to the surface-versus-volume gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import VOXEL_SIZE_M


def _radial_bins(shape: tuple[int, int, int], n_bands: int) -> tuple[np.ndarray, np.ndarray]:
    """Radial frequency index per FFT cell, and the bin edges.

    Frequencies are in cycles per voxel; the caller converts to cycles per metre.
    """
    axes = [np.fft.fftfreq(n) for n in shape]
    grid = np.meshgrid(*axes, indexing="ij")
    radius = np.sqrt(sum(g ** 2 for g in grid))

    # Nyquist is 0.5 cycles/voxel; bin uniformly up to it.
    edges = np.linspace(0.0, 0.5, n_bands + 1)
    index = np.clip(np.digitize(radius, edges) - 1, 0, n_bands - 1)
    return index, edges


@dataclass(frozen=True)
class Spectrum:
    """Radial power spectrum of an occupancy field."""

    power: np.ndarray            # (n_bands,) energy per band, normalised to sum 1
    edges_cycles_per_m: np.ndarray   # (n_bands + 1,)
    voxel_size_m: float

    @property
    def centres_cycles_per_m(self) -> np.ndarray:
        return 0.5 * (self.edges_cycles_per_m[:-1] + self.edges_cycles_per_m[1:])

    @property
    def nyquist_cycles_per_m(self) -> float:
        """The highest frequency this voxel grid can represent at all."""
        return 0.5 / self.voxel_size_m

    def bandwidth(self, fraction: float = 0.95) -> float:
        """Frequency below which ``fraction`` of the energy lies, cycles/metre.

        The number to compare against a Fourier encoding's reach. If a specimen's
        95% bandwidth exceeds what the encoding can represent, the encoding is
        the limiting factor for that specimen and no amount of training fixes it.
        """
        cumulative = np.cumsum(self.power)
        total = cumulative[-1]
        if total <= 0:
            return 0.0
        reached = np.searchsorted(cumulative / total, fraction)
        reached = int(min(reached, self.power.size - 1))
        return float(self.edges_cycles_per_m[reached + 1])

    def high_frequency_share(self, split_cycles_per_m: float) -> float:
        """Fraction of energy above a frequency. Higher means finer structure."""
        above = self.centres_cycles_per_m > split_cycles_per_m
        total = self.power.sum()
        return float(self.power[above].sum() / total) if total > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "power": self.power.tolist(),
            "edges_cycles_per_m": self.edges_cycles_per_m.tolist(),
            "bandwidth_95": self.bandwidth(0.95),
            "bandwidth_99": self.bandwidth(0.99),
            "nyquist_cycles_per_m": self.nyquist_cycles_per_m,
        }


def radial_power_spectrum(
    occupancy: np.ndarray,
    *,
    voxel_size_m: float = VOXEL_SIZE_M,
    n_bands: int = 24,
    remove_mean: bool = True,
) -> Spectrum:
    """Radial power spectrum of a 3D occupancy field.

    Args:
        occupancy: ``(R, R, R)`` boolean or float grid.
        n_bands: radial bins between zero and Nyquist.
        remove_mean: subtract the mean first. Without this the DC term -- which
            is just the total occupied volume -- dominates every spectrum and
            makes all specimens look alike, which is the opposite of what this
            measurement is for.

    Returns:
        A :class:`Spectrum`.
    """
    field = np.asarray(occupancy, dtype=np.float64)
    if remove_mean:
        field = field - field.mean()

    power = np.abs(np.fft.fftn(field)) ** 2
    index, edges = _radial_bins(field.shape, n_bands)

    banded = np.bincount(index.ravel(), weights=power.ravel(), minlength=n_bands)
    total = banded.sum()
    if total > 0:
        banded = banded / total

    return Spectrum(
        power=banded,
        edges_cycles_per_m=edges / voxel_size_m,
        voxel_size_m=voxel_size_m,
    )


def band_error(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    voxel_size_m: float = VOXEL_SIZE_M,
    n_bands: int = 12,
) -> dict:
    """Reconstruction error decomposed by spatial frequency band.

    Tracked across training epochs this tests the Frequency Principle directly:
    the prediction is that low bands converge first and high bands lag. For a
    branching plant the high bands *are* the branches, so a persistent
    high-band error is a measurement of the thin-structure problem rather than
    an anecdote about it.

    Returns:
        Per-band relative error, the band edges, and the share of total error
        sitting above the midpoint frequency.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {truth.shape}")

    residual = np.abs(np.fft.fftn(predicted - truth)) ** 2
    reference = np.abs(np.fft.fftn(truth - truth.mean())) ** 2

    index, edges = _radial_bins(truth.shape, n_bands)
    residual_bands = np.bincount(index.ravel(), weights=residual.ravel(), minlength=n_bands)
    reference_bands = np.bincount(index.ravel(), weights=reference.ravel(), minlength=n_bands)

    relative = residual_bands / np.maximum(reference_bands, 1e-12)
    total = residual_bands.sum()
    midpoint = n_bands // 2

    return {
        "relative_error": relative.tolist(),
        "residual_power": residual_bands.tolist(),
        "edges_cycles_per_m": (edges / voxel_size_m).tolist(),
        "high_band_error_share": float(
            residual_bands[midpoint:].sum() / total if total > 0 else 0.0
        ),
    }


def encoding_reach_cycles_per_m(
    n_bands: int,
    max_freq: float,
    extent_m: float,
) -> dict:
    """What a Fourier positional encoding can actually represent.

    :class:`~ggssvt.models.embedding.FourierFeatures` builds frequencies
    ``2^k`` for ``k`` in ``linspace(0, max_freq, n_bands)``, applied to
    coordinates normalised onto [-1, 1] over the working volume and multiplied
    by pi. The highest representable frequency is therefore ``2^max_freq``
    half-cycles across the extent.

    Comparing this against a specimen's measured bandwidth is what turns "is the
    encoding wide enough?" into a number, and it is the quantity H3's parameter
    efficiency claim is about.

    Returns:
        The top frequency in cycles per metre, the spacing between bands, and
        the parameter cost of the encoding.
    """
    top_half_cycles = 2.0 ** max_freq
    top_cycles_per_m = top_half_cycles / (2.0 * extent_m)

    return {
        "n_bands": n_bands,
        "max_freq_exponent": max_freq,
        "top_cycles_per_m": float(top_cycles_per_m),
        "octaves_per_band": float(max_freq / max(1, n_bands - 1)),
        "encoding_dims": 3 * (2 * n_bands + 1),
    }


def characterise(
    occupancy: np.ndarray,
    *,
    voxel_size_m: float = VOXEL_SIZE_M,
    n_bands: int = 24,
) -> dict:
    """Spectral summary of one specimen, for cross-specimen comparison."""
    spectrum = radial_power_spectrum(
        occupancy, voxel_size_m=voxel_size_m, n_bands=n_bands
    )
    split = spectrum.nyquist_cycles_per_m / 4.0

    return {
        "bandwidth_95": spectrum.bandwidth(0.95),
        "bandwidth_99": spectrum.bandwidth(0.99),
        "high_frequency_share": spectrum.high_frequency_share(split),
        "nyquist_cycles_per_m": spectrum.nyquist_cycles_per_m,
        "split_cycles_per_m": split,
    }


__all__ = [
    "Spectrum",
    "band_error",
    "characterise",
    "encoding_reach_cycles_per_m",
    "radial_power_spectrum",
]
