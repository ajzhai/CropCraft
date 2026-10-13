"""GG-SSVT: Geometry-Grounded Self-Supervised Vision Transformer.

Label-efficient volumetric reconstruction and biomass estimation from the
dual-Kinect multi-view RGB-D captures in ``dataset/``.

The package splits into two halves:

* ``ggssvt.data`` / ``ggssvt.geometry`` -- pure NumPy. Loading, rig
  registration, segmentation and space carving. Runs without PyTorch.
* ``ggssvt.models`` / ``ggssvt.training`` -- PyTorch. The GG-SSVT network,
  self-supervised pretraining and biomass fine-tuning.

Importing this package does not import PyTorch.
"""

__version__ = "0.1.0"

__all__ = ["config"]
