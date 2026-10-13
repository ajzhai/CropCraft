# DeepVoxels CropCraft integration

The adapter in `ggssvt/eval/deepvoxels_adapter.py` performs the required
dataset conversion:

1. converts CropCraft OpenGL camera-to-world poses to OpenCV poses;
2. estimates the canonical grid barycentre from optical-ray intersection;
3. writes the legacy `intrinsics.txt` contract;
4. supports validated view subsets when the legacy projection rejects a view.

For the E001 smoke test, the valid subset was views `0 1 2 3 4 8 11`, with a
2 m canonical volume and a 0.5 focal calibration factor. The one-epoch
training run then completed seven iterations and wrote a checkpoint. The
legacy runner required one compatibility fix: generator backpropagation must
occur before the discriminator optimizer step under modern PyTorch.

Rendered metric depth must be preserved before the runner's visualization
normalization. The downstream `deepvoxels_biomass.py` utility backprojects
those depths, removes points at or below the specimen-specific `pot_height_m`,
and emits the same above-ground features used by the existing ridge/allometric
biomass evaluator. No DeepVoxels biomass score is reported from the smoke
checkpoint; one epoch is an integration test, not a trained model.
