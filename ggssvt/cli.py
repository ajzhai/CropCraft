"""Command-line interface.

    python -m ggssvt.cli inspect        # dataset audit, no computation
    python -m ggssvt.cli access         # HuggingFace account and model access
    python -m ggssvt.cli preprocess     # register, segment, carve, cache
    python -m ggssvt.cli baselines      # LOOCV baselines from the cache
    python -m ggssvt.cli pretrain       # stage 1, self-supervised
    python -m ggssvt.cli loocv          # stage 2 + leave-one-out evaluation
    python -m ggssvt.cli report         # write the result tables and figures
    python -m ggssvt.cli visualise      # rig overlays and mask overlays
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import MODEL, TRAIN, WORK_DIR
from .geometry.fusion import FUSION_RESOLUTION, FUSION_VOXEL_M


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=WORK_DIR / "cache",
        help="where preprocessed specimens live",
    )


def _add_training(parser: argparse.ArgumentParser) -> None:
    """Flags that decide whether a run finishes today or next week."""
    parser.add_argument(
        "--workers",
        type=int,
        default=TRAIN.num_workers,
        help="dataloader workers; 4 keeps a GPU fed, 0 starves it",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=TRAIN.finetune_epochs,
        help=(
            "epochs per leave-one-out fold. This multiplies by the number of "
            "folds, so the default is a multi-day run on one GPU"
        ),
    )
    parser.add_argument(
        "--fourier-bands",
        type=int,
        default=MODEL.fourier_bands,
        help=(
            "frequency bands in the positional encoding. The H3 parameter-"
            "efficiency knob: the encoding currently reaches 83 cycles/m while "
            "the 12 mm grid tops out at 42, so it is over-provisioned"
        ),
    )
    parser.add_argument(
        "--fourier-max-freq",
        type=float,
        default=MODEL.fourier_max_freq,
        help="top exponent of the frequency ladder (2^k). Pairs with --fourier-bands",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TRAIN.seed,
        help=(
            "seeds Python, NumPy and torch. Without it two runs of the same "
            "command give different weights, shuffling and dropout"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAIN.batch_size,
        help=(
            "specimens per step, each carrying 12 views. 1 fits 8 GB; 2 or 4 "
            "roughly halves or quarters epoch time on a 16 GB card"
        ),
    )


def _model_config(args: argparse.Namespace):
    """Build a ModelConfig from the architecture flags."""
    import dataclasses

    return dataclasses.replace(
        MODEL,
        fourier_bands=getattr(args, "fourier_bands", MODEL.fourier_bands),
        fourier_max_freq=getattr(args, "fourier_max_freq", MODEL.fourier_max_freq),
    )


def _training_config(args: argparse.Namespace):
    """Build a TrainConfig from the common training flags."""
    import dataclasses

    return dataclasses.replace(
        TRAIN,
        num_workers=getattr(args, "workers", TRAIN.num_workers),
        finetune_epochs=getattr(args, "finetune_epochs", TRAIN.finetune_epochs),
        batch_size=getattr(args, "batch_size", TRAIN.batch_size),
        seed=getattr(args, "seed", TRAIN.seed),
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    """Audit the raw dataset without running any geometry."""
    from .data import dataset_summary, load_dataset

    specimens = load_dataset(require_ground_truth=False, min_views=0, exclude=())
    print(json.dumps(dataset_summary(specimens), indent=2))

    print("\nPer-specimen:")
    problems = 0
    for specimen in specimens:
        flags = []
        if specimen.n_views != 12:
            flags.append(f"{specimen.n_views} views")
        if specimen.ground_truth is None:
            flags.append("no ground truth")
        if specimen.warnings:
            flags.extend(specimen.warnings)
        if flags:
            problems += 1
            print(f"  {specimen.plant_id}: " + "; ".join(flags))

    if problems == 0:
        print("  (no issues)")

    print("\nCalibration:")
    from .config import CALIB_DIR

    intrinsics = sorted(CALIB_DIR.glob("intrinsics/*_intrinsics.json"))
    extrinsics = sorted(CALIB_DIR.glob("extrinsics/*/rig_positions.json"))
    print(f"  intrinsics files: {len(intrinsics)}")
    print(f"  per-day rig_positions files: {len(extrinsics)}")
    if not intrinsics and not extrinsics:
        print(
            "  none found. Extrinsics are being estimated from the depth data\n"
            "  instead (ggssvt.geometry.rig). Capturing the ChArUco sequence\n"
            "  described in dataset/README.md would replace that estimate with a\n"
            "  measurement."
        )
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    """Check the environment before hours are committed to it."""
    from .eval.preflight import report, run

    print("Pre-flight\n")
    return report(run(device=args.device), verbose=not args.quiet)


def cmd_attention(args: argparse.Namespace) -> int:
    """What the fusion stack attends to, for one specimen."""
    import torch
    from PIL import Image

    from .data.preprocess import usable_plant_ids
    from .eval.attention import read, render_per_view
    from .models.ggssvt import GGSSVT
    from .training.dataset import SpecimenDataset, collate

    if not args.checkpoint.exists():
        print(
            f"No checkpoint at {args.checkpoint}. Attention is a property of a "
            "trained model; run `pretrain` or the campaign first.",
            file=sys.stderr,
        )
        return 2

    plant_id = args.plant or usable_plant_ids(args.cache_dir)[0]
    dataset = SpecimenDataset([plant_id], cache_dir=args.cache_dir, mode="occupancy")
    batch = collate([dataset[0]])

    model = GGSSVT(config=_model_config(args))
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu")["state_dict"], strict=False
    )

    reading = read(model, batch)
    print(f"{plant_id}: {reading.per_view.shape[0]} fusion blocks over "
          f"{reading.n_views} views")
    print(f"  neighbour preference  {reading.neighbour_preference():.3f}")
    print("     above 1 means the fusion prefers views whose frusta overlap,")
    print("     which is what a model that has learned the rig should do.")
    print(f"  distance scales       {reading.distance_scales.mean():.4f} mean")
    print("     near zero means the geometry bias is inert and this is")
    print("     ordinary self-attention; that would be evidence against H2.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_per_view(reading)).save(args.out)
    args.out.with_suffix(".json").write_text(
        json.dumps(reading.as_dict(), indent=2), encoding="utf-8"
    )
    print(f"\nWrote {args.out} and {args.out.with_suffix('.json')}")
    return 0


def cmd_reciprocity(args: argparse.Namespace) -> int:
    """Let the reconstruction refine the masks, and see whether that helps."""
    from .eval.reciprocity import run

    report = run(refine_from=args.refine_from, verbose=not args.quiet)
    control = report["control"]

    print(f"\nControl: re-carving the unchanged masks drifts from the cache by "
          f"{control['median_drift']:.2%} median, {control['max_drift']:.2%} worst.")
    print("A rule that moves the volume by less than that has shown nothing.\n")

    print(f"  {'rule':22s} {'plausible':>10} {'median kg/m3':>13} {'mean vol L':>11}")
    for rule, row in report["summary"].items():
        print(f"  {rule:22s} {str(row['plausible']) + '/' + str(row['n']):>10} "
              f"{row['median_density']:>13} {row['mean_volume_l']:>11}")
    print("\nThe reconstruction is decided from twelve views at once and each "
          "mask from one,\nso re-projecting it gives every view a second "
          "opinion. Which direction should win\nis not obvious in advance, "
          "which is why all four rules are run rather than argued.")
    return 0


def cmd_robustness(args: argparse.Namespace) -> int:
    """H4: depth noise and occlusion, scored physically. No GPU."""
    from .eval.robustness import run

    report = run(seed=args.seed, verbose=not args.quiet)
    print()
    print(f"  {'degradation':18s} {'plausible':>10} {'fragments':>10} "
          f"{'median kg/m3':>13} {'mean vol L':>11}")
    for key, row in sorted(report["summary"].items(),
                           key=lambda kv: (kv[1]["kind"], kv[1]["level"])):
        print(f"  {key:18s} {str(row['plausible']) + '/' + str(row['n']):>10} "
              f"{row['fragments']:>10} {row['median_density']!s:>13} "
              f"{row['mean_volume_l']!s:>11}")
    print()
    print("A fragment is a reconstruction that lost more than half its volume. It")
    print("can still land inside the plausible band by coincidence, so it is")
    print("counted separately rather than as a success. Read that column first.")
    return 0


def cmd_label_efficiency(args: argparse.Namespace) -> int:
    """H1's second half: how many labels each representation needs. No GPU."""
    from .eval.label_efficiency import run

    report = run(exclude=tuple(args.exclude), verbose=not args.quiet)
    comparison = report["comparison"]
    print()
    print(f"Bar: the reference at full labels, {comparison['bar_rmse']} kg "
          f"on {comparison['bar_labels']} labels.")
    for name, labels in comparison["labels_to_reach"].items():
        reached = f"{labels} labels" if labels else "never reaches it"
        print(f"  {name:24s} {reached}")
    print()
    print("Fewer labels than the reference's own count is the label-efficiency")
    print("claim. Equal counts mean a condition is better but not more")
    print("label-efficient, which is H1's second half failing.")
    print("Points marked * are numerically unstable and excluded from the bar.")
    return 0


def cmd_batch_holdout(args: argparse.Namespace) -> int:
    """Leave-one-batch-out beside leave-one-out: how much was the confound? No GPU."""
    from .eval.batch_holdout import run

    report = run(exclude=tuple(args.exclude), verbose=not args.quiet)
    print()
    print(f"{report['n_specimens']} specimens in {len(report['batches'])} batches: "
          + ", ".join(f"{k} ({v})" for k, v in report["batches"].items()))
    print()
    print("Inflation is what leave-one-out gained by keeping the rest of a")
    print("specimen's capture batch in its training fold. A method whose two")
    print("scores are close generalises across sessions; one that collapses")
    print("under lobo was reading batch membership.")
    return 0


def cmd_external(args: argparse.Namespace) -> int:
    """Transfer the regression to the 4TU lettuce set. No GPU, ~4 min first run."""
    from pathlib import Path

    from .eval.external import run

    report = run(
        root=Path(args.root) if args.root else None,
        force=args.force,
        verbose=not args.quiet,
    )
    print()
    print(f"{report['n_after_screen']} of {report['n_plants']} plants after the "
          f"screen, mass {report['mass_range_g'][0]:.1f}-"
          f"{report['mass_range_g'][1]:.1f} g, "
          + ", ".join(f"{k} ({v})" for k, v in report["cultivars"].items()))
    print()
    print("A held-out cultivar is this dataset's leave-one-batch-out. Read the")
    print("two columns together: a method that holds up on a variety it never")
    print("saw is the only one that has transferred.")
    return 0


def cmd_virtual_views(args: argparse.Namespace) -> int:
    """Score carve and fusion against Pheno4D, where the truth is known. No GPU."""
    from pathlib import Path

    from .eval.virtual_views import run

    run(root=Path(args.root) if args.root else None,
        subsample=args.subsample, limit=args.limit, verbose=not args.quiet)
    print()
    print("Virtual views are clean: exact poses, no sensor noise, no")
    print("segmentation error. A reconstruction that fails here fails for")
    print("geometric reasons alone, so this bounds the operator from above")
    print("rather than estimating what our own captures achieve.")
    return 0


def cmd_plots(args: argparse.Namespace) -> int:
    """Figures for the batch holdout, the transfer and the virtual views."""
    from .eval.plots import run

    made = run(verbose=not args.quiet)
    if not made:
        print("nothing to draw: run batch-holdout, external or virtual-views first")
    return 0


def cmd_resolution(args: argparse.Namespace) -> int:
    """The ledger: every comparison, and what its design could have detected."""
    from .eval.resolution import run

    run(verbose=not args.quiet)
    print()
    print("A null means nothing until the design says what it could have found.")
    print("The detectable column is the smallest difference each comparison")
    print("would find four times in five, from its own bootstrap interval.")
    return 0


def cmd_pedestal(args: argparse.Namespace) -> int:
    """Where the carve discarded plant the camera had already photographed."""
    from .eval.pedestal import run

    report = run(verbose=not args.quiet)
    print()
    print(f"{report['n_flagged']} of {report['n_specimens']} specimens have plant")
    print("above the carve. The reported volume for those is the stand they were")
    print("raised on, so it should not be read as a plant until the carve is fixed.")
    return 0


def cmd_pot_mass(args: argparse.Namespace) -> int:
    """Check every pot's mass against the volume the reconstruction gives it."""
    from .eval.pot_mass import run

    run(verbose=not args.quiet)
    print()
    print("A pot of damp medium has a real density. Where the below-rim hull")
    print("holds an unweighed stand the check says nothing, which is every")
    print("specimen whose rim detection fell back to the constant.")
    return 0


def cmd_stages(args: argparse.Namespace) -> int:
    """Encode every stage of every specimen for the project page's stage panel."""
    from .eval.stages import run

    run(limit=args.limit, verbose=not args.quiet)
    print()
    print("Run `cli dashboard` to rebuild the page around it. The first two")
    print("panels are the point: where the segmentation reaches higher than the")
    print("carve, the plant was photographed and the carve discarded it.")
    return 0


def cmd_filmstrip(args: argparse.Namespace) -> int:
    """Render raw frames, masks, depth and volumes for the project page."""
    from .eval.filmstrip import run

    report = run(limit=args.limit, n_views=args.views, verbose=not args.quiet)
    print()
    print("Run `cli dashboard` to rebuild the page around it. The raw frames are")
    print("the part no number can carry: E001's staging is a fact about the")
    print("capture, and it is obvious in the frame and invisible in the volume.")
    print(f"{report['bytes_on_disk'] // 1024} KB of tiles, loaded on demand.")
    return 0


def cmd_surface_mesh(args: argparse.Namespace) -> int:
    """A third operator: volume as occupied surface voxels, after Nombambela."""
    from .eval.surface_mesh import run

    run(verbose=not args.quiet)
    print()
    print("The volume this operator reports scales with the view count, so it is")
    print("comparable only between specimens captured the same way. His ground")
    print("truth is plant and pot together; ours is plant alone.")
    return 0


def cmd_viewpoint(args: argparse.Namespace) -> int:
    """H2: agreement with a view the reconstruction never saw. No GPU."""
    from .eval.viewpoint import run

    report = run(verbose=not args.quiet)
    summary = report["summary"]
    print()
    print(f"  in-sample IoU   {summary['in_sample_iou']:.4f}   "
          f"(views the reconstruction was built from)")
    print(f"  held-out IoU    {summary['held_out_iou']:.4f}   "
          f"(a view it never saw)")
    print(f"  gap             {summary['iou_gap']:+.4f}   "
          f"({summary['relative_drop']:.1%} of the in-sample score)")
    print()
    print("The gap is what H2 is about. A hull agrees with its own silhouettes")
    print("by construction, so the in-sample column is not evidence of anything.")
    print("A small gap means the reconstruction generalises to new viewpoints.")
    return 0


def cmd_frequency(args: argparse.Namespace) -> int:
    """H3's measurement: what spectral content the occupancy target actually has.

    The campaign's band ladder is justified against the grid Nyquist, and this is
    what turns that from an argument about voxel size into something measured on
    the data. Run it before the H3 runs, so the ladder has a reason rather than a
    rationale.
    """
    from .data.preprocess import load_cached, usable_plant_ids
    from .eval.frequency import characterise, encoding_reach_cycles_per_m

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)
    rows = []
    for plant_id in plant_ids:
        cached = load_cached(plant_id, args.cache_dir)
        row = characterise(cached.occupancy, voxel_size_m=cached.voxel_size_m)
        row["plant_id"] = plant_id
        rows.append(row)

    nyquist = float(np.median([r["nyquist_cycles_per_m"] for r in rows]))
    from .config import VOLUME_EXTENT_M

    encoding = encoding_reach_cycles_per_m(
        MODEL.fourier_bands, MODEL.fourier_max_freq, VOLUME_EXTENT_M
    )
    reach = encoding["top_cycles_per_m"]
    b95 = float(np.median([r["bandwidth_95"] for r in rows]))
    b99 = float(np.median([r["bandwidth_99"] for r in rows]))

    print(f"Radial power spectrum over {len(rows)} specimens, medians:")
    print(f"  95% of the power below      {b95:8.1f} cycles/m")
    print(f"  99% of the power below      {b99:8.1f} cycles/m")
    print(f"  grid Nyquist                {nyquist:8.1f} cycles/m")
    print(f"  encoding reaches            {reach:8.1f} cycles/m "
          f"({encoding['n_bands']} bands, {encoding['encoding_dims']} dims)")
    print()
    if reach > nyquist:
        print(f"The encoding reaches {reach / nyquist:.1f} times the grid Nyquist, so")
        print("its top octaves address detail the voxel grid cannot represent. That")
        print("is the over-provisioning H3 predicts can be cut at no cost.")
    else:
        print("The encoding sits at or below the grid Nyquist, so H3's premise does")
        print("not hold on this configuration and the band ladder needs restating.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "median_bandwidth_95": round(b95, 2),
        "median_bandwidth_99": round(b99, 2),
        "grid_nyquist_cycles_per_m": round(nyquist, 2),
        "encoding_reach_cycles_per_m": round(reach, 2),
        "encoding": encoding,
        "per_specimen": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.out}")
    return 0


def cmd_funnel(args: argparse.Namespace) -> int:
    """The staged screening figure, drawn from whatever has actually run."""
    from .eval.funnel import write

    path = write(args.out)
    print(f"Wrote {path}")
    print("Every count is read from the report JSON, so the figure cannot")
    print("disagree with the results it summarises.")
    return 0


def cmd_neural_field(args: argparse.Namespace) -> int:
    """A trained neural field as a third reconstruction operator.

    Sweeps the density threshold rather than choosing one, because that
    threshold has no physical calibration and picking a value would be picking
    the answer.
    """
    from .eval.neural_field import run

    if not args.outputs.exists():
        print(
            f"No Nerfstudio outputs at {args.outputs}. Export the transforms "
            "with `cli nerfstudio`, then train in the `cropcraft` environment; "
            "the export prints the ns-train command lines.",
            file=sys.stderr,
        )
        return 2

    report = run(args.outputs, cache_dir=args.cache_dir, device=args.device,
                 verbose=not args.quiet)
    agreement = report["consensus"]

    print()
    print(f"  specimens scored              {agreement['n_specimens']}")
    print(f"  no threshold works for        "
          f"{len(agreement['specimens_with_no_working_threshold'])}")
    shared = agreement["shared_thresholds"]
    if shared:
        print(f"  plausible for every specimen  {shared[0]:.3g} to {shared[-1]:.3g}")
        print()
        print("A single density threshold serves the whole set, so it is a")
        print("calibration rather than a fitted parameter. That is a measured")
        print("density scale for this sensor and subject.")
    else:
        best = agreement["best_threshold"]
        covers = agreement["best_threshold_covers"]
        print(f"  best single threshold         {best:.3g}, covering {covers}")
        print()
        print("No single threshold is plausible for every specimen, so there is")
        print("no calibration here: a per-specimen value is a fitted parameter")
        print("and would be worth nothing on a new plant. If most specimens have")
        print("no working threshold at all, the envelope finding generalises")
        print("from silhouette hulls to neural fields.")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Look at any layer of any specimen, with every knob on the command line."""
    from .data.preprocess import usable_plant_ids
    from .eval.viz import COLORMAPS, LAYERS, SOURCES, VizConfig, save, show

    config = VizConfig(
        layers=tuple(args.layers),
        views=tuple(args.views),
        nerf_output=args.nerf_output,
        threshold=args.threshold,
        sweep_thresholds=tuple(args.sweep_thresholds),
        view_index=args.view,
        size=args.size,
        point_radius=args.point_radius,
        supersample=args.supersample,
        cmap=args.cmap,
        source=args.source,
        max_points=args.max_points,
        label=not args.no_label,
        backend=args.backend,
    )

    plant_ids = args.plants or usable_plant_ids(config.cache_dir)[:1]
    if not plant_ids:
        print(f"No usable specimens in {config.cache_dir}.", file=sys.stderr)
        return 2

    if config.backend in ("matplotlib", "text"):
        for plant_id in plant_ids:
            if config.backend == "text":
                print(f"\n{plant_id}")
            show(plant_id, config)
        return 0

    for plant_id in plant_ids:
        out = args.out_dir / f"{plant_id}_{'_'.join(config.layers)}.png"
        save(plant_id, out, config)
        print(f"  {plant_id} -> {out}")

    print()
    print(f"layers    {', '.join(LAYERS)}")
    print(f"sources   {', '.join(SOURCES)}")
    print(f"colormaps {', '.join(COLORMAPS)}")
    print("Jet and rainbow are deliberately absent: they put luminance edges at")
    print("arbitrary values, and a reader reads those as structure in the plant.")
    return 0


def cmd_access(args: argparse.Namespace) -> int:
    """Report which account is authenticated and what it can actually download.

    Authentication is ambient: `hf auth login` writes a token that
    huggingface_hub and transformers pick up on their own, so nothing in this
    codebase takes a token argument and no script should contain one. What goes
    wrong is almost never the token itself -- it is being logged in as a
    different account than the one that was approved.
    """
    from .geometry.sam3d import SAM3_REPO, SAM3D_OBJECTS_REPO, SAM_REPOS
    from .models.backbones import DINOV2_REPOS, DINOV3_REPOS, repo_access

    try:
        from huggingface_hub import get_token, whoami
    except ImportError:
        print("huggingface_hub is not installed", file=sys.stderr)
        return 2

    token = get_token()
    if not token:
        print("No HuggingFace token found.")
        print("  Interactive:  hf auth login")
        print("  Headless:     export HF_TOKEN=hf_...")
        print("Gated models will be unavailable until one of those is set.")
    else:
        try:
            account = whoami().get("name", "?")
            print(f"Authenticated as: {account}")
        except Exception as exc:
            print(f"Token present but rejected: {str(exc)[:120]}", file=sys.stderr)
            return 1
        import os

        # HF_TOKEN wins over the cached login, so a stale environment variable
        # silently overrides `hf auth login` and you stay on the wrong account
        # no matter how many times you log in again.
        source = (
            "HF_TOKEN environment variable"
            if os.environ.get("HF_TOKEN")
            else "cached login (~/.cache/huggingface)"
        )
        print(f"Token source: {source}")

    checks: list[tuple[str, str]] = []
    for variant, repo in DINOV2_REPOS.items():
        checks.append((f"dinov2-{variant}", repo))
    for variant, repo in DINOV3_REPOS.items():
        checks.append((f"dinov3-{variant}", repo))
    for variant, repo in SAM_REPOS.items():
        checks.append((f"sam-{variant}", repo))
    checks.append(("sam3", SAM3_REPO))
    checks.append(("sam-3d-objects", SAM3D_OBJECTS_REPO))

    print()
    blocked = []
    for label, repo in checks:
        accessible, reason = repo_access(repo)
        print(f"  {'OK     ' if accessible else 'BLOCKED'}  {label:16s} {repo}")
        if not accessible:
            blocked.append((label, reason))

    if blocked:
        print(f"\n{len(blocked)} repositories are not accessible to this account.")
        print("If the HuggingFace settings page shows them ACCEPTED, the usual")
        print("cause is being logged in here as a different account than the one")
        print("that was approved. Check the name printed above.")
    else:
        print("\nEverything this project can use is accessible.")
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    from .data.preprocess import preprocess_dataset

    if args.segmenter == "sam3d" and args.cache_dir == WORK_DIR / "cache":
        print(
            "Refusing to overwrite the geometric cache with SAM3D output. "
            "Pass --cache-dir work_dirs/ggssvt/cache_sam3d so both conditions "
            "can be compared.",
            file=sys.stderr,
        )
        return 2

    report = preprocess_dataset(
        require_ground_truth=not getattr(args, "include_unlabelled", False),
        cache_dir=args.cache_dir,
        plant_ids=args.plants,
        seed=args.seed,
        segmenter=args.segmenter,
        sam_model=args.sam_model,
        sam_device=args.sam_device,
        n_views=args.views,
    )
    usable = sum(1 for q in report if q.is_usable())
    print(f"\n{usable}/{len(report)} specimens passed the quality gate.")
    print(f"Quality report: {args.cache_dir / 'quality.json'}")
    return 0


def cmd_fuse(args: argparse.Namespace) -> int:
    """TSDF-fuse every specimen and score the result against the carve."""
    import time

    from .config import voxel_grid_centres
    from .data.preprocess import load_cached, usable_plant_ids
    from .eval.fusion_features import fusion_features
    from .eval.plausibility import classify, summarise

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)
    heights = voxel_grid_centres()[..., 2]

    print(f"Fusing {len(plant_ids)} specimens at {args.voxel * 1000:.0f} mm")
    print("This is slow: the field is evaluated once per view over the whole grid.\n")

    table, started = {}, time.time()
    for index, plant_id in enumerate(plant_ids, start=1):
        cached = load_cached(plant_id, args.cache_dir)
        features = fusion_features(
            cached, resolution=args.resolution, voxel_size_m=args.voxel
        )
        carved = float(
            (cached.occupancy & (heights > cached.pot_height_m)).sum()
        ) * cached.voxel_size_m ** 3
        features["carve_above_rim_m3"] = carved
        features["mass_kg"] = float(cached.target_kg)
        table[plant_id] = features

        print(
            f"  [{index:2d}/{len(plant_ids)}] {plant_id}  "
            f"carve {carved * 1000:7.2f} L   "
            f"fused {features['tsdf_above_rim_m3'] * 1000:6.2f} L   "
            f"coverage {features['tsdf_coverage']:.3f}"
        )

    print(f"\nfused in {(time.time() - started) / 60:.1f} min\n")

    for label, key in [
        ("carve (visual hull)", "carve_above_rim_m3"),
        ("TSDF fusion", "tsdf_above_rim_m3"),
    ]:
        checks = [
            classify(pid, row["mass_kg"], row[key]) for pid, row in table.items()
        ]
        summary = summarise(checks)
        print(
            f"  {label:22s} {summary['n_plausible']:2d}/{summary['n']} plausible, "
            f"median {summary['median_density_kg_m3']:.1f} kg/m3"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.out}")

    if args.write_cache:
        from .eval.fusion_features import write_fused_cache

        print(f"\nWriting a fused cache to {args.write_cache}")
        print("At 128^3 and 12 mm, matching the carve, so the only variable")
        print("between the two caches is the reconstruction method.\n")
        n = write_fused_cache(plant_ids, args.cache_dir, args.write_cache)
        print(f"\nwrote {n} specimens")
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    """Reconstruction metrics: re-projection accuracy and cross-method agreement."""
    import json as _json

    from .eval.reconstruction_quality import run, summarise

    rows = run(args.plants, carve_dir=args.cache_dir, out=args.out)
    print()
    print(_json.dumps(summarise(rows), indent=2))
    print()
    print("Read the silhouette IoU carefully. A visual hull is by construction")
    print("consistent with every silhouette it was built from, so this metric")
    print("structurally favours the carve, and it does. Physical plausibility and")
    print("biomass accuracy both favour the fusion. See FINDINGS.md section 7f.")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    """Run the acceptance checks and exit non-zero if anything is blocked."""
    import csv as _csv
    import json as _json

    from .config import GROUND_TRUTH_CSV
    from .data.preprocess import load_cached, usable_plant_ids
    from .eval.gates import check_reconstruction, check_segmentation, summarise

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)
    masses = {
        row["plant_id"]: float(row["net_weight_g"]) / 1000.0
        for row in _csv.DictReader(
            GROUND_TRUTH_CSV.open(newline="", encoding="utf-8")
        )
    }

    reports = []
    for plant_id in plant_ids:
        cached = load_cached(plant_id, args.cache_dir)
        for report in (
            check_segmentation(cached),
            check_reconstruction(cached, mass_kg=masses.get(plant_id)),
        ):
            reports.append(report)
            for failure in (c for c in report.checks if not c.passed):
                mark = "BLOCK" if failure.blocking else "note "
                detail = ""
                if failure.value is not None and failure.threshold is not None:
                    detail = f"  ({failure.value:.4g} vs {failure.threshold:.4g})"
                # Check names are phrased as the condition that should hold, so
                # printing one bare for a *failed* check reads as though it
                # passed. Every line here is an unmet condition; say so.
                print(f"  {mark} {plant_id}  not met: {failure.name}{detail}")
                if failure.blocking and failure.message:
                    print(f"        {failure.message}")

    summary = summarise(reports)
    print(
        f"\n{summary['checks_run']} checks over {len(plant_ids)} specimens: "
        f"{summary['blocked']} blocked, {summary['advisories']} advisories"
    )
    if summary["blocked_subjects"]:
        print("blocked: " + ", ".join(sorted(set(summary["blocked_subjects"]))))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            _json.dumps([r.as_dict() for r in reports], indent=2), encoding="utf-8"
        )
        print(f"Saved to {args.out}")

    # Non-zero on a blocking failure so this can sit in front of a long run.
    return 1 if summary["blocked"] else 0


def cmd_dino_segment(args: argparse.Namespace) -> int:
    """Lift DINOv2 features onto the carved points and cluster them."""
    import json as _json

    from .eval.dino_segment import run, summarise

    print(f"Lifting {args.backbone} patch features onto carved points, DITR-style.")
    print("Unsupervised: this dataset has no per-point labels, so the")
    print("supervised head is replaced by clustering.\n")

    results = run(
        args.plants, cache_dir=args.cache_dir, variant=args.variant,
        backbone_kind=args.backbone, out=args.out
    )
    print()
    print(_json.dumps(summarise(results), indent=2))
    print(f"\nSaved to {args.out}")
    return 0


def cmd_recon_metrics(args: argparse.Namespace) -> int:
    """Distance metrics on Pheno4D, where reference geometry exists."""
    from .eval.recon_metrics import run

    report = run(verbose=not args.quiet)
    print()
    print("Accuracy, completeness and F-score, after the form DUSt3R reports")
    print("for DTU. These need reference geometry, so they apply to Pheno4D and")
    print("to none of the 36 captures; the density screen is still the only")
    print("measure that transfers to those.")
    print(f"F-score and voxel IoU agree on {report['f_score_agrees_with_voxel_iou']}"
          f" of {report['n_scans']} scans.")
    return 0


def cmd_backbone_viz(args: argparse.Namespace) -> int:
    """Render what DINOv2 and DINOv3 each see, side by side."""
    from .eval.backbone_viz import run

    run(plant_ids=tuple(args.plants) if args.plants else None,
        limit=args.limit, verbose=not args.quiet)
    print()
    print("Patch features projected onto their own principal components, fitted")
    print("on the subject rather than the whole frame. Read the boundaries the")
    print("features draw, not the colours they take: the hues are arbitrary.")
    return 0


def cmd_architecture(args: argparse.Namespace) -> int:
    """Render one architecture diagram per methodology."""
    from .eval.architecture import write_all

    paths = write_all(args.out_dir)
    print(f"Wrote {len(paths)} diagrams to {args.out_dir}\n")
    for path in paths:
        print(f"  {path.name:34s} {path.stat().st_size / 1024:6.1f} KB")
    print(
        "\nDrawn from config and from measured results, so they cannot drift "
        "out of\ndate the way a hand-made figure does. SVG: selectable text, "
        "sharp at any size."
    )
    return 0


def cmd_views(args: argparse.Namespace) -> int:
    from .eval.view_ablation import format_table, run_ablation

    results = run_ablation(args.work_dir, verbose=True)
    if not results:
        print("\nNo view-count caches found. Build them first (see above).")
        return 1

    print()
    print(format_table(results))
    print(
        "\n`plausible` counts specimens whose reconstructed above-ground volume "
        "could\nphysically weigh the measured mass (200-1000 kg/m3). Agreement "
        "degrades gently\nas views are removed; plausibility does not, which is "
        "the point of reporting both."
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps([r.as_dict() for r in results], indent=2), encoding="utf-8"
        )
        print(f"\nSaved to {args.out}")
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    from .data.preprocess import usable_plant_ids
    from .eval.baselines import evaluate_baselines, load_features
    from .eval.metrics import bootstrap_interval

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)
    features = load_features(plant_ids, args.cache_dir)
    targets = np.array([f.target_kg for f in features])

    print(f"Leave-one-out over {len(features)} specimens\n")
    for name, (metrics, predictions) in evaluate_baselines(features).items():
        low, high = bootstrap_interval(predictions, targets)
        print(f"  {name:20s} {metrics}")
        print(f"  {'':20s} RMSE 95% CI [{low:.3f}, {high:.3f}]")
    return 0


def cmd_pretrain(args: argparse.Namespace) -> int:
    from .data.preprocess import usable_plant_ids
    from .models.ggssvt import GGSSVT
    from .training.dataset import SpecimenDataset
    from .training.trainer import resolve_device, train_stage

    # labelled=None: stage 1 supervises occupancy against the carve and never
    # reads target_kg, so an unharvested specimen trains it just as well. Every
    # downstream regression keeps the labelled-only default.
    plant_ids = args.plants or usable_plant_ids(args.cache_dir, labelled=None)
    unlabelled = len(plant_ids) - len(usable_plant_ids(args.cache_dir))
    if unlabelled:
        print(f"Pretraining on {len(plant_ids)} specimens, "
              f"{unlabelled} of them unharvested")
    dataset = SpecimenDataset(plant_ids, cache_dir=args.cache_dir, mode="occupancy")

    model = GGSSVT(
        config=_model_config(args),
        tokens_per_view=args.tokens_per_view,
        geometry_grounded=not args.no_geometry,
    )
    device = resolve_device(args.device)
    print(
        f"Pretraining on {len(dataset)} specimens, {model.n_parameters() / 1e6:.2f}M "
        f"parameters, device {device}"
    )

    run = train_stage(
        model,
        dataset,
        stage="pretrain",
        epochs=args.epochs,
        config=_training_config(args),
        device=device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import torch

    from . import __version__

    torch.save(
        {
            "version": __version__,
            "state_dict": model.state_dict(),
            "model_config": MODEL.__dict__,
            "tokens_per_view": args.tokens_per_view,
            "geometry_grounded": not args.no_geometry,
            "plant_ids": plant_ids,
        },
        args.out,
    )
    (args.out.with_suffix(".log.json")).write_text(run.to_json(), encoding="utf-8")
    print(f"Saved checkpoint to {args.out}")
    return 0


def cmd_loocv(args: argparse.Namespace) -> int:
    import torch

    from .data.preprocess import usable_plant_ids
    from .eval.metrics import bootstrap_interval, regression_metrics
    from .training.trainer import loocv, resolve_device

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)

    state = None
    if args.checkpoint and args.checkpoint.exists():
        state = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
        print(f"Starting each fold from {args.checkpoint}")
    elif not args.strict:
        print(
            "No pretrained checkpoint given and --strict not set: every fold will "
            "start from random initialisation, which is not the intended protocol.",
            file=sys.stderr,
        )

    results = loocv(
        plant_ids,
        cache_dir=args.cache_dir,
        model_config=_model_config(args),
        train_config=_training_config(args),
        tokens_per_view=args.tokens_per_view,
        geometry_grounded=not args.no_geometry,
        strict=args.strict,
        pretrained_state=state,
        device=resolve_device(args.device),
    )

    predicted = np.array([r.predicted_kg for r in results])
    target = np.array([r.target_kg for r in results])
    metrics = regression_metrics(predicted, target)
    low, high = bootstrap_interval(predicted, target)

    print(f"\nGG-SSVT leave-one-out: {metrics}")
    print(f"  RMSE 95% CI [{low:.3f}, {high:.3f}]")
    print(
        f"  occupancy: IoU@0.5 {np.mean([r.occupancy_iou for r in results]):.3f}  "
        f"best-threshold IoU {np.mean([r.occupancy_best_iou for r in results]):.3f}  "
        f"AP {np.mean([r.occupancy_ap for r in results]):.3f}"
    )
    print(
        f"  protocol: {'strict (inductive)' if args.strict else 'transductive pretraining'}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "strict": args.strict,
                "geometry_grounded": not args.no_geometry,
                "metrics": metrics.as_dict(),
                "folds": [r.__dict__ for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved fold results to {args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .eval.report import write_report

    model_predictions = None
    if args.folds and args.folds.exists():
        payload = json.loads(args.folds.read_text(encoding="utf-8"))
        model_predictions = {
            fold["held_out"]: fold["predicted_kg"] for fold in payload["folds"]
        }
        print(f"Including GG-SSVT predictions from {args.folds}")

    path = write_report(args.plants, model_predictions=model_predictions)
    print(f"Wrote {path}")
    print(f"  and {path.parent / 'comparison.csv'}, {path.parent / 'scatter.svg'}")
    return 0


def cmd_visualise(args: argparse.Namespace) -> int:
    from .eval.visualise import write_overlays

    for plant_id in args.plants or ["E002", "M001"]:
        paths = write_overlays(plant_id, out_dir=args.out_dir)
        for path in paths:
            print(f"Wrote {path}")
    return 0


def _write_comparison(report, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.to_json(), encoding="utf-8")
    print(f"\nSaved comparison to {out}")


def cmd_dino_probe(args: argparse.Namespace) -> int:
    """Frozen-feature DINO vs no-DINO comparison. CPU, minutes."""
    from .data.preprocess import usable_plant_ids
    from .eval.experiment import probe_experiment

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)
    print(f"Frozen-feature probe over {len(plant_ids)} specimens, leave-one-out\n")

    report = probe_experiment(
        plant_ids,
        backbones=tuple(args.backbones),
        variant=args.variant,
        cache_dir=args.cache_dir,
        n_components=args.components,
        alpha=args.alpha,
    )
    print()
    print(report.to_table())
    _write_comparison(report, args.out)
    return 0


def cmd_experiment(args: argparse.Namespace) -> int:
    """Full GG-SSVT backbone comparison. Needs a GPU."""
    from .data.preprocess import usable_plant_ids
    from .eval.experiment import backbone_experiment
    from .training.trainer import resolve_device

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)
    device = resolve_device(args.device)

    if device.type != "cuda":
        print(
            "WARNING: no CUDA device. This trains one model per backbone and will "
            "take many hours on a CPU. Use `dino-probe` for a CPU-scale answer.",
            file=sys.stderr,
        )

    report = backbone_experiment(
        plant_ids,
        backbones=tuple(args.backbones),
        variant=args.variant,
        cache_dir=args.cache_dir,
        train_config=_training_config(args),
        epochs=args.epochs,
        tokens_per_view=args.tokens_per_view,
        out_dir=args.out_dir,
        device=device,
    )
    print()
    print(report.to_table())
    _write_comparison(report, args.out_dir / "backbone_comparison.json")
    return 0


def cmd_factorial(args: argparse.Namespace) -> int:
    """SAM3D on/off crossed with the appearance backbone."""
    from .eval.factorial import probe_factorial, train_factorial

    if args.train:
        from .training.trainer import resolve_device

        device = resolve_device(args.device)
        if device.type != "cuda":
            print(
                "WARNING: no CUDA device. --train runs one pretrain plus one "
                "leave-one-out sweep per cell and will take many hours on a CPU. "
                "Drop --train for the CPU-scale probe.",
                file=sys.stderr,
            )
        report = train_factorial(
            segmenters=tuple(args.segmenters),
            backbones=tuple(args.backbones),
            variant=args.variant,
            train_config=_training_config(args),
            epochs=args.epochs,
            tokens_per_view=args.tokens_per_view,
            device=device,
        )
    else:
        report = probe_factorial(
            segmenters=tuple(args.segmenters),
            backbones=tuple(args.backbones),
            variant=args.variant,
            n_components=args.components,
            alpha=args.alpha,
        )
    print()
    print(report.to_table())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report.to_json(), encoding="utf-8")
    print(f"\nSaved factorial to {args.out}")
    return 0


def cmd_gallery(args: argparse.Namespace) -> int:
    """Render every reconstruction: contact sheets, PLY clouds, interactive page."""
    import json

    from .data.preprocess import usable_plant_ids
    from .eval.factorial import CACHE_DIRS
    from .eval.gallery_html import build_html
    from .eval.render import build_gallery

    caches = {
        name: path
        for name, path in CACHE_DIRS.items()
        if (path / "quality.json").exists()
    }
    if not caches:
        print("No preprocessed cache found. Run `preprocess` first.", file=sys.stderr)
        return 2

    plant_ids = args.plants or sorted(
        {pid for path in caches.values() for pid in usable_plant_ids(path)}
    )
    print(f"Rendering {len(plant_ids)} specimens across {len(caches)} segmenter(s)")

    manifest = build_gallery(
        plant_ids,
        cache_dirs=caches,
        out_dir=args.out_dir,
        write_ply=not args.no_ply,
        write_sheets=not args.no_sheets,
        columns=args.columns,
        card_size=args.card_size,
        point_radius=args.point_radius,
        supersample=args.supersample,
        max_points=args.max_points,
        downsample=args.downsample,
    )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    page = build_html(manifest, args.out_dir / "reconstructions.html")
    print(f"  wrote {page}")
    if not args.no_ply:
        print(f"  wrote {len(manifest['files'])} files under {args.out_dir}")
    return 0


def cmd_nerfstudio(args: argparse.Namespace) -> int:
    """Export estimated rig poses as Nerfstudio transforms.json."""
    from .eval.nerfstudio_export import export_dataset, training_commands

    results = export_dataset(
        args.plants,
        in_place=not args.out_dir,
        out_root=args.out_dir or (WORK_DIR / "nerfstudio"),
        include_depth=not args.no_depth,
        write_rig_positions=not args.no_rig_positions,
    )
    if not results:
        print("No specimens exported.", file=sys.stderr)
        return 2

    first = sorted(results)[0]
    print(f"\nExported {len(results)} specimens.")
    print(
        "\nPoses are estimated from depth, not measured by ChArUco calibration.\n"
        "Train with a camera optimiser so the radiance field can refine them:\n"
    )
    for line in training_commands(first):
        print("  " + line)
    return 0


def cmd_mesh(args: argparse.Namespace) -> int:
    """Extract meshes and score biomass from them against every other method."""
    from .data.preprocess import load_cached, usable_plant_ids
    from .eval.baselines import load_features
    from .eval.mesh_baseline import evaluate_with_mesh
    from .eval.metrics import paired_bootstrap_difference
    from .geometry.mesh import mesh_from_occupancy

    plant_ids = args.plants or usable_plant_ids(args.cache_dir)
    print(f"Meshing {len(plant_ids)} specimens from {args.cache_dir}")

    results, table = evaluate_with_mesh(
        plant_ids, cache_dir=args.cache_dir, alpha=args.alpha
    )
    targets = np.array([f.target_kg for f in load_features(plant_ids, args.cache_dir)])

    header = f"\n{'method':28s} {'RMSE':>7s} {'MAE':>7s} {'MARE%':>7s} {'R2':>7s}"
    print(header)
    print("-" * (len(header) - 1))
    for name, (metrics, _) in sorted(results.items(), key=lambda kv: kv[1][0].rmse_kg):
        print(
            f"{name:28s} {metrics.rmse_kg:7.3f} {metrics.mae_kg:7.3f} "
            f"{metrics.mare * 100:7.1f} {metrics.r2:7.3f}"
        )

    reference = "geometric features"
    if reference in results:
        print(f"\nPaired bootstrap against '{reference}':")
        for name in ("mesh geometry", "canopy area allometric"):
            if name not in results:
                continue
            d = paired_bootstrap_difference(
                results[name][1], results[reference][1], targets
            )
            verdict = (
                "significant" if d["high"] < 0 or d["low"] > 0 else "not resolved"
            )
            print(
                f"  {name:26s} dRMSE {d['difference']:+.3f} kg  "
                f"95% CI [{d['low']:+.3f}, {d['high']:+.3f}]  {verdict}"
            )

    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        for plant_id in plant_ids:
            cached = load_cached(plant_id, args.cache_dir)
            mesh = mesh_from_occupancy(
                cached.occupancy,
                voxel_size_m=cached.voxel_size_m,
                smoothing=args.smoothing,
            )
            (args.export / f"{plant_id}.obj").write_text(mesh.to_obj(), encoding="utf-8")
        print(f"\nWrote {len(plant_ids)} OBJ meshes to {args.export}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "n_specimens": len(plant_ids),
                "mesh_metrics": table,
                "results": {
                    name: {**m.as_dict(), "predictions": p.tolist()}
                    for name, (m, p) in results.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved to {args.out}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Build the self-contained research walkthrough page."""
    from .eval.dashboard_data import build_payload
    from .eval.site_html import build_site

    payload = build_payload()
    teaser = WORK_DIR / "reports" / "gallery" / "contact_sheet_geometric.png"
    index = build_site(
        payload.to_json(), payload.summary, out_dir=args.out, teaser=teaser
    )
    size = index.stat().st_size / 1e6
    print(f"Wrote {index} ({size:.2f} MB, {len(payload.specimens)} specimens)")
    if not teaser.exists():
        print(f"  no teaser image at {teaser}; run `cli gallery` to produce it")
    print("A GitHub Pages layout: index.html plus static/. Open index.html directly,")
    print("or push the whole directory to a *.github.io repository.")
    return 0


def cmd_posefree(args: argparse.Namespace) -> int:
    """Compare DUSt3R / MASt3R / Fast3R against the estimated rig."""
    from .eval.pose_free_experiment import run_experiment
    from .geometry.pose_free import available_backends

    if args.check_only:
        print("Pose-free backend availability:\n")
        for name, (ok, reason) in available_backends().items():
            print(f"  {'OK     ' if ok else 'MISSING'}  {name}")
            if reason:
                for line in reason.splitlines():
                    print(f"      {line}")
        return 0

    report = run_experiment(
        args.plants,
        methods=tuple(args.methods),
        device=args.device,
        image_size=args.image_size,
        cache_dir=args.cache_dir,
        out_path=args.out,
    )
    print()
    print(report.to_table())
    if args.out:
        print(f"\nSaved to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ggssvt",
        description="Geometry-grounded self-supervised volumetric transformer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="audit the raw dataset")
    inspect.set_defaults(func=cmd_inspect)

    attention = sub.add_parser(
        "attention", help="what the fusion stack attends to (needs a checkpoint)"
    )
    _add_common(attention)
    attention.add_argument("--plant", help="specimen id; the first usable one by default")
    attention.add_argument(
        "--checkpoint", type=Path, default=WORK_DIR / "checkpoints" / "pretrain.pt"
    )
    attention.add_argument(
        "--out", type=Path,
        default=WORK_DIR / "reports" / "attention" / "per_view.png",
    )
    attention.set_defaults(func=cmd_attention)

    reciprocity = sub.add_parser(
        "reciprocity",
        help="refine the masks with the reconstruction, and re-carve",
    )
    reciprocity.add_argument(
        "--refine-from", default="fused", choices=("fused", "self"),
        help="which reconstruction supplies the second opinion",
    )
    reciprocity.add_argument("--quiet", action="store_true")
    reciprocity.set_defaults(func=cmd_reciprocity)

    robustness = sub.add_parser(
        "robustness", help="H4: depth noise and occlusion sweeps (CPU)"
    )
    robustness.add_argument("--seed", type=int, default=0)
    robustness.add_argument("--quiet", action="store_true")
    robustness.set_defaults(func=cmd_robustness)

    label_efficiency = sub.add_parser(
        "label-efficiency",
        help="H1: labels needed per representation (CPU, seconds)",
    )
    label_efficiency.add_argument(
        "--exclude", nargs="*", default=["V010"],
        help="specimens kept out; V010 fails the plausibility criterion",
    )
    label_efficiency.add_argument("--quiet", action="store_true")
    label_efficiency.set_defaults(func=cmd_label_efficiency)

    batch_holdout = sub.add_parser(
        "batch-holdout",
        help="leave-one-batch-out against leave-one-out (CPU, seconds)",
    )
    batch_holdout.add_argument(
        "--exclude", nargs="*", default=["V010"],
        help="specimens kept out; V010 fails the plausibility criterion",
    )
    batch_holdout.add_argument("--quiet", action="store_true")
    batch_holdout.set_defaults(func=cmd_batch_holdout)

    external = sub.add_parser(
        "external",
        help="external validation on the 4TU lettuce set (CPU, ~4 min first run)",
    )
    external.add_argument(
        "--root", default=None,
        help="where the download was unpacked; defaults to dataset_biomass/",
    )
    external.add_argument(
        "--force", action="store_true", help="re-measure instead of using the cache",
    )
    external.add_argument("--quiet", action="store_true")
    external.set_defaults(func=cmd_external)

    virtual = sub.add_parser(
        "virtual-views",
        help="carve and fusion against Pheno4D ground truth (CPU, ~10 min)",
    )
    virtual.add_argument("--root", default=None, help="where Pheno4D was unpacked")
    virtual.add_argument("--subsample", type=int, default=2,
                         help="keep one point in this many when rendering")
    virtual.add_argument("--limit", type=int, default=None,
                         help="stop after this many plants")
    virtual.add_argument("--quiet", action="store_true")
    virtual.set_defaults(func=cmd_virtual_views)

    plots = sub.add_parser(
        "plots", help="draw the figures for the newer experiments (CPU, ~1 min)",
    )
    plots.add_argument("--quiet", action="store_true")
    plots.set_defaults(func=cmd_plots)

    resolution = sub.add_parser(
        "resolution",
        help="what each comparison could detect, and which settled (CPU, seconds)",
    )
    resolution.add_argument("--quiet", action="store_true")
    resolution.set_defaults(func=cmd_resolution)

    pedestal = sub.add_parser(
        "pedestal",
        help="plant the carve discarded, per specimen (CPU, ~1 min)",
    )
    pedestal.add_argument("--quiet", action="store_true")
    pedestal.set_defaults(func=cmd_pedestal)

    pot_mass = sub.add_parser(
        "pot-mass", help="check pot masses against the reconstruction (CPU, ~1 min)")
    pot_mass.add_argument("--quiet", action="store_true")
    pot_mass.set_defaults(func=cmd_pot_mass)

    stages = sub.add_parser(
        "stages", help="per-stage clouds for the project page (CPU, ~3 min)")
    stages.add_argument("--limit", type=int, default=None,
                        help="stop after this many specimens")
    stages.add_argument("--quiet", action="store_true")
    stages.set_defaults(func=cmd_stages)

    filmstrip = sub.add_parser(
        "filmstrip",
        help="raw frames, masks, depth and volumes per specimen (CPU, ~30 s)")
    filmstrip.add_argument("--limit", type=int, default=None,
                           help="stop after this many specimens")
    filmstrip.add_argument("--views", type=int, default=6,
                           help="how many of the twelve views to show")
    filmstrip.add_argument("--quiet", action="store_true")
    filmstrip.set_defaults(func=cmd_filmstrip)

    surface_mesh = sub.add_parser(
        "surface-mesh",
        help="surface-voxel volume against the carve (CPU, ~2 min)")
    surface_mesh.add_argument("--quiet", action="store_true")
    surface_mesh.set_defaults(func=cmd_surface_mesh)

    viewpoint = sub.add_parser(
        "viewpoint", help="H2: held-out-view consistency (CPU, ~25 min)"
    )
    viewpoint.add_argument("--quiet", action="store_true")
    viewpoint.set_defaults(func=cmd_viewpoint)

    frequency = sub.add_parser(
        "frequency",
        help="H3: radial power spectrum of the occupancy target (CPU)",
    )
    _add_common(frequency)
    frequency.add_argument("--plants", nargs="*")
    frequency.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "frequency.json"
    )
    frequency.set_defaults(func=cmd_frequency)

    funnel = sub.add_parser(
        "funnel", help="the staged screening figure, from the report JSON"
    )
    funnel.add_argument(
        "--out", type=Path,
        default=WORK_DIR / "reports" / "figures" / "screening_funnel.png",
    )
    funnel.set_defaults(func=cmd_funnel)

    neural_field = sub.add_parser(
        "neural-field",
        help="a trained Nerfstudio field as a third operator, threshold swept",
    )
    _add_common(neural_field)
    neural_field.add_argument(
        "--outputs", type=Path, default=WORK_DIR / "nerfstudio" / "outputs",
        help="directory of Nerfstudio training outputs, one per plant id",
    )
    neural_field.add_argument("--device", default="cuda")
    neural_field.add_argument("--quiet", action="store_true")
    neural_field.set_defaults(func=cmd_neural_field)

    show_parser = sub.add_parser(
        "show",
        help="look at any layer of any specimen; every knob is a flag",
    )
    show_parser.add_argument("--plants", nargs="*",
                             help="specimen ids; the first usable one by default")
    show_parser.add_argument(
        "--layers", nargs="+", default=["occupancy"],
        help="rgb depth segmentation occupancy points mesh, in draw order",
    )
    show_parser.add_argument(
        "--views", nargs="+", default=["front", "side", "top"],
        help="axes for the occupancy and points layers",
    )
    show_parser.add_argument("--view", type=int, default=0,
                             help="which captured view the per-view layers use")
    show_parser.add_argument("--size", type=int, default=300,
                             help="pixels per panel; 400+ for a print figure")
    show_parser.add_argument("--point-radius", type=int, default=1)
    show_parser.add_argument("--supersample", type=int, default=3,
                             help="render at this multiple and average down")
    show_parser.add_argument("--cmap", default="viridis",
                             choices=["viridis", "greys", "greys_r"])
    show_parser.add_argument(
        "--source", default="carve",
        choices=["carve", "fused", "sam3d", "neural"],
        help="neural needs --nerf-output and a threshold",
    )
    show_parser.add_argument(
        "--nerf-output", type=Path,
        help="a directory cli neural-field sampled, holding density_grid.npz",
    )
    show_parser.add_argument(
        "--threshold", type=float, default=1.0,
        help="density cut for source=neural; stated in the caption",
    )
    show_parser.add_argument(
        "--sweep-thresholds", type=float, nargs="+",
        default=[0.1, 1.0, 10.0, 100.0],
        help="cuts drawn side by side by the threshold_sweep layer",
    )
    show_parser.add_argument("--max-points", type=int, default=20000)
    show_parser.add_argument("--no-label", action="store_true")
    show_parser.add_argument(
        "--backend", default="pil", choices=["pil", "text", "matplotlib"],
        help="text prints into the terminal, which is what you want over SSH; "
             "matplotlib opens an interactive 3D scatter and needs matplotlib",
    )
    show_parser.add_argument(
        "--out-dir", type=Path, default=WORK_DIR / "reports" / "figures"
    )
    show_parser.set_defaults(func=cmd_show)

    preflight = sub.add_parser(
        "preflight",
        help="check the environment before a long run",
    )
    preflight.add_argument("--device", default="cuda")
    preflight.add_argument("--quiet", action="store_true",
                           help="only show problems")
    preflight.set_defaults(func=cmd_preflight)

    access = sub.add_parser(
        "access", help="which HuggingFace account is active, and what it can download"
    )
    access.set_defaults(func=cmd_access)

    preprocess = sub.add_parser(
        "preprocess", help="register, segment and carve every specimen"
    )
    _add_common(preprocess)
    preprocess.add_argument("--plants", nargs="*", help="specific plant ids")
    preprocess.add_argument("--seed", type=int, default=0)
    preprocess.add_argument(
        "--include-unlabelled",
        action="store_true",
        help=(
            "also carve specimens with no row in ground_truth.csv. Stage-1 "
            "pretraining fits occupancy and never reads the mass, so an "
            "unharvested plant is a valid training example there"
        ),
    )
    preprocess.add_argument(
        "--segmenter",
        default="geometric",
        choices=["geometric", "sam3d"],
        help="subject segmentation; sam3d needs its own --cache-dir",
    )
    preprocess.add_argument(
        "--sam-model", default="base", help="SAM checkpoint: base | large | huge"
    )
    preprocess.add_argument("--sam-device", default="cpu")
    preprocess.add_argument(
        "--views", type=int, default=None,
        help=(
            "use an evenly spaced subset of the 12 views (must divide 12: "
            "2, 3, 4, 6). Needs its own --cache-dir"
        ),
    )
    preprocess.set_defaults(func=cmd_preprocess)

    baselines = sub.add_parser("baselines", help="LOOCV baselines from the cache")
    _add_common(baselines)
    baselines.add_argument("--plants", nargs="*")
    baselines.set_defaults(func=cmd_baselines)

    fuse = sub.add_parser(
        "fuse", help="TSDF depth fusion, an alternative to space carving"
    )
    _add_common(fuse)
    fuse.add_argument("--plants", nargs="*")
    fuse.add_argument(
        "--voxel", type=float, default=FUSION_VOXEL_M,
        help="fusion voxel size in metres; 6 mm is two depth samples across",
    )
    fuse.add_argument("--resolution", type=int, default=FUSION_RESOLUTION)
    fuse.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "fusion.json"
    )
    fuse.add_argument(
        "--write-cache",
        type=Path,
        nargs="?",
        const=WORK_DIR / "cache_tsdf",
        default=None,
        help=(
            "also write a cache of fused occupancy at the carve's own "
            "resolution, so the viewer and every other tool can read it"
        ),
    )
    fuse.set_defaults(func=cmd_fuse)

    quality = sub.add_parser(
        "quality", help="reconstruction metrics: re-projection and agreement"
    )
    _add_common(quality)
    quality.add_argument("--plants", nargs="*")
    quality.add_argument(
        "--out", type=Path,
        default=WORK_DIR / "reports" / "reconstruction_quality.json",
    )
    quality.set_defaults(func=cmd_quality)

    gate = sub.add_parser(
        "gate", help="acceptance checks; exits non-zero if anything is blocked"
    )
    _add_common(gate)
    gate.add_argument("--plants", nargs="*")
    gate.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "gates.json"
    )
    gate.set_defaults(func=cmd_gate)

    dino_seg = sub.add_parser(
        "dino-segment",
        help="lift DINO features onto the points and cluster (DITR-style)",
    )
    _add_common(dino_seg)
    dino_seg.add_argument("--plants", nargs="*")
    dino_seg.add_argument("--variant", default="base", choices=["small", "base", "large"])
    dino_seg.add_argument(
        "--backbone", default="dinov2", choices=["dinov2", "dinov3"],
        help="which frozen features to lift; dinov3 needs granted access")
    dino_seg.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "dino_segment.json"
    )
    dino_seg.set_defaults(func=cmd_dino_segment)

    recon_metrics = sub.add_parser(
        "recon-metrics",
        help="accuracy, completeness and F-score on Pheno4D (needs virtual-views)")
    recon_metrics.add_argument("--quiet", action="store_true")
    recon_metrics.set_defaults(func=cmd_recon_metrics)

    backbone_viz = sub.add_parser(
        "backbone-viz",
        help="what DINOv2 and DINOv3 each see, as feature maps and clusterings")
    backbone_viz.add_argument(
        "--plants", nargs="*", default=None,
        help="default is every usable specimen")
    backbone_viz.add_argument("--limit", type=int, default=None)
    backbone_viz.add_argument("--quiet", action="store_true")
    backbone_viz.set_defaults(func=cmd_backbone_viz)

    architecture = sub.add_parser(
        "architecture", help="architecture diagram per methodology, as SVG"
    )
    architecture.add_argument(
        "--out-dir",
        type=Path,
        default=WORK_DIR / "reports" / "architecture",
    )
    architecture.set_defaults(func=cmd_architecture)

    views = sub.add_parser(
        "views", help="view-count ablation across the per-view caches"
    )
    views.add_argument("--work-dir", type=Path, default=WORK_DIR)
    views.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "view_ablation.json"
    )
    views.set_defaults(func=cmd_views)

    pretrain = sub.add_parser("pretrain", help="stage 1, self-supervised")
    _add_common(pretrain)
    pretrain.add_argument("--plants", nargs="*")
    pretrain.add_argument("--epochs", type=int, default=TRAIN.pretrain_epochs)
    pretrain.add_argument("--device", default=TRAIN.device)
    pretrain.add_argument("--tokens-per-view", type=int, default=64)
    pretrain.add_argument(
        "--no-geometry",
        action="store_true",
        help="ablation: disable Fourier grounding and the distance bias",
    )
    pretrain.add_argument(
        "--out", type=Path, default=WORK_DIR / "checkpoints" / "pretrain.pt"
    )
    _add_training(pretrain)
    pretrain.set_defaults(func=cmd_pretrain)

    evaluate = sub.add_parser("loocv", help="stage 2 and leave-one-out evaluation")
    _add_common(evaluate)
    evaluate.add_argument("--plants", nargs="*")
    evaluate.add_argument("--device", default=TRAIN.device)
    evaluate.add_argument("--tokens-per-view", type=int, default=64)
    evaluate.add_argument("--no-geometry", action="store_true")
    evaluate.add_argument(
        "--checkpoint", type=Path, default=WORK_DIR / "checkpoints" / "pretrain.pt"
    )
    evaluate.add_argument(
        "--strict",
        action="store_true",
        help="re-run pretraining inside each fold for an inductive result",
    )
    evaluate.add_argument("--out", type=Path, default=WORK_DIR / "reports" / "folds.json")
    _add_training(evaluate)
    evaluate.set_defaults(func=cmd_loocv)

    report = sub.add_parser("report", help="write result tables and figures")
    report.add_argument("--plants", nargs="*")
    report.add_argument(
        "--folds",
        type=Path,
        default=WORK_DIR / "reports" / "folds.json",
        help="GG-SSVT fold results to include in the comparison",
    )
    report.set_defaults(func=cmd_report)

    visualise = sub.add_parser("visualise", help="rig and segmentation overlays")
    visualise.add_argument("--plants", nargs="*")
    visualise.add_argument(
        "--out-dir", type=Path, default=WORK_DIR / "reports" / "overlays"
    )
    visualise.set_defaults(func=cmd_visualise)

    probe = sub.add_parser(
        "dino-probe",
        help="frozen-feature DINO vs no-DINO comparison (CPU, minutes)",
    )
    _add_common(probe)
    probe.add_argument("--plants", nargs="*")
    probe.add_argument(
        "--backbones",
        nargs="+",
        default=["dinov2", "dinov3"],
        choices=["dinov2", "dinov3"],
        help="DINO backbones to probe; the no-DINO control always runs",
    )
    probe.add_argument("--variant", default="base", choices=["small", "base", "large"])
    probe.add_argument("--components", type=int, default=8, help="PCA components")
    probe.add_argument("--alpha", type=float, default=1.0, help="ridge strength")
    probe.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "dino_probe.json"
    )
    probe.set_defaults(func=cmd_dino_probe)

    experiment = sub.add_parser(
        "experiment", help="full GG-SSVT backbone comparison (needs a GPU)"
    )
    _add_common(experiment)
    experiment.add_argument("--plants", nargs="*")
    experiment.add_argument(
        "--backbones",
        nargs="+",
        default=["cnn", "dinov2", "dinov3"],
        choices=["cnn", "dinov2", "dinov3"],
    )
    experiment.add_argument(
        "--variant", default="base", choices=["small", "base", "large"]
    )
    experiment.add_argument("--epochs", type=int, default=TRAIN.pretrain_epochs)
    experiment.add_argument("--device", default=TRAIN.device)
    experiment.add_argument("--tokens-per-view", type=int, default=64)
    experiment.add_argument("--out-dir", type=Path, default=WORK_DIR / "experiments")
    _add_training(experiment)
    experiment.set_defaults(func=cmd_experiment)

    factorial = sub.add_parser(
        "factorial",
        help="full SAM3D x DINO factorial via the frozen-feature probe (CPU)",
    )
    factorial.add_argument(
        "--segmenters", nargs="+", default=["geometric", "sam3d"],
        choices=["geometric", "sam3d"],
    )
    factorial.add_argument(
        "--backbones", nargs="+", default=["cnn", "dinov2", "dinov3"],
        choices=["cnn", "dinov2", "dinov3"],
    )
    factorial.add_argument(
        "--variant", default="base", choices=["small", "base", "large"]
    )
    factorial.add_argument("--components", type=int, default=8)
    factorial.add_argument("--alpha", type=float, default=1.0)
    factorial.add_argument(
        "--train",
        action="store_true",
        help="train GG-SSVT per cell instead of probing frozen features (GPU)",
    )
    factorial.add_argument("--epochs", type=int, default=TRAIN.pretrain_epochs)
    factorial.add_argument("--device", default=TRAIN.device)
    factorial.add_argument("--tokens-per-view", type=int, default=64)
    factorial.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "factorial.json"
    )
    _add_training(factorial)
    factorial.set_defaults(func=cmd_factorial)

    gallery = sub.add_parser(
        "gallery", help="render every reconstruction for viewing"
    )
    gallery.add_argument("--plants", nargs="*")
    gallery.add_argument(
        "--out-dir", type=Path, default=WORK_DIR / "reports" / "gallery"
    )
    gallery.add_argument("--columns", type=int, default=4)
    gallery.add_argument(
        "--card-size", type=int, default=200,
        help="pixels per view panel; raise for print figures",
    )
    gallery.add_argument(
        "--point-radius", type=int, default=1,
        help="half-width of the square drawn per voxel, in output pixels",
    )
    gallery.add_argument(
        "--supersample", type=int, default=3,
        help="render at this multiple and average down; 1 disables antialiasing",
    )
    gallery.add_argument(
        "--max-points", type=int, default=20000,
        help="cap on points sent to the interactive viewer",
    )
    gallery.add_argument(
        "--downsample", type=int, default=2,
        help="voxel stride for the interactive viewer; 1 sends every voxel",
    )
    gallery.add_argument("--no-ply", action="store_true", help="skip PLY export")
    gallery.add_argument(
        "--no-sheets", action="store_true", help="skip the PNG contact sheets"
    )
    gallery.set_defaults(func=cmd_gallery)

    mesh = sub.add_parser(
        "mesh", help="extract meshes and score biomass from them"
    )
    _add_common(mesh)
    mesh.add_argument("--plants", nargs="*")
    mesh.add_argument("--alpha", type=float, default=1.0, help="ridge strength")
    mesh.add_argument(
        "--smoothing", type=int, default=0,
        help="Laplacian smoothing passes on exported meshes (changes area/volume)",
    )
    mesh.add_argument(
        "--export", type=Path, default=None, help="also write OBJ meshes here"
    )
    mesh.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "mesh.json"
    )
    mesh.set_defaults(func=cmd_mesh)

    dashboard = sub.add_parser(
        "dashboard", help="build the research walkthrough page"
    )
    dashboard.add_argument(
        "--out",
        type=Path,
        default=WORK_DIR / "site",
        help="site directory to write (index.html plus static/)",
    )
    dashboard.set_defaults(func=cmd_dashboard)

    posefree = sub.add_parser(
        "posefree",
        help="DUSt3R / MASt3R / Fast3R poses vs the estimated rig (needs a GPU)",
    )
    _add_common(posefree)
    posefree.add_argument("--plants", nargs="*")
    posefree.add_argument(
        "--methods", nargs="+", default=["fast3r", "dust3r", "mast3r"],
        choices=["dust3r", "mast3r", "fast3r"],
    )
    posefree.add_argument("--device", default="cuda")
    posefree.add_argument("--image-size", type=int, default=512)
    posefree.add_argument(
        "--check-only", action="store_true",
        help="report which backends are installed and stop",
    )
    posefree.add_argument(
        "--out", type=Path, default=WORK_DIR / "reports" / "posefree.json"
    )
    posefree.set_defaults(func=cmd_posefree)

    nerf = sub.add_parser(
        "nerfstudio", help="export estimated poses as Nerfstudio transforms.json"
    )
    nerf.add_argument("--plants", nargs="*")
    nerf.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="write here instead of into each specimen directory",
    )
    nerf.add_argument("--no-depth", action="store_true", help="omit depth_file_path")
    nerf.add_argument(
        "--no-rig-positions",
        action="store_true",
        help="skip the rig_positions.json / intrinsics files make_transforms.py uses",
    )
    nerf.set_defaults(func=cmd_nerfstudio)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
