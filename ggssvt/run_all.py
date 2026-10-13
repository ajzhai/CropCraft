"""The whole programme, unattended, in one command.

The runbook is a list of commands to type. That is fine when you are sitting in
front of it and can react to each one, and it is the wrong shape for a machine
you are leaving overnight: a step that needs an optional dependency, or a model
whose access request has not been granted, will stop a sequence joined with
``&&`` and waste the remaining hours on nothing.

So this runs the same steps in the same order, and classifies each one:

**required**   its output is what the following steps read. A failure here stops
               the run, because everything after it would either crash or, worse,
               silently score a stale cache.
**optional**   a real result, but nothing downstream depends on it. The mesh arm
               needs scikit-image; the DINOv3 cells need an access grant that
               Meta approves by hand. Neither is a reason to abandon the night.

Every step is timed, its output is teed to its own log, and a table at the end
says what ran, what did not, and why. Re-running skips whatever already has a
fresh artefact, so an interrupted night resumes rather than restarts.

    python -m ggssvt.run_all --device cuda

    python -m ggssvt.run_all --device cuda --plan core --skip posefree

The GPU steps dominate: the training campaign is eight to ten hours and the
pose-free check about two, against roughly one hour for everything else.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import WORK_DIR


@dataclass(frozen=True)
class Step:
    """One runbook step, and how to tell whether it needs to run."""

    key: str
    label: str
    argv: list[str]
    artefact: Path | None = None
    required: bool = True
    needs_gpu: bool = False
    note: str = ""


@dataclass
class Outcome:
    key: str
    label: str
    status: str                 # done | skipped | failed | blocked-by
    seconds: float = 0.0
    detail: str = ""
    log: Path | None = None

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "seconds": round(self.seconds, 1),
            "detail": self.detail,
            "log": str(self.log) if self.log else None,
        }


def programme(work_dir: Path, *, device: str, plan: str, batch_size: int,
              workers: int) -> tuple[Step, ...]:
    """The runbook, in order, as a list this module can execute."""
    reports = work_dir / "reports"
    cli = [sys.executable, "-m", "ggssvt.cli"]

    return (
        Step("preflight", "Pre-flight checks",
             cli + ["preflight", "--device", device], None, True,
             note="Seconds, and it is the step that would have caught every "
                  "environment failure this project has hit: a CPU torch "
                  "wheel, the wrong HuggingFace account, a missing optional "
                  "dependency, malformed proxy variables, a damaged ground "
                  "truth. Required, because each of those makes the rest of "
                  "the run either wrong or wasted."),

        Step("inspect", "Audit the dataset", cli + ["inspect"], None, True,
             note="Cheap, and it catches a damaged ground truth before an "
                  "hour of preprocessing does."),

        Step("preprocess", "Preprocess, geometric carve",
             cli + ["preprocess"], work_dir / "cache" / "quality.json"),

        Step("preprocess_sam3d", "Preprocess, SAM3D masks",
             cli + ["preprocess", "--segmenter", "sam3d",
                    "--cache-dir", str(work_dir / "cache_sam3d"),
                    "--sam-device", device],
             work_dir / "cache_sam3d" / "quality.json",
             required=False, needs_gpu=True,
             note="Only the factorial reads this cache."),

        *[
            Step(f"views_{n}", f"View cache, {n} views",
                 cli + ["preprocess", "--views", str(n),
                        "--cache-dir", str(work_dir / f"cache_v{n}")],
                 work_dir / f"cache_v{n}" / "quality.json",
                 required=False,
                 note="Only the view ablation reads this cache.")
            for n in (3, 4, 6)
        ],

        Step("gate", "Acceptance gates", cli + ["gate"], reports / "gates.json",
             note="Exits non-zero when a blocking check fails, which is exactly "
                  "what should stop an unattended run."),

        Step("fuse", "TSDF depth fusion", cli + ["fuse", "--write-cache"],
             reports / "fusion.json",
             note="Writes the fused cache that baselines and the campaign read."),

        Step("quality", "Reconstruction metrics", cli + ["quality"],
             reports / "reconstruction_quality.json", required=False),

        Step("baselines", "Classical baselines", cli + ["baselines"],
             reports / "metrics.json"),

        Step("mesh", "Mesh arm", cli + ["mesh"], reports / "mesh.json",
             required=False,
             note="Needs scikit-image. Nothing else does."),

        Step("views", "View-count ablation", cli + ["views"],
             reports / "view_ablation.json", required=False),

        Step("dino_probe", "DINO frozen-feature probe", cli + ["dino-probe"],
             reports / "dino_probe.json", required=False, needs_gpu=True,
             note="DINOv3 cells skip themselves when the account lacks access."),

        Step("factorial", "Segmenter by backbone factorial", cli + ["factorial"],
             reports / "factorial.json", required=False, needs_gpu=True),

        Step("dino_segment", "DITR-style DINO lifting", cli + ["dino-segment"],
             reports / "dino_segment.json", required=False),

        Step("frequency", "Spectral analysis", cli + ["frequency"],
             reports / "frequency.json", required=False,
             note="H3's premise, measured. Run before the campaign's band "
                  "ladder so the ladder has a reason."),

        Step("label_efficiency", "Label-efficiency curves",
             cli + ["label-efficiency"],
             reports / "label_efficiency.json", required=False,
             note="H1's second half. Seconds, on frozen features."),

        Step("viewpoint", "Held-out-view consistency", cli + ["viewpoint"],
             reports / "viewpoint.json", required=False,
             note="H2's viewpoint claim. About 25 minutes on CPU."),

        Step("robustness", "Noise and occlusion sweeps", cli + ["robustness"],
             reports / "robustness.json", required=False,
             note="H4's other two thirds. About 20 minutes on CPU."),

        Step("reciprocity", "Reconstruction refining the masks",
             cli + ["reciprocity"],
             reports / "reciprocity.json", required=False,
             note="About 12 minutes on CPU."),

        Step("gallery", "Reconstruction gallery", cli + ["gallery"],
             work_dir / "reports" / "gallery" / "reconstructions.html",
             required=False),

        Step("architecture", "Architecture diagrams", cli + ["architecture"],
             reports / "architecture", required=False),

        Step("funnel", "Screening funnel figure", cli + ["funnel"],
             reports / "figures" / "screening_funnel.png", required=False,
             note="Reads its counts from the other reports, so it goes after "
                  "them."),

        Step("smoke", "Campaign smoke test",
             [sys.executable, "-m", "ggssvt.campaign",
              "--plan", "smoke", "--device", device],
             work_dir / "campaign_smoke" / "summary.txt",
             required=False, needs_gpu=True,
             note="Four specimens, two epochs. Proves the loop before the night "
                  "is committed to it. It is EXPECTED to report `blocked`: two "
                  "epochs cannot train an occupancy field, so the mass integral "
                  "is meaningless and the gate says so."),

        Step("campaign", f"Training campaign ({plan})",
             [sys.executable, "-m", "ggssvt.campaign",
              "--plan", plan, "--device", device,
              "--workers", str(workers), "--batch-size", str(batch_size)],
             work_dir / "campaign" / "summary.txt",
             required=False, needs_gpu=True,
             note="Eight to ten hours. Resumable: the same command continues."),

        Step("posefree", "Pose-free reconstruction",
             cli + ["posefree", "--methods", "fast3r", "--device", device],
             reports / "posefree.json", required=False, needs_gpu=True,
             note="Needs the cloned repos; see POSEFREE.md."),

        Step("report", "Result tables and figures", cli + ["report"],
             None, required=False),

        Step("dashboard", "Project page", cli + ["dashboard"],
             work_dir / "site" / "index.html", required=False),
    )


def _records_a_gated_skip(artefact: Path) -> bool:
    """Did this artefact finish, but with a condition skipped for lack of access?

    The DINOv3 cells write themselves as ``{"skipped": ...}`` rather than
    failing, which is right: an ungranted access request should not take down a
    probe whose other conditions are fine. The consequence is that the artefact
    looks complete afterwards, so a resumable runner would skip the step forever
    and the grant, when it finally lands, would never be picked up.

    So an artefact carrying a skip is only treated as current while the skip is
    still true. Re-checking costs one API call, against the alternative of
    quietly never running the condition the account was approved for.
    """
    try:
        data = json.loads(artefact.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False

    def skipped(node) -> bool:
        if isinstance(node, dict):
            if set(node) == {"skipped"}:
                return True
            return any(skipped(v) for v in node.values())
        return False

    if not skipped(data):
        return False

    try:
        from .models.backbones import backbone_is_available

        available, _ = backbone_is_available("dinov3", "base")
    except Exception:            # noqa: BLE001 - offline is not a reason to re-run
        return False
    return available


def _fresh(step: Step, newer_than: float) -> bool:
    """Has this step already run against the current inputs?"""
    if step.artefact is None or not step.artefact.exists():
        return False
    if step.artefact.stat().st_mtime < newer_than:
        return False
    return not (step.artefact.suffix == ".json"
                and _records_a_gated_skip(step.artefact))


def run(
    *,
    work_dir: Path = WORK_DIR,
    device: str = "cuda",
    plan: str = "core",
    batch_size: int = 2,
    workers: int = 8,
    skip: frozenset[str] = frozenset(),
    only: frozenset[str] = frozenset(),
    force: bool = False,
) -> int:
    # Local time, because this directory is read by whoever comes back to the
    # machine in the morning; astimezone() makes it tz-aware without moving it.
    stamp_name = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_dir = work_dir / "logs" / stamp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Anything older than the ground truth was fitted to superseded labels.
    from .config import DATASET_DIR
    stamp = (DATASET_DIR / "ground_truth.csv")
    baseline = stamp.stat().st_mtime if stamp.exists() else 0.0

    steps = programme(work_dir, device=device, plan=plan,
                      batch_size=batch_size, workers=workers)
    if only:
        steps = tuple(s for s in steps if s.key in only)

    started = time.time()
    outcomes: list[Outcome] = []
    stopped_at: str | None = None

    print(f"Logs: {log_dir}")
    print(f"{len(steps)} steps. Required failures stop the run; optional ones "
          f"are recorded and skipped past.\n")

    for index, step in enumerate(steps, start=1):
        head = f"[{index:2d}/{len(steps)}] {step.label}"

        if stopped_at:
            outcomes.append(Outcome(step.key, step.label, "blocked-by",
                                    detail=f"did not run: {stopped_at} failed"))
            print(f"{head}\n         skipped, {stopped_at} failed")
            continue

        if step.key in skip:
            outcomes.append(Outcome(step.key, step.label, "skipped",
                                    detail="--skip"))
            print(f"{head}\n         skipped by request")
            continue

        if not force and _fresh(step, baseline):
            outcomes.append(Outcome(step.key, step.label, "skipped",
                                    detail="artefact already current"))
            print(f"{head}\n         already current, use --force to redo")
            continue

        log = log_dir / f"{index:02d}_{step.key}.log"
        print(f"{head}\n         {' '.join(step.argv[2:])}\n         -> {log.name}",
              flush=True)

        began = time.time()
        with log.open("w", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(step.argv)}\n\n")
            handle.flush()
            code = subprocess.call(step.argv, stdout=handle,
                                   stderr=subprocess.STDOUT)
        took = time.time() - began

        if code == 0:
            outcomes.append(Outcome(step.key, step.label, "done", took, log=log))
            print(f"         done in {took / 60:.1f} min\n", flush=True)
            continue

        tail = _tail(log)
        outcomes.append(Outcome(step.key, step.label, "failed", took,
                                detail=tail, log=log))
        if step.required:
            stopped_at = step.key
            print(f"         FAILED (exit {code}) after {took / 60:.1f} min")
            print(f"         REQUIRED, so the rest is abandoned. {tail}\n",
                  flush=True)
        else:
            print(f"         failed (exit {code}) after {took / 60:.1f} min, "
                  f"optional so continuing")
            print(f"         {tail}\n", flush=True)

    _summarise(outcomes, log_dir, time.time() - started)
    return 1 if any(o.status == "failed" for o in outcomes) else 0


def _tail(log: Path, lines: int = 3) -> str:
    """The last few meaningful lines, for the summary table."""
    try:
        text = [ln.strip() for ln in log.read_text(encoding="utf-8",
                                                   errors="replace").splitlines()
                if ln.strip()]
    except OSError:
        return ""
    return " | ".join(text[-lines:])[:300]


def _summarise(outcomes: list[Outcome], log_dir: Path, elapsed: float) -> None:
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1

    width = max(len(o.label) for o in outcomes)
    print("=" * (width + 34))
    print(f"{'step':{width}}  {'status':10}  {'minutes':>8}")
    print("=" * (width + 34))
    for o in outcomes:
        mins = f"{o.seconds / 60:.1f}" if o.seconds else ""
        print(f"{o.label:{width}}  {o.status:10}  {mins:>8}")
    print("=" * (width + 34))
    print(f"{elapsed / 3600:.1f} hours total.  "
          + ",  ".join(f"{n} {k}" for k, n in sorted(counts.items())))

    failed = [o for o in outcomes if o.status == "failed"]
    if failed:
        print("\nFailures, most recent output:")
        for o in failed:
            print(f"  {o.label}\n    {o.detail}\n    full log: {o.log}")

    path = log_dir / "summary.json"
    path.write_text(json.dumps({
        "finished": datetime.now(tz=timezone.utc).isoformat(),
        "elapsed_hours": round(elapsed / 3600, 2),
        "steps": [o.as_dict() for o in outcomes],
    }, indent=2), encoding="utf-8")
    print(f"\nSummary: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ggssvt.run_all", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan", default="core",
                        help="campaign plan (default: core)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="step keys to skip, e.g. --skip posefree mesh")
    parser.add_argument("--only", nargs="*", default=[],
                        help="run only these step keys")
    parser.add_argument("--force", action="store_true",
                        help="re-run steps whose artefact is already current")
    parser.add_argument("--list", action="store_true",
                        help="print the steps and exit")
    args = parser.parse_args(argv)

    if args.list:
        for step in programme(WORK_DIR, device=args.device, plan=args.plan,
                              batch_size=args.batch_size, workers=args.workers):
            kind = "required" if step.required else "optional"
            gpu = "GPU" if step.needs_gpu else "   "
            print(f"  {step.key:18s} {kind:8s} {gpu}  {step.label}")
        return 0

    if shutil.which("nvidia-smi") is None and args.device == "cuda":
        print("warning: nvidia-smi not found; --device cuda may fail",
              file=sys.stderr)

    return run(device=args.device, plan=args.plan, batch_size=args.batch_size,
               workers=args.workers, skip=frozenset(args.skip),
               only=frozenset(args.only), force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
