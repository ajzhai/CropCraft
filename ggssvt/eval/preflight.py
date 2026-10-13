"""Everything worth knowing before committing hours to a run.

Every hurdle this project has hit on a fresh machine was knowable in under a
minute, and was instead discovered somewhere between forty minutes and eight
hours in: an optional dependency missing, so the mesh arm died after the caches
were built; a HuggingFace session logged into the wrong account, so every DINOv3
cell skipped itself; proxy variables holding a malformed URL, so nothing could
reach the network at all; conflict markers in the ground truth, which took down
three CLI entry points with an AttributeError that named neither the file nor
the line.

None of those are interesting failures. They are all *pre-flight* failures, and
the fix is to look before taking off rather than to write more prose in the
runbook telling somebody to remember.

Checks are one of three kinds:

**fatal**    the run cannot produce a correct result. CUDA missing when the
             device is cuda, a damaged ground truth, no dataset.
**degraded** the run works but something in it will silently do less than you
             think. A missing optional dependency, an ungranted model.
**note**     worth knowing, decides nothing.

Nothing here is expensive: no model is loaded, no specimen is processed. The
whole pass is a few seconds, dominated by the network probes, and it is designed
to be safe to run on a machine with no network at all.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..config import DATASET_DIR, WORK_DIR

FATAL = "fatal"
DEGRADED = "degraded"
NOTE = "note"


@dataclass(frozen=True)
class Finding:
    """One pre-flight check and what it found."""

    name: str
    level: str                  # fatal | degraded | note | ok
    detail: str
    fix: str = ""

    @property
    def ok(self) -> bool:
        return self.level == "ok"

    def as_dict(self) -> dict:
        return {"name": self.name, "level": self.level,
                "detail": self.detail, "fix": self.fix}


def _torch(device: str) -> list[Finding]:
    """A CPU wheel is the expensive failure: it does not error, it just crawls."""
    try:
        import torch
    except ImportError:
        return [Finding("torch", FATAL, "not installed",
                        "pip install torch==2.5.1 torchvision==0.20.1 "
                        "--index-url https://download.pytorch.org/whl/cu121")]

    out = [Finding("torch", "ok", torch.__version__)]
    if device != "cuda":
        return out

    if not torch.cuda.is_available():
        out.append(Finding(
            "cuda", FATAL,
            f"torch {torch.__version__} reports no CUDA device",
            "A CPU wheel runs everything at roughly a hundredth of the speed "
            "without ever failing. Reinstall from the cu121 index."))
        return out

    name = torch.cuda.get_device_name(0)
    gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    out.append(Finding("cuda", "ok", f"{name}, {gb:.0f} GB"))
    if gb < 12:
        out.append(Finding(
            "vram", DEGRADED, f"{gb:.0f} GB is below the 16 GB this was tuned on",
            "Run the campaign with --batch-size 1."))
    return out


def _dependencies() -> list[Finding]:
    """Optional imports, checked without importing them."""
    out = []
    optional = {
        "skimage": ("the mesh arm", "pip install -r ggssvt/requirements.txt"),
        "sklearn": ("the alternative regressors", "pip install scikit-learn"),
        "PIL": ("raster architecture diagrams", "pip install pillow"),
    }
    for module, (what, fix) in optional.items():
        if importlib.util.find_spec(module) is None:
            out.append(Finding(module, DEGRADED, f"missing, so {what} will fail", fix))
        else:
            out.append(Finding(module, "ok", "present"))
    return out


def _proxies() -> list[Finding]:
    """Malformed proxy variables break every tool, and say so unhelpfully."""
    out = []
    for var in ("http_proxy", "https_proxy", "all_proxy",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        value = os.environ.get(var)
        if not value:
            continue

        # Two schemes, or a scheme followed by something that is not a host,
        # means somebody prefixed a URL that already had one. Git's own message
        # for this names neither the variable nor the value.
        body = value.split("://", 1)[-1]
        if "://" in body:
            out.append(Finding(
                var, FATAL, f"{value!r} has two schemes",
                f"A proxy is http://host:port, one scheme, no path. unset {var}"))
        elif ".pac" in body:
            out.append(Finding(
                var, FATAL, f"{value!r} points at a PAC file",
                "A .pac file is a script that *chooses* a proxy; it is not one. "
                f"unset {var}"))
    if not out:
        out.append(Finding("proxies", "ok", "none set, or well formed"))
    return out


def _ground_truth() -> list[Finding]:
    """The check that would have saved an afternoon."""
    from ..data.dataset import load_ground_truth

    path = DATASET_DIR / "ground_truth.csv"
    if not path.exists():
        return [Finding("ground truth", FATAL, f"not found at {path}")]
    try:
        rows = load_ground_truth(path)
    except ValueError as exc:
        return [Finding("ground truth", FATAL, str(exc)[:200],
                        "Repair the CSV before anything else runs.")]

    masses = sorted(r.net_weight_g / 1000 for r in rows.values())
    measured = sum(1 for r in rows.values() if r.pot_weight_source == "measured")
    return [Finding("ground truth", "ok",
                    f"{len(rows)} specimens, {masses[0]:.2f} to {masses[-1]:.2f} kg, "
                    f"{measured} with measured pots")]


def _huggingface() -> list[Finding]:
    """Which account, and what it can actually download.

    Being logged in is not being granted, and these are different failures with
    the same symptom: a gated cell quietly skipping itself.
    """
    try:
        from huggingface_hub import get_token, whoami
    except ImportError:
        return [Finding("huggingface", DEGRADED, "huggingface_hub not installed")]

    if not get_token():
        return [Finding("huggingface", DEGRADED, "not logged in",
                        "hf auth login  (gated backbones will skip until then)")]
    try:
        account = whoami().get("name", "?")
    except Exception as exc:  # noqa: BLE001 - offline is a note, not a failure
        return [Finding("huggingface", NOTE,
                        f"token present, could not verify: {str(exc)[:80]}")]

    out = [Finding("huggingface", "ok", f"authenticated as {account}")]
    if os.environ.get("HF_TOKEN"):
        out.append(Finding(
            "HF_TOKEN", NOTE,
            "set, and it overrides the cached login",
            "If you are on the wrong account no amount of `hf auth login` "
            "will fix it while this is set."))

    try:
        from ..models.backbones import backbone_is_available

        for kind in ("dinov2", "dinov3"):
            available, reason = backbone_is_available(kind, "base")
            out.append(Finding(
                kind, "ok" if available else DEGRADED,
                "reachable" if available else reason.splitlines()[0][:120],
                "" if available else "That condition will skip itself."))
    except Exception as exc:  # noqa: BLE001
        out.append(Finding("backbones", NOTE, f"not probed: {str(exc)[:80]}"))
    return out


def _disk(work_dir: Path) -> list[Finding]:
    """The caches and checkpoints are tens of gigabytes across a full run."""
    target = work_dir if work_dir.exists() else work_dir.parent
    while not target.exists() and target != target.parent:
        target = target.parent
    free_gb = shutil.disk_usage(target).free / 1024 ** 3
    if free_gb < 20:
        return [Finding("disk", DEGRADED, f"{free_gb:.0f} GB free under {target}",
                        "A full run writes tens of GB of caches and checkpoints.")]
    return [Finding("disk", "ok", f"{free_gb:.0f} GB free")]


def run(*, device: str = "cuda", work_dir: Path = WORK_DIR) -> list[Finding]:
    """Every check, in the order a failure would bite."""
    findings: list[Finding] = []
    findings += _proxies()
    findings += _ground_truth()
    findings += _torch(device)
    findings += _dependencies()
    findings += _huggingface()
    findings += _disk(work_dir)
    return findings


def report(findings: list[Finding], *, verbose: bool = True) -> int:
    """Print the findings. Returns non-zero when something fatal was found."""
    order = {FATAL: 0, DEGRADED: 1, NOTE: 2, "ok": 3}
    width = max(len(f.name) for f in findings)

    for finding in sorted(findings, key=lambda f: order[f.level]):
        if finding.ok and not verbose:
            continue
        mark = {"fatal": "FATAL", "degraded": "warn ", "note": "note ",
                "ok": "ok   "}[finding.level]
        print(f"  {mark} {finding.name:{width}}  {finding.detail}")
        if finding.fix and not finding.ok:
            print(f"        {finding.fix}")

    fatal = [f for f in findings if f.level == FATAL]
    degraded = [f for f in findings if f.level == DEGRADED]
    print(f"\n{len(findings)} checks: {len(fatal)} fatal, {len(degraded)} degraded")
    if fatal:
        print("Fix the fatal ones before starting. Nothing downstream will be right.")
    elif degraded:
        print("Safe to run. The degraded items will silently do less than you "
              "expect, so know which they are.")
    return 1 if fatal else 0


__all__ = ["DEGRADED", "FATAL", "NOTE", "Finding", "report", "run"]
