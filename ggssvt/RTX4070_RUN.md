# GG-SSVT on an RTX 4070

The GG-SSVT environment is separate from the repository's Nerfstudio
environment. From the repository root:

```bash
tools/setup_ggssvt_rtx4070.sh
conda activate ggssvt
python -m ggssvt.cli preflight
```

Run a campaign conservatively on a 12 GB RTX 4070:

```bash
tools/run_campaigns_rtx4070.sh smoke
tools/run_campaigns_rtx4070.sh core
tools/run_campaigns_rtx4070.sh full
```

The helper defaults to `query_chunk=128`, `workers=0`, and batch size 1. These
settings trade speed for lower peak VRAM. A campaign writes its checkpoints and
fold results below `work_dirs/ggssvt/campaign/` and resumes completed runs and
folds after interruption.

To continue an existing campaign in the background:

```bash
nohup tools/run_campaigns_rtx4070.sh core \
  > work_dirs/ggssvt/campaign_core.log 2>&1 &
```

Monitor it with:

```bash
tail -f work_dirs/ggssvt/campaign_core.log
nvidia-smi
```

The current repository dependency file is
[requirements.txt](requirements.txt). Install PyTorch first using the CUDA
index as shown by the setup script; do not install a CPU-only wheel.
