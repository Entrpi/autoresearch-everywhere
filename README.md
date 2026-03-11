# autoresearch-everywhere

## About autoresearch-everywhere

autoresearch-everywhere is the glue, generalization, and experiment-logging regime half of the autoresearch core from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The main idea is simple: clone the repo on a machine, let it figure out a good starting configuration for that hardware, and then run the usual autonomous agent research loop from there. MLX is the best-supported path today, and CUDA is already beyond feature parity with upstream here: it runs behind the same generic entrypoints, engine boundary, and architecture-aware runtime layer rather than living as a separate legacy path. The long-term goal is to let more backends plug into the same workflow instead of growing separate forks.

[![Autonomy Golf Badge](docs/autonomy-golf-badge.svg)](#autonomy-golf)

![teaser](docs/assets/progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The core idea is unchanged: give an agent a small but real language-model training loop, let it run an autonomous research loop against a fixed metric and fixed time budget, and keep the ideas that improve validation BPB. In autoresearch-everywhere, the extra work goes into making that loop practical on more than one hardware target instead of assuming a single NVIDIA setup.
By policy, this branch is reserved for AI-shaped or AI-authored code changes; fully human-authored code changes should happen in a fork rather than this mainline history.

## Start Here

Because this project targets broad platform support, the first step is to calibrate the training shape to the machine. The best setup for a $40K datacenter GPU is not the best setup for a laptop, so the repo is designed to find a good starting point for your hardware before it starts the usual autonomous agent research loop.

Initial development was validated against an M5 MacBook Pro, so there are ready-to-use M5 presets you can jump into immediately. More presets should follow with broader adoption and further testing. If this is a new machine, do bring-up first. If this is an M5 machine close to the current reference setup, you can skip straight to training.

**Default path requirements:** Apple Silicon, macOS, Python 3.10+, and [uv](https://docs.astral.sh/uv/). That is the full MLX bring-up path described below. CUDA is also supported behind `--engine cuda`, but with a different hardware/runtime envelope.

### New Hardware Bring-Up

If you are on unfamiliar hardware, start here:

```bash
uv sync
uv run prepare.py
uv run calibrate.py --mode fast
uv run calibrate.py --mode full --output-dir <same-dir-as-fast-run>
```

`calibrate.py` is the new front door to the repo. It:

- identifies the machine
- tries a practical range of preset families
- searches for a good starting training shape
- checks how expensive higher-fidelity evaluation is on that machine
- writes out a report with a recommended default

The report includes:

- a candidate default for that hardware
- lower / recommended / upper / reference zones
- machine-readable artifacts for later promotion or re-checking

By default, the bring-up sweep only considers the practical MLX preset families (`m5-fast` through `m5-xlarge`). Add `--presets ...,upstream` only when you explicitly want the slower upstream-style reference included in the same run.
On Apple Silicon, `prepare.py`, `train.py`, and `calibrate.py` default to the MLX engine automatically. `--engine cuda` is available too. CUDA already exceeds the original upstream path in structure and runtime-awareness here, but MLX is still the only engine with the full evaluation-calibration and default-promotion flow today.

For the actual implementation details, see [docs/platform-calibration.md](docs/platform-calibration.md).

The same tool is also the intended re-check path after meaningful model or runtime changes. If you land something that could change the best settings across machines or preset sizes, rerun platform calibration.

### M4 / M5 Shortcut

If you have a base M4 or M5 Mac, the shipped presets are a good starting point and you can usually skip bring-up to start directly with training. If you are on an M4 Pro, M4 Max, M5 Pro, or M5 Max system, `calibrate.py` should give you a better starting point than the base-machine presets:

```bash
uv sync
uv run prepare.py
uv run train.py --smoke
uv run train.py
```

The current MLX port and shipped defaults were developed on and tested against an Apple M5 MacBook Pro with 32 GB unified memory and a 10-core GPU. They are a calibrated starting point for that workstation class, not a promise of universal optimality across the whole M5 family.

## How It Is Organized

The repo now has a simple top-level surface:

- `prepare.py` prepares data and caches
- `train.py` runs experiments
- `calibrate.py` finds the best starting point for a machine
- `kernel-lab.py` is the backend-specific kernel-lab front door
- `program.md` is the generic agent prompt

Under that, the code is split by role:

- `autoresearch_mlx/` contains the main MLX training stack
- `autoresearch_cuda/` contains the CUDA path
- `autoresearch_platform/` contains the shared engine boundary
- `autoresearch_lab/` contains the shared kernel-lab boundary
- `tools/` contains calibration and optional workstation tooling
- `docs/` contains architecture and workflow notes

The important shift is that the repo is no longer just “an MLX port of `train.py`.” It is now trying to be a small research platform that can bring up a new machine, pick sane defaults, and keep those choices inspectable as the codebase evolves.

For the full architecture, subsystem boundaries, and feature matrix, see [docs/mlx-port-architecture.md](docs/mlx-port-architecture.md).
For a grounded history of changes, including measured effects and provenance tiers, see [CHANGELOG.md](CHANGELOG.md).
For the preset and hardware calibration workflow beneath the one-button bring-up path, see [docs/preset-calibration.md](docs/preset-calibration.md).

## What The Trainer Does For You

By default, the trainer still runs the classic autoresearch pattern: a fixed 5-minute training budget and a final validation BPB score.

The main automatic behaviors are:

- long-context validation by default (`seq_len=2048`)
- automatic evaluation batch sizing based on sequence length
- different validation “rungs” (`cheap`, `reference`, `full`) so short runs do not pay the full cost of the longest possible eval
- conservative fallback when a machine or preset shape is not well calibrated yet
- exact resumable checkpoints for longer MLX runs

On known hardware, the trainer can reuse measured evaluation tradeoffs. On unknown or stale setups, it falls back visibly instead of pretending the old numbers still apply.

The MLX prepare path also builds token caches and prepacked caches by default, so the shipped presets can use the fast data path without extra manual setup.

If you want more detail about the calibration logic beneath those defaults, see [docs/preset-calibration.md](docs/preset-calibration.md).

## Kernel Lab

The repo now also has a top-level `kernel-lab.py` entrypoint for backend-specific kernel work.

The idea is the same one that makes the rest of the repo manageable:

- keep a generic front door at the top level
- keep backend-specific implementation details under `autoresearch_*`
- make room for more than one backend without growing separate one-off workflows

Today the lab is intentionally narrow and MLX-first:

- `uv run kernel-lab.py --engine mlx list-targets`
- `uv run kernel-lab.py --engine mlx init --target rmsnorm --workspace /tmp/mlx-rmsnorm-lab`
- `uv run kernel-lab.py --engine mlx bench --workspace /tmp/mlx-rmsnorm-lab`

That current lab exists to build the pattern, not to claim broad coverage yet. The current starter-ready MLX targets are:

- `rmsnorm`
- `layernorm`
- `rmsnorm_backward`
- `layernorm_backward`
- `residual_blend`
- `residual_rmsnorm`
- `qk_rmsnorm`
- `rope_qk_fused`
- `logits_softcap`
- `activation_pointwise`
- `rotary_embedding`
- `reduce`
- `softmax`
- `value_embed_gate`
- `attention_mask_local`
- `cross_entropy_prelude`
- `fused_mlp`

`flash_attention` stays explicitly deferred as a first target. The long-term reason to keep the lab at the top level now is that the same outer workflow should later host Triton/CUDA, ROCm, and ANE labs without inventing a new orchestration tree each time.

For the current design and scope, see [docs/kernel-lab.md](docs/kernel-lab.md).

## Presets

The MLX training path supports named presets so the default shape is reasonable for Apple Silicon instead of mirroring an H100-oriented baseline.

| Preset | Seq len | Depth / d_model / heads | Params | Batch (device / total tokens) | Window | Approx. tok/sec | Approx. peak memory | 5-min `val_bpb` | 5-min last loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `m5-fast` | `256` | `2 / 128 / 1` | `3.5M` | `2 / 512` | `L` | `75k-90k` | `~213 MB` | `1.929023` | `5.331471` |
| `m5-balanced` | `512` | `4 / 256 / 2` | `11.5M` | `4 / 2048` | `L` | `35k-37k` | `~1.0 GB` | `1.575952` | `4.234088` |
| `m5-large` | `1024` | `6 / 384 / 3` | `26.3M` | `2 / 4096` | `L` | `14k-15k` | `~2.46 GB` | `1.601354` | `4.136220` |
| `m5-xlarge` | `2048` | `8 / 512 / 4` | `50.3M` | `2 / 4096` | `L` | `7.4k-8.0k` | `~4.19 GB` | `1.765520` | `4.688538` |
| `upstream` | `2048` | `8 / 512 / 4` | `50.3M` | `8 / 65536` | `SSSL` | not recommended on this machine | not recommended on this machine | `n/a` | `n/a` |

Window legend: `L` = full causal attention at that layer; `S` = local sliding-window attention; patterns such as `SSSL` repeat across layers with the last layer forced to `L`.

Examples:

```bash
uv run train.py --preset m5-fast
uv run train.py --preset m5-balanced
uv run train.py --preset m5-large
uv run train.py --preset m5-xlarge
uv run train.py --preset upstream
```

These presets may be revised after profiling on newly released M5 Pro and M5 Max systems.

The throughput figures above are approximate steady-state numbers from local runs on the tested 32 GB / 10-core-GPU M5 MacBook Pro with token caches enabled. The batch column is `device_batch_size / total_batch_size`, where `total_batch_size` is tokens per optimizer step after gradient accumulation. The `5-min val_bpb` column is the canonical comparison metric from a full fixed-budget run, and `5-min last loss` is the final debiased smoothed training loss printed at the end of that run.

`m5-xlarge` is the practical way to test the upstream-scale `50.3M` / `2048` model on this machine. `upstream` is kept as the literal reference port, including its H100-shaped batch and `SSSL` attention pattern.

## Optional Local Tooling

This repo also includes optional local automation under `tools/` for slower Apple Silicon machines. It is not part of the core MLX port, but it can be useful when you want unattended preset sweeps on a workstation.

Examples:

```bash
# 30-minute test
./tools/launch_overnight_mlx.sh test30 0.5

# 8-hour overnight run
./tools/launch_overnight_mlx.sh overnight 8

# autonomy scores by day for plotting
python3 tools/changelog_scores.py --group-by day --format csv > autonomy_by_day.csv
```

Artifacts are written under `results/overnight/<run-tag>/`, and the summary ledger is appended to `results/results.tsv`. The sweep runner keeps or discards experiments using canonical `val_bpb`, not the preset-shaped proxy metric.

## Running an Agent

Point your coding agent at `program.md` first. Use `docs/program-mlx.md` as the MLX-specific supplement when the task is Apple-Silicon-first or otherwise MLX-specific.

Example prompt:

```text
Read program.md, then docs/program-mlx.md if the task is MLX-specific, verify the setup, and start a new experiment loop.
```

## Autonomy Golf

[![Autonomy Golf Badge](docs/autonomy-golf-badge.svg)](https://github.com/Entrpi/autonomy-golf)

This repo is playing autonomy golf. We use [CHANGELOG.md](CHANGELOG.md), a parser, and a badge to keep score as we try to drive the project down toward total autonomy, or hole-in-one games, without getting sloppy about evidence.

Canonical GitHub home: [Entrpi/autonomy-golf](https://github.com/Entrpi/autonomy-golf)

This repo already has the full bundle installed:

- [CHANGELOG.md](CHANGELOG.md): local autonomy-golf history and parser source of truth
- [docs/autonomy-golf.md](docs/autonomy-golf.md): the portable manifesto
- [docs/autonomy-golf-agent.md](docs/autonomy-golf-agent.md): reusable integration brief
- [docs/autonomy-golf-checklist.md](docs/autonomy-golf-checklist.md): the maintenance loop for future updates
- [tools/changelog_scores.py](tools/changelog_scores.py): rollups and verification
- [tools/render_autonomy_badge.py](tools/render_autonomy_badge.py): badge and README snapshot refresh

Autonomy golf works here because the Score and its Grounding are managed with agent integration and tooling in a gamified loop that also helps clarify project purpose and change motivation:

- `score`: how autonomous a change really was
- `Grounding`: how well the change was validated

Lower is better, and this branch is reserved for AI-shaped or AI-authored code changes; fully human-authored code changes should happen in a fork rather than this mainline history.

For the full scoring model, the meaning of the game, and the reusable adoption docs, see [docs/autonomy-golf.md](docs/autonomy-golf.md).

<!-- autonomy-golf-snapshot:start -->
Current project snapshot from [CHANGELOG.md](CHANGELOG.md):

| Metric | Value |
| --- | --- |
| Mean autonomy score | `3.40 / 6` |
| Mean complexity | `7.56 / commit` |
| Mean score per top-level bullet | `3.47 / 6` |
| History covered | `39` commits across `12` subsystems |
<!-- autonomy-golf-snapshot:end -->

Refresh with:

```bash
python3 tools/render_autonomy_badge.py
```

## Project Structure

```text
prepare.py            — generic data prep entrypoint with engine dispatch
train.py              — generic training entrypoint with engine dispatch
calibrate.py          — one-button platform bring-up calibration
kernel-lab.py                — top-level backend-specific kernel lab entrypoint
program.md            — generic agent instructions
autoresearch_mlx/prepare.py        — direct MLX data prep implementation
autoresearch_mlx/train.py          — direct MLX training implementation
autoresearch_mlx/     — MLX data/model/optimizer implementation
autoresearch_mlx/kernel-lab.py            — MLX kernel lab CLI implementation
autoresearch_mlx/lab_workspace.py  — MLX mutable workspace + fixed bench harness
autoresearch_lab/     — shared kernel-lab boundary
docs/program-mlx.md        — MLX agent instructions
docs/kernel-lab.md         — kernel-lab design and workflow note
docs/assets/         — generated docs assets such as the progress figure
notebooks/           — exploratory notebooks and analysis
results/results.tsv  — experiment result ledger
autoresearch_cuda/    — CUDA implementation and runtime policy
tools/               — optional local sweep tooling
pyproject.toml        — dependencies
```

## Appendix: CUDA Path

The NVIDIA-oriented path is no longer just the original upstream code kept around for comparison. It now lives behind the same generic entrypoints and shared engine boundary as MLX, with architecture-family-aware runtime policy layered on top.

- `autoresearch_cuda/prepare.py`: CUDA data prep and runtime utilities.
- `autoresearch_cuda/train.py`: CUDA training implementation.
- `program.md`: generic agent prompt, with CUDA-specific guidance pointing into `autoresearch_cuda/`.

If you are on a single NVIDIA GPU and want the original workflow, use those files instead of the MLX ones.

## Other Upstream Forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) is another macOS-focused fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), but it is closer to a minimal PyTorch/MPS compatibility shim: it keeps the upstream structure largely intact and swaps in SDPA-based attention. autoresearch-everywhere is a more opinionated MLX-first rewrite with a packaged training stack, preset system, token caching, and Apple-Silicon-specific workflow changes.

## License

MIT
