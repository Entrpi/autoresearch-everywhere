# autoresearch

## About this fork

This fork is an Apple Silicon-first continuation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). It keeps the original CUDA/PyTorch path in-tree for reference, but the primary path here is a clean MLX implementation for macOS, tuned for smaller unified-memory GPUs like the M5.

[![Autonomy Golf Badge](docs/autonomy-golf-badge.svg)](#autonomy-golf)

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The core idea is unchanged: give an agent a small but real language-model training loop, let it run short experiments against a fixed metric and fixed time budget, and keep the ideas that improve validation BPB. In this fork, that loop is centered on MLX and Apple Silicon rather than a single NVIDIA GPU.
By policy, this branch is reserved for AI-shaped or AI-authored code changes; fully human-authored code changes should happen in a fork rather than this mainline history.

## Start Here

The idealized new-user path is now platform bring-up first, experimentation second.

**Requirements:** Apple Silicon, macOS, Python 3.10+, and [uv](https://docs.astral.sh/uv/).

### New Hardware Bring-Up

If you are on unfamiliar hardware, the intended path is:

```bash
# 1. Install dependencies
uv sync

# 2. Download data, train the tokenizer, and build caches
uv run prepare_mlx.py

# 3. Find the best starting point for this machine
uv run tools/calibrate_platform.py --mode fast

# 4. Optional stronger calibration pass
uv run tools/calibrate_platform.py --mode full --output-dir <same-dir-as-fast-run>
```

That flow is the new front door to the repo. The bring-up tool fingerprints the machine, explores the shipped preset families, runs a constrained local search, calibrates eval rungs on the chosen operating point, and emits:

- a Markdown report
- a JSON artifact
- a candidate new default for the autoresearch stage on that hardware
- lower / recommended / upper / reference zones
- a promotion bundle that says which artifacts are immediately promotable and which still need a fuller audit

By default, the bring-up sweep only considers the practical MLX preset families (`m5-fast` through `m5-xlarge`). Add `--presets ...,upstream` only when you explicitly want the slower upstream-style reference included in the same run.

For the actual implementation details, see [docs/platform-calibration.md](docs/platform-calibration.md).

The same tool is also the intended revalidation path after meaningful autoresearch changes to the model or runtime, such as a new MLP block or a more efficient attention implementation. Use judgment first: if the finding looks like it could generalize across preset shapes or hardware classes, rerun platform calibration proactively. Platform and eval calibrations are also stamped with runtime and eval signatures, but those are the conservative backstop rather than the main trigger.

### Known M5 Reference Path

If you are on a machine close to the current reference hardware, you can skip bring-up and start directly with the known M5 defaults:

```bash
uv sync
uv run prepare_mlx.py
uv run train_mlx.py --smoke
uv run train_mlx.py
```

The current MLX port and shipped defaults were developed on and tested against an Apple M5 MacBook Pro with 32 GB unified memory and a 10-core GPU. They are a calibrated starting point for that workstation class, not a promise of universal optimality across the whole M5 family.

## System Overview

The MLX workflow is now built around five subsystems:

- **Platform bring-up**: `tools/calibrate_platform.py`
- **Runtime eval policy**: `autoresearch_mlx/eval_policy.py` and `autoresearch_mlx/eval_telemetry.py`
- **Training engine**: `train_mlx.py`, `autoresearch_mlx/model.py`, `autoresearch_mlx/optim.py`
- **Data and evaluation substrate**: `prepare_mlx.py`, `autoresearch_mlx/data.py`
- **Agent loop**: `program_mlx.md`

The important shift is that the repo is no longer just “an MLX port of `train.py`.” It is now a small Apple-Silicon research platform with explicit machine bring-up, runtime trust signals, and promotion-ready calibration artifacts.

For the full architecture, subsystem boundaries, and feature matrix, see [docs/mlx-port-architecture.md](docs/mlx-port-architecture.md).
For a grounded history of changes, including measured effects and provenance tiers, see [CHANGELOG.md](CHANGELOG.md).
For the preset and hardware calibration workflow beneath the one-button bring-up path, see [docs/preset-calibration.md](docs/preset-calibration.md).

## Training Defaults

The trainer still uses a **fixed 5-minute training budget** by default. `train_mlx.py` supports `--time-budget-mode train|wall`, but the default `train` mode preserves the original autoresearch intent: budget is accounted in accumulated optimizer-step time rather than raw wall time.

Canonical evaluation is now long-context and upstream-oriented by default:

- `seq_len=2048`
- reduced-budget `cheap/reference/full` rungs
- auto batch sizing that targets about `4096` tokens per eval step
- sliced sampling across the upstream horizon for reduced-budget rungs
- sequential full-eval for the full upstream-sized rung

Shipped preset shapes use the checked-in eval tradeoff tables by default when canonical eval settings are not manually overridden, but only on exact hardware-key matches. The runtime surfaces calibration status, effective confidence, freshness, telemetry coverage, stable rung coverage, and last-seen date so underfilled or stale calibration is visible instead of implicit; unmatched or stale rows fall back visibly to the default canonical settings.
The runtime also surfaces code-signature matches for eval semantics and runtime shape, so architecture or runtime changes can visibly invalidate a previously trusted calibration even on the same machine.

`prepare_mlx.py` builds both the reusable shard token cache under `~/.cache/autoresearch/token_cache/` and the shipped prepacked row caches under `~/.cache/autoresearch/prepacked_cache/` by default, so all shipped M5 presets can hit the fast path without extra setup. Use `uv run prepare_mlx.py --skip-token-cache` if you only want the raw data and tokenizer artifacts, or `uv run prepare_mlx.py --skip-prepacked-cache` if you explicitly want to leave training on the live packing fallback path.

The trainer also supports exact resumable checkpoints. For runs longer than 5 minutes, the MLX path enables exact full-state checkpoints by default using a conservative interval selector grounded in measured resume costs on this machine. Exact sync remains the default checkpoint path. `--checkpoint-save-mode async` is available as an optional exact-resume variant for wall-clock-constrained runs, but it is still not the default path.

## Presets

`train_mlx.py` supports named presets so the default shape is reasonable for Apple Silicon instead of mirroring an H100-oriented baseline.

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
uv run train_mlx.py --preset m5-fast
uv run train_mlx.py --preset m5-balanced
uv run train_mlx.py --preset m5-large
uv run train_mlx.py --preset m5-xlarge
uv run train_mlx.py --preset upstream
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

Artifacts are written under `results/overnight/<run-tag>/`, and the summary ledger is appended to `results.tsv`. The sweep runner keeps or discards experiments using canonical `val_bpb`, not the preset-shaped proxy metric.

## Running an Agent

Point your coding agent at `program_mlx.md`, not `program.md`.

Example prompt:

```text
Read program_mlx.md, verify the MLX setup, and start a new experiment loop.
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
| Mean autonomy score | `3.38 / 6` |
| Mean complexity | `7.48 / commit` |
| Mean score per top-level bullet | `3.47 / 6` |
| History covered | `31` commits across `9` subsystems |
<!-- autonomy-golf-snapshot:end -->

Refresh with:

```bash
python3 tools/render_autonomy_badge.py
```

## Project Structure

```text
prepare_mlx.py        — MLX data prep entrypoint
train_mlx.py          — MLX training entrypoint
autoresearch_mlx/     — MLX data/model/optimizer implementation
program_mlx.md        — MLX agent instructions
tools/calibrate_platform.py — one-button platform bring-up calibration
tools/               — optional local sweep tooling
pyproject.toml        — dependencies
```

## Appendix: Upstream CUDA Path

The original NVIDIA-oriented path is still present for reference and comparison.

- `prepare.py`: upstream-style data prep and runtime utilities.
- `train.py`: upstream single-file CUDA/PyTorch training script.
- `program.md`: upstream-style agent prompt for the CUDA path.

If you are on a single NVIDIA GPU and want the original workflow, use those files instead of the MLX ones.

## Other Upstream Forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) is another macOS-focused fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), but it is closer to a minimal PyTorch/MPS compatibility shim: it keeps the upstream structure largely intact and swaps in SDPA-based attention. This fork is a more opinionated MLX-first rewrite with a packaged training stack, preset system, token caching, and Apple-Silicon-specific workflow changes.

## License

MIT
