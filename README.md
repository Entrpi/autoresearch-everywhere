# autoresearch

## About this fork

This fork is an Apple Silicon-first continuation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). It keeps the original CUDA/PyTorch path in-tree for reference, but the primary path here is a clean MLX implementation for macOS, tuned for smaller unified-memory GPUs like the M5.

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The core idea is unchanged: give an agent a small but real language-model training loop, let it run short experiments against a fixed metric and fixed time budget, and keep the ideas that improve validation BPB. In this fork, that loop is centered on MLX and Apple Silicon rather than a single NVIDIA GPU.

## MLX Path

The MLX workflow is built around four files:

- **`prepare_mlx.py`**: one-time data and tokenizer setup for the MLX path.
- **`train_mlx.py`**: the MLX training entrypoint, including M5-oriented presets.
- **`autoresearch_mlx/`**: the MLX dataloader, model, optimizer, and evaluation implementation.
- **`program_mlx.md`**: the baseline prompt/program for agents running the MLX path.

For the full architecture, subsystem boundaries, feature-gap matrix, and flow diagrams, see [docs/mlx-port-architecture.md](docs/mlx-port-architecture.md).

Training still uses a **fixed 5-minute training budget**. `val_bpb` is now a fixed canonical BPB used for cross-preset comparisons, while `proxy_val_bpb` reports the same-shape local evaluation used for quick inspection.

## Quick Start

**Requirements:** Apple Silicon, macOS, Python 3.10+, and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Download data and train the tokenizer
uv run prepare_mlx.py

# 3. Optional smoke test
uv run train_mlx.py --smoke

# 4. Run the default M5-oriented baseline
uv run train_mlx.py
```

If that works, the MLX environment is ready.

The current MLX port and preset defaults were developed on and tested against an Apple M5 MacBook Pro with 32 GB unified memory and a 10-core GPU. They are a calibrated starting point for that machine, not a promise of universal optimality across the whole M5 family.

## Presets

`train_mlx.py` supports named presets so the default shape is reasonable for Apple Silicon instead of mirroring an H100-oriented baseline.

- `m5-fast`: quick local iteration with small memory use.
- `m5-balanced`: default preset for M5-class machines.
- `m5-large`: slower, larger-capacity run for longer experiments.
- `upstream`: reference shape closest to the original upstream defaults.

Examples:

```bash
uv run train_mlx.py --preset m5-fast
uv run train_mlx.py --preset m5-balanced
uv run train_mlx.py --preset m5-large
uv run train_mlx.py --preset upstream
```

These presets may be revised after profiling on newly released M5 Pro and M5 Max systems.

## Optional Local Tooling

This repo also includes optional local automation under `tools/` for slower Apple Silicon machines. It is not part of the core MLX port, but it can be useful when you want unattended preset sweeps on a workstation.

Examples:

```bash
# 30-minute test
./tools/launch_overnight_mlx.sh test30 0.5

# 8-hour overnight run
./tools/launch_overnight_mlx.sh overnight 8
```

Artifacts are written under `results/overnight/<run-tag>/`, and the summary ledger is appended to `results.tsv`. The sweep runner keeps or discards experiments using canonical `val_bpb`, not the preset-shaped proxy metric.

## Running an Agent

Point your coding agent at `program_mlx.md`, not `program.md`.

Example prompt:

```text
Read program_mlx.md, verify the MLX setup, and start a new experiment loop.
```

## Project Structure

```text
prepare_mlx.py        — MLX data prep entrypoint
train_mlx.py          — MLX training entrypoint
autoresearch_mlx/     — MLX data/model/optimizer implementation
program_mlx.md        — MLX agent instructions
tools/               — optional local sweep tooling
pyproject.toml        — dependencies
```

## Appendix: Upstream CUDA Path

The original NVIDIA-oriented path is still present for reference and comparison.

- `prepare.py`: upstream-style data prep and runtime utilities.
- `train.py`: upstream single-file CUDA/PyTorch training script.
- `program.md`: upstream-style agent prompt for the CUDA path.

If you are on a single NVIDIA GPU and want the original workflow, use those files instead of the MLX ones.

## Notable Forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos)

## License

MIT
