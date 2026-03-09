# autoresearch

## About this fork

This fork is an Apple Silicon-first continuation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). It keeps the original CUDA/PyTorch path in-tree for reference, but the primary path here is a clean MLX implementation for macOS, tuned for smaller unified-memory GPUs like the M5.

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The core idea is unchanged: give an agent a small but real language-model training loop, let it run short experiments against a fixed metric and fixed time budget, and keep the ideas that improve validation BPB. In this fork, that loop is centered on MLX and Apple Silicon rather than a single NVIDIA GPU.
By policy, this branch is reserved for AI-shaped or AI-authored code changes; fully human-authored code changes should happen in a fork rather than this mainline history.

## MLX Path

The MLX workflow is built around four files:

- **`prepare_mlx.py`**: one-time data and tokenizer setup for the MLX path.
- **`train_mlx.py`**: the MLX training entrypoint, including M5-oriented presets.
- **`autoresearch_mlx/`**: the MLX dataloader, model, optimizer, and evaluation implementation.
- **`program_mlx.md`**: the baseline prompt/program for agents running the MLX path.

For the full architecture, subsystem boundaries, feature-gap matrix, and flow diagrams, see [docs/mlx-port-architecture.md](docs/mlx-port-architecture.md).
For a grounded history of changes, including measured effects and explicit provenance tiers, see [CHANGELOG.md](CHANGELOG.md).

Training still uses a **fixed 5-minute training budget**. `val_bpb` is now a fixed canonical BPB used for cross-preset comparisons, while `proxy_val_bpb` reports the same-shape local evaluation used for quick inspection.

## Quick Start

**Requirements:** Apple Silicon, macOS, Python 3.10+, and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Download data, train the tokenizer, and build token caches
uv run prepare_mlx.py

# 3. Optional smoke test
uv run train_mlx.py --smoke

# 4. Run the default M5-oriented baseline
uv run train_mlx.py
```

If that works, the MLX environment is ready.

The current MLX port and preset defaults were developed on and tested against an Apple M5 MacBook Pro with 32 GB unified memory and a 10-core GPU. They are a calibrated starting point for that machine, not a promise of universal optimality across the whole M5 family.

`prepare_mlx.py` now builds a reusable shard token cache under `~/.cache/autoresearch/token_cache/` by default so training does not need to re-tokenize parquet text on every run. Use `uv run prepare_mlx.py --skip-token-cache` if you only want the raw data and tokenizer artifacts.

If you want to precompute the row-packing step as well, run `uv run prepare_mlx.py --build-prepacked-cache`. That builds reusable packed-row caches under `~/.cache/autoresearch/prepacked_cache/`, keyed by split and sequence length. `train_mlx.py` and evaluation will prefer those caches automatically when they are present; use `uv run train_mlx.py --no-prepacked-cache` to force the live packing path for debugging or ablations.

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

## Other Upstream Forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) is another macOS-focused fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), but it is closer to a minimal PyTorch/MPS compatibility shim: it keeps the upstream structure largely intact and swaps in SDPA-based attention. This fork is a more opinionated MLX-first rewrite with a packaged training stack, preset system, token caching, and Apple-Silicon-specific workflow changes.

## License

MIT
