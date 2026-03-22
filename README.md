# AUTORESEARCH-EVERYWHERE

![Banner](docs/assets/autoresearch-everywhere.png)

## About AUTORESEARCH-EVERYWHERE

AUTORESEARCH-EVERYWHERE is the glue, generalization, and experiment-logging regime half of the autoresearch core from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The main idea is simple: clone the repo on a machine, let it figure out a good starting configuration for that hardware, and then run the usual autonomous agent research loop from there. Two validated fast-track lanes exist today: Apple M4/M5 on MLX and DGX Spark / GB10 on CUDA with FA4. Beyond those reference paths, the repo still uses the same generic entrypoints, engine boundary, and architecture-aware runtime layer rather than growing separate hardware forks. The long-term goal is to let more backends plug into one workflow instead of splitting into per-platform branches.

[![Autonomy Golf Badge](docs/autonomy-golf-badge.svg)](#autonomy-golf)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The core idea is unchanged: give an agent a small but real language-model training loop, let it run an autonomous research loop against a fixed metric and fixed time budget, and keep the ideas that improve validation BPB. In autoresearch-everywhere, the extra work goes into making that loop practical on more than one hardware target instead of assuming a single NVIDIA setup.
By policy, this branch is reserved for AI-shaped or AI-authored code changes; fully human-authored code changes should happen in a fork rather than this mainline history.

## Start Here

Because this project targets broad platform support, the first step is to find the fastest path to being productive on the machine you actually have. The best setup for a recent ML-focused professional laptop is not the best setup for a recent Blackwell workstation, so the repo is designed to calibrate model shape, batch shape, and evaluation cost to your hardware before it starts the usual autonomous agent research loop.

Two validated fast-track paths exist today:

- Apple M4 / M5 on MLX
- DGX Spark / GB10 on CUDA with the FlashAttention 4 runtime from [docs/dgx-spark-setup.md](docs/dgx-spark-setup.md)

If you are on one of those reference machines, skip to the matching fast-track section below. If this is a different machine, use the generic bring-up flow first.

**Validated fast-track requirements:** either Apple Silicon, macOS, Python 3.12+, and [uv](https://docs.astral.sh/uv/), or a DGX Spark / GB10 host with the FA4-capable runtime image described in [docs/dgx-spark-setup.md](docs/dgx-spark-setup.md). Other NVIDIA hardware still uses the same top-level commands, but Spark / GB10 is the currently validated CUDA shortcut path.

## Running an Agent

Point your coding agent at `program.md` first.

Example prompt:

```text
Read program.md, verify the setup, and start a new experiment loop. NEVER STOP EXPERIMENTING.
```

## M4 / M5 Fast Track

If you have a base M4 or M5 Mac, the shipped presets are a good starting point and you can usually skip straight to training. If you are on an M4 Pro, M4 Max, M5 Pro, or M5 Max system, `calibrate.py` should still give you a better starting point than the base-machine presets:

```bash
uv sync
uv run prepare.py
uv run train.py --smoke
uv run train.py
```

The current MLX port and shipped defaults were developed on and tested against an Apple M5 MacBook Pro with 32 GB unified memory and a 10-core GPU. They are a calibrated starting point for that workstation class, not a promise of universal optimality across the whole M5 family.

## DGX Spark / GB10 Fast Track

If you are on a DGX Spark / GB10 box then follow (or have your agent follow) [docs/dgx-spark-setup.md](docs/dgx-spark-setup.md) to get your runtime setup with FlashAttention 4. The normal loop is smoke, fast calibration, then the research run using the calibrated CUDA default.

Inside the `vllm-node-tf5-fa4:sm120` runtime image, the shortcut is:

```bash
python prepare.py --engine cuda
python train.py --engine cuda --smoke
python -u calibrate.py --engine cuda --mode fast --output-dir /output
python train.py --engine cuda
```

That path is already validated on real GB10 hardware and you can expect `val_bpb` to start in the 1.16 range *before* the autoresearch loop.

## Other Hardware Bring-Up

If you are not on one of the validated fast-track paths above, start here:

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
- a comparison between your machine, what works best on an M5 laptop, and the H100-oriented starting point Karpathy hand-shaped in the upstream project
- when the data supports it, a secondary scaling candidate: a larger near-frontier model that stays close enough to the strict `300s` winner to be interesting for longer horizons
- machine-readable artifacts for later promotion or re-checking

It also writes the candidate default into the local platform-default cache for that engine and hardware key. After that, real kernel-lab integration tests can use the calibrated point for the current device automatically instead of requiring a manual preset every time.

By default, the bring-up sweep only considers the practical MLX preset families (`m5-tiny`, `m5-small`, `m5-balanced`, `m5-large`, and `m5-xlarge`). Add `--presets ...,upstream` only when you explicitly want the slower upstream-style reference included in the same run.
On Apple Silicon, `prepare.py`, `train.py`, and `calibrate.py` default to the MLX engine automatically. On NVIDIA, use the same commands with `--engine cuda`. MLX currently has the deepest local eval-calibration and default-promotion flow. CUDA already runs as a first-class engine with the same front-door commands, architecture-aware runtime behavior, and the stronger automated trace-review path in kernel-lab; DGX Spark / GB10 is the currently validated fast-track reference for that CUDA path.

For the actual implementation details, see [docs/platform-calibration.md](docs/platform-calibration.md).

The same tool is also the intended re-check path after meaningful model or runtime changes. If you land something that could change the best settings across machines or preset sizes, rerun platform calibration.

### Other NVIDIA Hardware

CUDA is still a first-class engine on A100-class Ampere, Ada, Hopper, B200, and other NVIDIA systems. The same front-door commands apply there too:

```bash
uv sync
uv run prepare.py --engine cuda
uv run train.py --engine cuda --smoke
uv run calibrate.py --engine cuda --mode fast
```

What changes by hardware is the runtime envelope and what is already validated:

- DGX Spark / GB10 is the current validated CUDA fast track
- other NVIDIA hardware should still run `calibrate.py` first and treat the report as the source of truth
- CUDA keeps the same top-level entrypoints as MLX, plus architecture-aware runtime behavior, exact checkpoints, and the stronger automated kernel-lab trace-review path

For the concrete GB10 host, container, FA4, and profiling setup path, see [docs/dgx-spark-setup.md](docs/dgx-spark-setup.md). For the broader parity and rollout assessment, see [docs/cuda-core-loop-parity.md](docs/cuda-core-loop-parity.md).

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
- `tools/` contains calibration and manual workstation sweep tooling
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

## Presets

The preset system exists to give you a clear scale reference between what works best on a ~$2,000 Apple M5 laptop vs a ~$20,000 H100 datacenter GPU. `calibrate.py` then shows you where your machine fits on that spectrum.

If you are on a base M4 or M5 Mac and want to start quickly, use:

- `m5-small` if you want the best default starting point
- `m5-tiny` if you want the fastest cheap experiment loop
- `m5-balanced` if you want the best current validation-centered local run
- `m5-large` if you want something closer to the upstream model shape without jumping all the way to xlarge
- `m5-xlarge` if you want the largest practical local model on this class of machine

If you are on an M4 Pro, M4 Max, M5 Pro, M5 Max, or anything outside that reference class, run `calibrate.py` first. The bring-up report is the better answer for both productivity and comparison; the table below is just rough orientation.

`upstream` is not a normal starting preset. It is the literal upstream-shaped reference: the H100-oriented starting point Karpathy hand-shaped in the original project. Keep it around for comparison, not as the usual first thing to run locally.

### Preset Reference (metrics are results from an M5 Mac)

| Preset          | Best first use                        | Seq len  | Depth / d_model / heads | Params    | Batch (device / total tokens) | Window    | Approx. tok/sec | 5-min steps | Approx. peak memory | 5-min `val_bpb` | 5-min last loss |
| --------------- | ------------------------------------- | -------- | ----------------------- | --------- | ----------------------------- | --------- | --------------- | ----------- | ------------------- | ----------------- | --------------- |
| `m5-tiny`     | Fast iteration                        | `256`  | `2 / 128 / 1`         | `3.5M`  | `4 / 12288`                 | `L`     | `~103k`       | `2516`    | `~282 MB`         | `1.715180`      | `4.134272`    |
| `m5-small`    | Default starting point                | `512`  | `4 / 256 / 2`         | `11.5M` | `4 / 12288`                 | `L`     | `~46k`        | `1066`    | `~1.01 GB`        | `1.441619`      | `3.939787`    |
| `m5-balanced` | Best validation target                | `1024` | `6 / 384 / 3`         | `26.3M` | `4 / 12288`                 | `SSSSL` | `~18.2k`      | `444`     | `~2.77 GB`        | `1.428708`      | `4.162026`    |
| `m5-large`    | Upstream-leaning bridge run           | `512`  | `8 / 512 / 4`         | `50.3M` | `4 / 16384`                 | `SSSSL` | `~13.5k`      | `250`     | `~2.66 GB`        | `1.606594`      | `4.521518`    |
| `m5-xlarge`   | Largest practical local run           | `2048` | `8 / 512 / 4`         | `50.3M` | `4 / 16384`                 | `L`     | `~8.8k`       | `162`     | `~7.44 GB`        | `1.748045`      | `4.933070`    |
| `upstream`    | Too heavy for laptops, abysmally slow | `2048` | `8 / 512 / 4`         | `50.3M` | `8 / 65536`                 | `SSSL`  | `n/a`         | `n/a`     | `n/a`             | `n/a`           | `n/a`         |

Window legend: `L` = full causal attention at that layer; `S` = local sliding-window attention; patterns such as `SSSL` repeat across layers with the last layer forced to `L`.

Examples:

```bash
uv run train.py --preset m5-tiny
uv run train.py --preset m5-small
uv run train.py --preset m5-balanced
uv run train.py --preset m5-large
uv run train.py --preset m5-xlarge
uv run train.py --preset upstream
```

These presets may be revised after profiling on newer Apple Silicon systems and on non-Apple hardware as the broader calibration flow gets more adoption.

The throughput figures above are approximate session-average numbers from fresh 5-minute local runs on the tested 32 GB / 10-core-GPU M5 MacBook Pro with token caches enabled. The batch column is `device_batch_size / total_batch_size`, where `total_batch_size` is tokens per optimizer step after gradient accumulation. The `5-min steps` column is the total optimizer-step count completed in that fixed budget. The `5-min val_bpb` column is the canonical comparison metric from the final evaluation, and `5-min last loss` is the final debiased smoothed training loss printed at the end of the run. `m5-xlarge` and `m5-large` use `16384` total tokens because `12288` is not divisible by `4 × 2048`, and the new `m5-balanced` row uses `SSSSL` because that long-context local-window mix beat dense `L` on both `val_bpb` and throughput in matched 5-minute reruns. `m5-large` is intentionally shipped before it has a checked-in eval ladder row, so it currently uses the explicit canonical fallback path until that calibration is added. The names now describe where a preset sits relative to the current best validation-centered local target, not just raw parameter count.

`m5-xlarge` is the practical way to test the upstream-scale `50.3M` / `2048` model on this machine. `upstream` is kept as the literal reference port, including the H100-shaped batch and `SSSL` attention pattern Karpathy chose upstream, so it is useful for comparison but usually not the right first thing to run.

## Kernel Lab

Kernel-lab is the safe workshop for low-level speedups that are too risky to drop straight into the main trainer.

Its job is simple:

- find repeated backend-level operations that look worth optimizing
- open a small mutable workspace for one target
- prove the candidate in isolation with fixed bench and verify steps
- gather real backend evidence
- only then test it against the real trainer and decide whether it deserves promotion

That workflow is shared across backends even though the substrate changes:

- Metal kernels on Apple GPUs
- Triton/CUDA kernels on NVIDIA
- future ROCm or other accelerator-specific paths later

If you already know tools like CUTLASS or Triton, the easiest framing is: those are implementation substrates; kernel-lab is the workflow layer above them that decides what is worth pursuing, how it is measured, and when it is strong enough to enter the real training path.

What it can do today:

- MLX:
  - broad starter-target catalog
  - workspace bench/verify flow
  - Metal capture artifacts
  - trainer-side integration A/B and promotion checks
- CUDA:
  - trace-first workflow with Nsight Systems and Nsight Compute
  - starter Triton-backed workspaces for narrow target families
  - tested against GB10 (DGX Spark, Blackwell) with FlashAttention via [FA4 PR](https://github.com/Dao-AILab/flash-attention/pull/2268)
- Shared:
  - one top-level entrypoint: `kernel-lab.py`
  - persistent evidence ledger in `results/kernel_lab/ledger.jsonl`
  - orchestration that reuses prior evidence instead of treating every target as a fresh idea

Typical MLX path:

```bash
uv run kernel-lab.py --engine mlx profile --preset m5-balanced --top-k 8 --output /tmp/mlx-profile.json
uv run kernel-lab.py --engine mlx orchestrate --profile /tmp/mlx-profile.json --workspace-root /tmp/mlx-lab
```

Typical CUDA path:

```bash
uv run kernel-lab.py --engine cuda capture --preset upstream --time-budget 20 --output /tmp/cuda-upstream-trace
uv run kernel-lab.py --engine cuda trace-profile --metadata /tmp/cuda-upstream-trace.metadata.json --output /tmp/cuda-upstream-trace.profile.json
```

The practical difference between the backends is:

- MLX has the deeper workspace and trainer-integration loop
- CUDA has the stronger automated trace-review path

The detailed workflows, target catalogs, GB10 setup notes, and profiling requirements live in [docs/kernel-lab.md](docs/kernel-lab.md).

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
| Mean autonomy score | `3.20 / 6` |
| Mean complexity | `6.98 / commit` |
| Mean score per top-level bullet | `3.26 / 6` |
| History covered | `97` commits across `17` subsystems |
<!-- autonomy-golf-snapshot:end -->

Refresh with:

```bash
python3 tools/render_autonomy_badge.py
```

For autonomy-history plotting:

```bash
python3 tools/changelog_scores.py --group-by day --format csv --include-latest > autonomy_by_day.csv
```

## Manual Longer Sweeps

If you want to let one machine grind through longer preset sweeps by hand, the repo also includes local sweep tooling under `tools/`. It is not part of the core training or bring-up path; it is there for cases where you want to manually run a longer workstation sweep and inspect the results afterward.

Examples:

```bash
# 30-minute test
./tools/launch_overnight_mlx.sh test30 0.5

# 8-hour overnight run
./tools/launch_overnight_mlx.sh overnight 8
```

Artifacts are written under `results/overnight/<run-tag>/`, and the summary ledger is appended to `results/results.tsv`. The sweep runner keeps or discards experiments using canonical `val_bpb`, not the preset-shaped proxy metric

## Project Structure

Top-level entrypoints:

- `prepare.py` — generic data preparation entrypoint with engine dispatch
- `train.py` — generic training entrypoint with engine dispatch
- `calibrate.py` — platform bring-up and default-selection entrypoint
- `kernel-lab.py` — backend-specific kernel-lab entrypoint
- `program.md` — generic agent instructions

Core packages:

- `autoresearch_mlx/` — MLX trainer, data, model, optimizer, and MLX-specific kernel-lab helpers
- `autoresearch_cuda/` — CUDA trainer, runtime policy, checkpoints, and attention/runtime integration
- `autoresearch_platform/` — shared engine boundary, calibration logic, and cross-backend projection/reporting code
- `autoresearch_lab/` — shared kernel-lab orchestration and promotion boundary

Supporting directories:

- `docs/` — setup guides, architecture notes, workflow docs, and generated assets
- `tools/` — reporting, calibration helpers, and manual longer-sweep tooling
- `results/` — run artifacts and summary ledgers
- `notebooks/` — exploratory analysis
- `pyproject.toml` — project metadata and dependencies

## Other Upstream Forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) is another macOS-focused fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch), but it is closer to a minimal PyTorch/MPS compatibility shim: it keeps the upstream structure largely intact and swaps in SDPA-based attention. autoresearch-everywhere is a more opinionated MLX-first rewrite with a packaged training stack, preset system, token caching, and Apple-Silicon-specific workflow changes.

## License

MIT
