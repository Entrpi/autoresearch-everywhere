# autoresearch-everywhere

## About autoresearch-everywhere

autoresearch-everywhere is the glue, generalization, and experiment-logging regime half of the autoresearch core from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The main idea is simple: clone the repo on a machine, let it figure out a good starting configuration for that hardware, and then run the usual autonomous agent research loop from there. MLX is the best-supported path today, and CUDA is already beyond feature parity with upstream here: it runs behind the same generic entrypoints, engine boundary, and architecture-aware runtime layer rather than living as a separate legacy path. The long-term goal is to let more backends plug into the same workflow instead of growing separate forks.

[![Autonomy Golf Badge](docs/autonomy-golf-badge.svg)](#autonomy-golf)

![teaser](docs/assets/progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The core idea is unchanged: give an agent a small but real language-model training loop, let it run an autonomous research loop against a fixed metric and fixed time budget, and keep the ideas that improve validation BPB. In autoresearch-everywhere, the extra work goes into making that loop practical on more than one hardware target instead of assuming a single NVIDIA setup.
By policy, this branch is reserved for AI-shaped or AI-authored code changes; fully human-authored code changes should happen in a fork rather than this mainline history.

## Start Here

Because this project targets broad platform support, the first step is to find the fastest path to being productive on the machine you actually have. The best setup for a recent ML-focused professional laptop is not the best setup for a recent $40K datacenter GPU, so the repo is designed to calibrate model shape, batch shape, and evaluation cost to your hardware before it starts the usual autonomous agent research loop.

Initial development was validated against an M5 MacBook Pro, so there are ready-to-use M5 presets you can jump into immediately. More presets should follow with broader adoption and further testing. If this is a new machine, do bring-up first. If this is a base M4 or M5 machine, you can usually skip straight to training.

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
- a comparison between your machine, what works best on an M5 laptop, and the H100-oriented starting point Karpathy hand-shaped in the upstream project
- machine-readable artifacts for later promotion or re-checking

It also writes the candidate default into the local platform-default cache for that engine and hardware key. After that, real kernel-lab integration tests can use the calibrated point for the current device automatically instead of requiring a manual preset every time.

By default, the bring-up sweep only considers the practical MLX preset families (`m5-tiny`, `m5-small`, `m5-balanced`, `m5-large`, and `m5-xlarge`). Add `--presets ...,upstream` only when you explicitly want the slower upstream-style reference included in the same run.
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

Kernel-lab is the workshop for trying low-level speedups safely.

Training spends time in lots of small repeated operations: norms, reshapes, rotary embedding, attention setup, loss-side reductions, and related path glue. Some of those are good candidates for custom kernels, but dropping kernel experiments straight into the main trainer is risky. A candidate can be correct but irrelevant, fast in isolation but useless end to end, or only beneficial on one machine.

Kernel-lab exists to separate:

- "this looks like a promising low-level optimization"
- "this is proven enough to earn a place in the real training path"

So the flow is deliberately staged:

- profile likely targets
- create a small mutable workspace for one target
- benchmark and verify it in isolation
- capture a real backend trace when needed
- then test it against the real trainer before considering promotion

That workflow is meant to generalize across backends even though the kernel substrate changes. Metal kernels on Apple GPUs, Triton/CUDA kernels on NVIDIA, HIP/ROCm kernels on AMD, and future accelerator-specific paths can all plug into the same outer loop.

If you already know tools like CUTLASS or Triton, the easiest framing is: those are implementation substrates; kernel-lab is the workflow layer above them that decides which kernel opportunities are worth pursuing, how they are benchmarked, how trace evidence is collected, and what it takes to promote them into the actual training engine.

Today the lab is still MLX-deep first, but CUDA now has the first trace-first automation path:

- `uv run kernel-lab.py --engine mlx list-targets`
- `uv run kernel-lab.py --engine mlx profile --preset m5-balanced --top-k 8 --output /tmp/mlx-profile.json`
- `uv run kernel-lab.py --engine mlx orchestrate --profile /tmp/mlx-profile.json --workspace-root /tmp/mlx-lab`
- `uv run kernel-lab.py --engine mlx evidence --target block_prelude --preset m5-balanced`
- `uv run kernel-lab.py --engine mlx promotion-check --target block_prelude`
- `uv run kernel-lab.py --engine mlx init --target rmsnorm --workspace /tmp/mlx-rmsnorm-lab`
- `uv run kernel-lab.py --engine mlx bench --workspace /tmp/mlx-rmsnorm-lab`
- `uv run kernel-lab.py --engine mlx verify --workspace /tmp/mlx-rmsnorm-lab --quick`
- `uv run kernel-lab.py --engine mlx capture --workspace /tmp/mlx-rmsnorm-lab --output /tmp/mlx-rmsnorm-lab.gputrace --quick`
- `uv run kernel-lab.py --engine mlx review-trace --workspace /tmp/mlx-rmsnorm-lab --metadata /tmp/mlx-rmsnorm-lab.metadata.json --relevance high`
- `uv run kernel-lab.py --engine mlx integration-ab --workspace /tmp/mlx-rmsnorm-lab --time-budget 20 --benchmark-skip-eval --no-checkpoint`

CUDA now has the first trace-first automation path too, and a small first starter-workspace layer on top of it:

- `uv run kernel-lab.py --engine cuda list-targets`
- `uv run kernel-lab.py --engine cuda capture --preset upstream --time-budget 20 --output /tmp/cuda-upstream-trace`
- `uv run kernel-lab.py --engine cuda trace-profile --metadata /tmp/cuda-upstream-trace.metadata.json --output /tmp/cuda-upstream-trace.profile.json`
- `uv run kernel-lab.py --engine cuda auto-review --trace-profile /tmp/cuda-upstream-trace.profile.json`
- `uv run kernel-lab.py --engine cuda deep-profile --trace-profile /tmp/cuda-upstream-trace.profile.json --rank 1`
- `uv run kernel-lab.py --engine cuda evidence --target launch_fusion --preset upstream`
- `uv run kernel-lab.py --engine cuda orchestrate --trace-profile /tmp/cuda-upstream-trace.profile.json --workspace-root /tmp/cuda-lab`
- `uv run kernel-lab.py --engine cuda promotion-check --target launch_fusion --preset upstream`
- `uv run kernel-lab.py --engine cuda extract --profile /tmp/cuda-upstream-trace.profile.json --workspace /tmp/cuda-lab/launch_fusion --rank 1`
- `uv run kernel-lab.py --engine cuda bench --workspace /tmp/cuda-lab/launch_fusion --device cuda --quick`
- `uv run kernel-lab.py --engine cuda verify --workspace /tmp/cuda-lab/launch_fusion --device cuda --quick`
- `uv run kernel-lab.py --engine cuda integration-ab --workspace /tmp/cuda-lab/norm --preset upstream --time-budget 20 --benchmark-skip-eval --no-checkpoint`
- `uv run kernel-lab.py --engine cuda integration-suite --workspace /tmp/cuda-lab/norm --preset upstream --time-budget 20 --repeats 2 --benchmark-skip-eval --no-checkpoint`

The lab now has two layers on purpose:

- heuristic layer:
  - `profile` ranks likely MLX kernel targets for a preset using model-aware heuristics
  - `extract` turns a ranked profile result into a mutable workspace with saved context
  - `orchestrate` emits the next ready command sequence for a selected target, and can now reuse an existing workspace when earlier verify/capture evidence already exists
  - `verify` reruns the fixed harness as the promotion gate above a quick bench
- trace layer:
  - `capture` records a real MLX Metal trace and a metadata sidecar
  - the `.gputrace` artifact is the truth source when a candidate starts making performance claims instead of just being an interesting idea
  - `orchestrate --trace-metadata ...` can fold a real capture back into the next suggested workflow

For CUDA, the split is similar but the trace side is more automatable:

- `capture` wraps the real trainer under Nsight Systems and writes a `.nsys-rep` plus a metadata sidecar
- `trace-profile` turns the exported Nsight reports into ranked kernel target families
- `auto-review` classifies the run as launch-bound, sync-bound, copy-bound, kernel-dominated, or mixed
- `deep-profile` optionally reruns the top trace-ranked family under Nsight Compute so the lab can record a more specific diagnosis such as compute-bound, bandwidth-bound, or under-occupied
- a strong deeper diagnosis now increases rank/promotion confidence, while a weak or mixed one keeps the target in review instead of letting it drift toward promotion on timing share alone
- `evidence` and `promotion-check` expose whether a target is still just trace-ranked, already trace-backed, ready for a starter workspace, or deprioritized by the automated review
- `orchestrate` now consumes the trace profile plus accumulated evidence to pick the next CUDA target family to pursue, and for starter-ready families it emits real `extract` / `bench` / `verify` commands instead of just placeholder notes
- starter CUDA workspaces currently exist for:
  - `launch_fusion`
  - `norm`
  - `logits_softcap`
  - `loss_prelude`
- `data_movement`
- `matmul_epilogue`
- `attention_prelude`
- `value_embed_gate`
- `rope_qk_fused`
- `fused_mlp`
- the first Triton-backed CUDA workspace slice is intentionally narrow:
  - `launch_fusion` now ships with an optional Triton residual-add kernel
  - `norm` now ships with an optional Triton RMSNorm kernel
  - `logits_softcap` now ships with an optional Triton pointwise softcap kernel
  - `value_embed_gate` now ships with an optional Triton pointwise gate-application kernel
  - `data_movement` now ships with an optional Triton copy/reshape kernel
  - `matmul_epilogue` now ships with an optional Triton matmul+bias kernel
  - the rest of the CUDA starter catalog, including `loss_prelude`, `attention_prelude`, `rope_qk_fused`, and `fused_mlp`, is still reference-first, so the CUDA substrate can grow incrementally instead of pretending every starter target is already Triton-native
- `norm`, `logits_softcap`, `loss_prelude`, `matmul_epilogue`, `fused_mlp`, `attention_prelude`, `value_embed_gate`, `rope_qk_fused`, `launch_fusion`, and `data_movement` currently have direct CUDA trainer-side hooks, so they are the CUDA starter targets that can collect real `integration-ab` / `integration-suite` evidence today
- on non-CUDA machines or machines without PyTorch/CUDA installed, those CUDA integration commands return structured `missing-runtime` results instead of pretending the target is promotable
- the long-term goal is that CUDA trace review becomes automated-by-default, with GUI inspection as the escalation path rather than the first step

The lab also keeps a small evidence ledger at `results/kernel_lab/ledger.jsonl`. That lets later profiles and plans see whether a target is still unexplored, only verified, trace-backed, or ready for an integration A/B instead of treating every target as a fresh idea.

The extra commands make that visible:

- `evidence` summarizes the current ledger state for one target/preset or across presets
- `promotion-check` says whether a target is still gathering evidence, ready for an end-to-end integration suite, mixed after repeated A/B runs, or already validated strongly enough to move toward a real trainer patch
- if you omit `--preset`, `promotion-check`, `integration-ab`, and `integration-suite` use the calibrated platform default for the current device; only smoke or deliberately targeted tests should usually pin a different preset by hand
- `review-trace` lets a human or agent record whether a trace showed strong, weak, or negligible end-to-end relevance, so ranking can move down as well as up
- `integration-ab` now runs repeated balanced trainer-side comparisons for the subset of targets that already have direct MLX integration hooks, and promotion stays conservative until the effect is repeated and directionally stable
- `integration-suite` takes that one step further by testing the calibrated point and the next stronger preset by default, so trainer-side evidence can survive beyond one operating point

Only part of the MLX target catalog is directly wired into the trainer today. Small path targets like `logits_softcap`, `rotary_embedding`, `value_embed_gate`, `attention_prelude`, and `fused_mlp` can already run end-to-end A/B through `integration-ab` and `integration-suite`. Broader composed targets like `block_prelude` can still be profiled, verified, and traced, but they will report `needs-integration-adapter` until there is a direct training-path hook for them.

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
- `ve_lookup_reshape`
- `attention_mask_local`
- `proj_head_reshape`
- `loss_logits_cast_softcap`
- `cross_entropy_prelude`
- `cross_entropy_full`
- `attention_prelude`
- `block_prelude`
- `fused_mlp`

`flash_attention` stays explicitly deferred as a first target. The long-term reason to keep the lab at the top level now is that the same outer workflow should later host Triton/CUDA, ROCm, and ANE labs without inventing a new orchestration tree each time.

For the current design and scope, see [docs/kernel-lab.md](docs/kernel-lab.md).

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

| Preset          | Best first use              | Seq len  | Depth / d_model / heads | Params    | Batch (device / total tokens) | Window   | Approx. tok/sec | 5-min steps | Approx. peak memory | 5-min `val_bpb` | 5-min last loss |
| --------------- | --------------------------- | -------- | ----------------------- | --------- | ----------------------------- | -------- | ---------------- | ------------ | ------------------- | ---------------- | --------------- |
| `m5-tiny`       | Fast iteration              | `256`    | `2 / 128 / 1`           | `3.5M`    | `4 / 12288`                  | `L`      | `~103k`          | `2516`       | `~282 MB`           | `1.715180`       | `4.134272`      |
| `m5-small`      | Default starting point      | `512`    | `4 / 256 / 2`           | `11.5M`   | `4 / 12288`                  | `L`      | `~46k`           | `1066`       | `~1.01 GB`          | `1.441619`       | `3.939787`      |
| `m5-balanced`   | Best validation target      | `1024`   | `6 / 384 / 3`           | `26.3M`   | `4 / 12288`                  | `SSSSL`  | `~18.2k`         | `444`        | `~2.77 GB`          | `1.428708`       | `4.162026`      |
| `m5-large`      | Upstream-leaning bridge run | `512`    | `8 / 512 / 4`           | `50.3M`   | `4 / 16384`                  | `SSSSL`  | `~13.5k`         | `250`        | `~2.66 GB`          | `1.606594`       | `4.521518`      |
| `m5-xlarge`     | Largest practical local run | `2048`   | `8 / 512 / 4`           | `50.3M`   | `4 / 16384`                  | `L`      | `~8.8k`          | `162`        | `~7.44 GB`          | `1.748045`       | `4.933070`      |
| `upstream`      | Too heavy for laptops, abysmally slow | `2048`   | `8 / 512 / 4`           | `50.3M`   | `8 / 65536`                  | `SSSL`   | `n/a`            | `n/a`        | `n/a`               | `n/a`            | `n/a`           |

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
| Mean autonomy score | `3.31 / 6` |
| Mean complexity | `7.19 / commit` |
| Mean score per top-level bullet | `3.37 / 6` |
| History covered | `67` commits across `12` subsystems |
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
kernel-lab.py         — top-level backend-specific kernel lab entrypoint
program.md            — generic agent instructions
autoresearch_mlx/prepare.py        — direct MLX data prep implementation
autoresearch_mlx/train.py          — direct MLX training implementation
autoresearch_mlx/     — MLX data/model/optimizer implementation
autoresearch_mlx/lab.py            — MLX kernel lab CLI implementation
autoresearch_mlx/lab_profile.py    — MLX profile/extract/orchestrate heuristics
autoresearch_mlx/lab_trace.py      — MLX capture-mode and trace artifact support
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
