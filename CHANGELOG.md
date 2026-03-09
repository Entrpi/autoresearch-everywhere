# Changelog

This changelog is intended to be useful for research, not just release bookkeeping.
Each entry records:

- `Human-driven`: the human identified the change and specified it tightly enough that the agent mostly executed.
- `Human-directed, AI-shaped`: the human set the direction or requirement, but the agent designed the concrete mechanism, structure, or validation plan.
- `AI-identified within brief, human-shaped`: inside a broad human-scoped workstream, the agent surfaced the opportunity, and the human materially shaped the exact target, scope, or framing before implementation.
- `AI-identified within brief, human-approved`: inside a broad human-scoped workstream, the agent surfaced the opportunity and the human approved it with little additional shaping.
- `Self-initiated, human-approved`: the agent initiated the change outside explicit human direction in the thread, but still got human approval before landing it.
- `Fully autonomous`: changes or experiments the agent initiated without explicit human direction or approval in the thread.
- `Grounding`: the files changed, the checks run, and any measured effects.

If an entry has no measurements yet, it should say so explicitly.
When provenance is ambiguous, prefer `Human-directed, AI-shaped` over `AI-identified within brief, human-shaped`, prefer `AI-identified within brief, human-shaped` over `AI-identified within brief, human-approved`, prefer `AI-identified within brief, human-approved` over `Self-initiated, human-approved`, and prefer `Self-initiated, human-approved` over `Fully autonomous`.
This branch does not admit fully human-authored code changes. If a change must be authored entirely by a human, it belongs in a fork rather than this branch's mainline history.

## Unreleased

### March 9, 2026 — Working tree — Changelog and provenance policy

**Human-driven**

- Requested a `CHANGELOG.md` with grounded reports on changes rather than a lightweight release log.
- Requested explicit distinction between autonomous and human-in-the-loop work.
- Requested progressively finer provenance tiers for the changelog.
- Corrected specific provenance attributions where the changelog overstated AI agency, with a deliberate bias toward under-claiming rather than over-claiming successful autonomy; the goal remains to push as much work as possible into the `Fully autonomous` category over time.
- Required a branch policy that fully human-authored code changes are out of scope and should happen in a fork.

**Human-directed, AI-shaped**

- Added `CHANGELOG.md` and seeded it with grounded history for the MLX port, evaluation split, preset/token-cache work, and the current streamed-accumulation change.
- Linked the changelog from `README.md`.
- Updated `program_mlx.md` so future experiment loops keep the changelog current.

**AI-identified within brief, human-shaped**

- Proposed intermediate provenance ladders and wording variants; the human materially reshaped them into the current six-tier taxonomy and policy wording.

**AI-identified within brief, human-approved**

- None in this entry.

**Self-initiated, human-approved**

- None in this entry.

**Fully autonomous**

- None in this entry.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program_mlx.md`
- Validation:
  - docs/process only; no code-path tests were needed

### March 9, 2026 — Working tree — Streamed gradient accumulation

**Human-driven**

- None in this entry.

**Human-directed, AI-shaped**

- None in this entry.

**AI-identified within brief, human-shaped**

- Surfaced streamed gradient accumulation earlier as a likely next optimization target in the MLX optimization list.
- The human selected it as the next optimization pass and approved continuing after the `e06f85c` checkpoint.
- Replaced stacked microbatch gathering with streamed gradient accumulation in `train_mlx.py`.
- Split the train step into a compiled per-microbatch gradient pass plus a compiled gradient-application pass.
- Updated the architecture report to reflect the new training flow.

**AI-identified within brief, human-approved**

- None in this entry.

**Self-initiated, human-approved**

- None in this entry.

**Fully autonomous**

- None in this entry.

**Grounding**

- Files:
  - `train_mlx.py`
  - `docs/mlx-port-architecture.md`
- Validation:
  - `python3 -m py_compile train_mlx.py`
  - matched A/B benchmark on `m5-large` against the previous committed training loop from `e06f85c`
- Measured effect on `m5-large` (`5s`, matched settings):
  - steady-state throughput: `15,278.9 tok/s -> 15,279.2 tok/s` (`+0.00%`, effectively flat)
  - peak memory: `2453.8 MB -> 1946.6 MB` (`-507.2 MB`, `-20.7%`)
  - step-0 latency: `533 ms -> 354 ms` (`-33.6%`)
  - steps completed in budget: `18 -> 19`
  - canonical `val_bpb`: `2.373981 -> 2.368727`

## Committed History

### March 9, 2026 — `e06f85c` — Add token caching and calibrate M5 presets

**Human-driven**

- Requested M5-oriented practical defaults rather than retaining H100-shaped settings.
- Requested measured preset documentation in the README, including architecture, batch, throughput, memory, and 5-minute baseline figures.
- Requested investigation of upstream-scale performance on the tested M5 hardware.
- Requested calibration and documentation of the `m5-fast`, `m5-balanced`, `m5-large`, and `m5-xlarge` presets on the reference machine.

**Human-directed, AI-shaped**

- Requested an `m5-xlarge` or "upstream-ish" preset for the tested M5 while leaving the exact shape to the agent.
- Ran the fixed-budget measurements and wrote the concrete throughput, memory, and baseline metric figures used to document the M5 presets.

**AI-identified within brief, human-shaped**

- Surfaced the time-budget bug during benchmarking; the human then requested a fix for that specific issue.
- Implemented the time-budget fix so slow presets stop once accumulated training time exceeds the configured budget.

**AI-identified within brief, human-approved**

- Added prepare-time token caches and cache-aware loading so training no longer has to re-tokenize parquet text on every run.

**Self-initiated, human-approved**

- None in this entry.

**Fully autonomous**

- None in this entry.

**Grounding**

- Files:
  - `prepare_mlx.py`
  - `train_mlx.py`
  - `autoresearch_mlx/data.py`
  - `README.md`
  - `program_mlx.md`
  - `docs/mlx-port-architecture.md`
- Validation:
  - `python3 -m py_compile prepare_mlx.py train_mlx.py tools/overnight_mlx.py autoresearch_mlx/*.py`
  - `./.venv/bin/python prepare_mlx.py --num-shards 1`
  - `./.venv/bin/python train_mlx.py --smoke`
  - multiple fixed-budget runs on the reference M5 MacBook Pro (`32 GB`, `10 GPU cores`)
- Measured effect of token caching, isolated by swapping only the data path back to the pre-cache implementation:
  - `m5-fast`:
    - throughput: `61.6k tok/s -> 64.2k tok/s` (`+4.2%`)
    - step-0 latency: `130 ms -> 55 ms` (`-57.7%`)
    - peak memory: `179.6 MB -> 147.1 MB` (`-18.1%`)
  - `m5-balanced`:
    - throughput: `35.2k tok/s -> 36.0k tok/s` (`+2.3%`)
    - step-0 latency: `180 ms -> 132 ms` (`-26.7%`)
    - peak memory: `951.0 MB -> 951.0 MB` (flat in this probe)
- Baseline 5-minute canonical `val_bpb` figures on the reference machine:
  - `m5-fast`: `1.929023`
  - `m5-balanced`: `1.575952`
  - `m5-large`: `1.601354`
  - `m5-xlarge`: `1.765520`
- Upstream-scale ablation on the same machine:
  - literal `upstream` preset ran at roughly `1.1k-1.9k tok/s`
  - the same `50.3M` / `2048` architecture with M5-sized batch (`m5-xlarge`) ran at roughly `7.4k-8.0k tok/s`

### March 9, 2026 — `eee26f5` — Separate canonical eval and demote local tooling

**Human-driven**

- None in this entry.

**Human-directed, AI-shaped**

- Requested that the fork stay centered on the core MLX research loop rather than letting overnight tooling define the repo.
- Approved moving optional workstation automation out of the core path and clarifying that it is secondary.

**AI-identified within brief, human-shaped**

- Surfaced tooling demotion as part of keeping the core MLX path front and center.
- Moved the overnight automation under `tools/` and rewrote docs to describe it as optional local tooling rather than core architecture.

**AI-identified within brief, human-approved**

- Surfaced canonical-vs-proxy evaluation separation as part of the MLX optimization plan.
- Split evaluation into canonical `val_bpb` and preset-shaped `proxy_val_bpb`.
- Updated the sweep runner to keep or discard runs using canonical `val_bpb`.

**Self-initiated, human-approved**

- None in this entry.

**Fully autonomous**

- None in this entry.

**Grounding**

- Files:
  - `train_mlx.py`
  - `tools/overnight_mlx.py`
  - `tools/launch_overnight_mlx.sh`
  - `tools/detach_exec.py`
  - `README.md`
  - `program_mlx.md`
  - `docs/mlx-port-architecture.md`
- Validation:
  - `python3 -m py_compile train_mlx.py overnight_mlx.py autoresearch_mlx/*.py`
  - `python3 -m py_compile tools/overnight_mlx.py tools/detach_exec.py`
  - `./tools/launch_overnight_mlx.sh tooling-dry-run 0.01 --dry-run`
  - detached one-experiment sweep via the launcher path
- Measured effect:
  - detached sweep produced distinct metrics on the same run:
    - canonical `val_bpb`: `2.509401`
    - proxy `val_bpb`: `2.473907`
  - the sweep runner kept or discarded runs based on canonical `val_bpb`, not the proxy metric

### March 9, 2026 — `c3b3d8d` — Add initial MLX port for Apple Silicon

**Human-driven**

- None in this entry.

**Human-directed, AI-shaped**

- Requested a clean, feature-complete, idiomatic, maintainable MLX reimplementation of upstream `karpathy/autoresearch`.
- Chose Apple Silicon as the primary target and explicitly rejected settling for the PyTorch/MPS SDPA path.
- Implemented the MLX data path, model, optimizer, evaluation path, and Apple-Silicon-first documentation.
- Kept the upstream CUDA path in-tree for reference while making the MLX path the primary workflow.
- Added an MLX-specific program file and smoke-test path.

**AI-identified within brief, human-shaped**

- None in this entry.

**AI-identified within brief, human-approved**

- None in this entry.

**Self-initiated, human-approved**

- None in this entry.

**Fully autonomous**

- None in this entry.

**Grounding**

- Files:
  - `prepare_mlx.py`
  - `train_mlx.py`
  - `autoresearch_mlx/data.py`
  - `autoresearch_mlx/model.py`
  - `autoresearch_mlx/optim.py`
  - `program_mlx.md`
  - `README.md`
  - `pyproject.toml`
- Validation:
  - `python3 -m py_compile prepare_mlx.py train_mlx.py autoresearch_mlx/*.py`
  - `./.venv/bin/python prepare_mlx.py --num-shards 1`
  - `./.venv/bin/python train_mlx.py --smoke`
- Smoke-test result immediately before the baseline commit:
  - canonical `val_bpb`: `2.187049`
  - `training_seconds`: `1.0`
  - `peak_vram_mb`: `194.8`
