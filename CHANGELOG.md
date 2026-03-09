# Changelog

This changelog is intended to be useful for research, not just release bookkeeping.
Each entry records:

- `Human-driven (5)`: the human identified the change and specified it tightly enough that the agent mostly executed.
- `Human-directed, AI-shaped (4)`: the human set the direction or requirement, but the agent designed the concrete mechanism, structure, or validation plan.
- `AI-identified within brief, human-shaped (3)`: inside a broad human-scoped workstream, the agent surfaced the opportunity, and the human materially shaped the exact target, scope, or framing before implementation.
- `AI-identified within brief, human-approved (2)`: inside a broad human-scoped workstream, the agent surfaced the opportunity and the human approved it with little additional shaping.
- `Self-initiated, human-approved (1)`: the agent initiated the change outside explicit human direction in the thread, but still got human approval before landing it.
- `Fully autonomous (0)`: changes or experiments the agent initiated without explicit human direction or approval in the thread.
- `Grounding`: the files changed, the checks run, and any measured effects.

If an entry has no measurements yet, it should say so explicitly.
Entries should omit empty provenance sections rather than spelling out `None in this entry.`
Each commit entry should also show an autonomy golf score in the header. Score each provenance bullet as `Human-driven = 5`, `Human-directed, AI-shaped = 4`, `AI-identified within brief, human-shaped = 3`, `AI-identified within brief, human-approved = 2`, `Self-initiated, human-approved = 1`, and `Fully autonomous = 0`. `Grounding` does not contribute to the score.
Top-level provenance bullets are the scored units. If a point is directly derivative of a main bullet and stays at the same autonomy level, record it as a nested sub-bullet so it remains visible without adding score.
When provenance is ambiguous, prefer `Human-directed, AI-shaped` over `AI-identified within brief, human-shaped`, prefer `AI-identified within brief, human-shaped` over `AI-identified within brief, human-approved`, prefer `AI-identified within brief, human-approved` over `Self-initiated, human-approved`, and prefer `Self-initiated, human-approved` over `Fully autonomous`.
This changelog should bias toward under-claiming rather than over-claiming successful autonomy. When specific provenance attributions are corrected, prefer the more conservative tiering if there is real ambiguity. The long-term goal remains to push as much work as possible into the `Fully autonomous` category over time.
This branch does not admit fully human-authored code changes. If a change must be authored entirely by a human, it belongs in a fork rather than this branch's mainline history.
Grounding should also be conservative. When a claim is about performance, stability, or behavioral improvement, prefer the strongest practical evidence over the quickest smoke pass, and record the actual strength of that evidence rather than the intended standard.
For benchmarked changes, "strong enough" means long enough and heavy enough to produce a high-signal result on the changed behavior. Choose a run shape where the affected path executes enough times to matter. For example, checkpoint-overhead claims should usually be grounded with a run that produces many checkpoint saves rather than only one or two, and scaling claims should prefer a model/preset large enough for the bottleneck to show up clearly.
On this hardware, the default canonical matched benchmark window for optimization grounding is `60s`, not `30s`. Use shorter runs for smoke checks or when the changed path cannot practically support a longer benchmark, and say so explicitly when you do.

## Unreleased

### New commit — Add checkpoint/resume for MLX training — score `6`

**Human-directed, AI-shaped (4)**

- Requested stronger grounding on the checkpoint/resume work by extending the benchmark window and then testing the heavier `m5-xlarge` preset, while leaving the exact comparison design to the agent.

**AI-identified within brief, human-approved (2)**

- Requested that work continue to the next optimization item after landing the lazy-cache checkpoint.
  - Added resumable checkpoint save/load for the MLX trainer, including model weights, optimizer state, runtime counters, and train-loader cursor state.
  - Added periodic step-boundary checkpoint saves plus explicit `--resume-from`, `--checkpoint-path`, and `--checkpoint-interval` flags.
  - Added train-loader serialization for both prepacked and live-packed paths so resume continues from the saved training position instead of restarting the data stream.

**Grounding**

- Files:
  - `autoresearch_mlx/checkpoints.py`
  - `autoresearch_mlx/data.py`
  - `train_mlx.py`
  - `README.md`
  - `docs/mlx-port-architecture.md`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile train_mlx.py autoresearch_mlx/data.py autoresearch_mlx/checkpoints.py`
  - `./.venv/bin/python train_mlx.py --smoke --checkpoint-path /tmp/autoresearch_resume_smoke2 --checkpoint-interval 0.5`
  - `./.venv/bin/python train_mlx.py --resume-from /tmp/autoresearch_resume_smoke2 --time-budget 1.5`
  - `./.venv/bin/python train_mlx.py --smoke --checkpoint-path /tmp/autoresearch_resume_smoke3 --checkpoint-interval 0.5`
- Measurements:
  - matched `m5-large` run (`20s`, `512` eval tokens) with and without periodic checkpoint saves every `2s`:
    - wall-clock overhead outside tracked training time: `0.1s -> 0.8s` (`+0.7s`) across ten checkpoint writes, or about `0.07s/save`
    - peak memory: `1944.3 MB -> 1944.3 MB` (flat)
    - steady-state per-step throughput stayed in the same `~15k tok/s` band
    - `num_steps` stayed flat at `76 -> 76`, which is the cleaner fixed-budget result
  - matched `m5-xlarge` run (`60s`, `512` eval tokens) with and without periodic checkpoint saves every `2s`:
    - wall-clock overhead outside tracked training time: `0.2s -> 3.5s` (`+3.3s`) across twenty-nine checkpoint writes, or about `0.11s/save`
    - peak memory: `4294.2 MB -> 4294.2 MB` (flat)
    - `num_steps` stayed flat at `115 -> 115`
    - steady-state per-step throughput stayed in the same `~7.8k-8.0k tok/s` band
- Confirmed behavior:
  - checkpoint saves succeeded during smoke runs without breaking evaluation
  - resume restored the saved state at `step=133`, continued training to `step=186`, and preserved the same train-loader position rather than restarting from zero

### New commit — Tighten grounding guidance — score `4`

**Human-directed, AI-shaped (4)**

- Requested that the changelog explicitly track grounding shaping and that the agent guidance become more eager to robustly ground changes before claiming wins.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `program_mlx.md`
- Validation:
  - docs-only change; no runtime validation needed

### New commit — Standardize 60s benchmark grounding window — score `5`

**Human-driven (5)**

- Requested that, given the constrained Apple Silicon hardware, the canonical matched benchmark run length for grounding optimization changes be `60s` instead of `30s`.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `program_mlx.md`
- Validation:
  - matched `60s` preset reruns with `--eval-tokens 512 --canonical-eval-tokens 512` for:
    - `m5-fast`
    - `m5-balanced`
    - `m5-large`
    - `m5-xlarge`
- Measurements:
  - `m5-fast`: `val_bpb=1.964834`, `proxy_val_bpb=1.900012`, `~70.1k tok/s`, `174.5 MB`, `8213` steps
  - `m5-balanced`: `val_bpb=1.757194`, `proxy_val_bpb=1.741310`, `~34.5k tok/s`, `949.9 MB`, `1011` steps
  - `m5-large`: `val_bpb=1.901965`, `proxy_val_bpb=1.935591`, `~14.6k tok/s`, `1944.3 MB`, `215` steps
  - `m5-xlarge`: `val_bpb=2.150093`, `proxy_val_bpb=2.122628`, `~7.8k tok/s`, `4294.2 MB`, `114` steps
- Notes:
  - `m5-fast` and `m5-balanced` used the train-side prepacked cache path during these reruns.
  - `m5-large` and `m5-xlarge` used the token-cache plus live-packing path, so their throughput figures are conservative relative to a fully prepacked `1024/2048` cache setup.

## Committed History

### March 9, 2026 — `f1d14e8` — Lazy-grow model caches — score `2`

**AI-identified within brief, human-approved (2)**

- Requested the next optimization pass on lazy-growing the model caches after the current priority review.
  - Replaced the fixed `sequence_len * 10` RoPE cache with a smaller startup cache that grows up to the configured model sequence length.
  - Reworked local-attention mask caching to keep one growable mask per window size and slice it for shorter requests instead of caching separate masks per exact sequence length.
  - Prewarmed runtime caches in `train_mlx.py` before compiling the train step so cache growth stays out of the compiled hot path.

**Grounding**

- Files:
  - `autoresearch_mlx/model.py`
  - `train_mlx.py`
  - `CHANGELOG.md`
- Validation:
  - `python3 -m py_compile train_mlx.py autoresearch_mlx/model.py`
  - `./.venv/bin/python train_mlx.py --smoke`
  - `./.venv/bin/python train_mlx.py --preset m5-xlarge --time-budget 0.01 --eval-tokens 512 --canonical-eval-tokens 512`
  - matched A/B benchmark on `m5-xlarge` against the last committed cache behavior from `2be14fe`
- Measurements:
  - matched `m5-xlarge` run (`5s`, `512` eval tokens):
    - completed updates in budget: `10 -> 11`
    - fixed-budget throughput: `7.88k tok/s -> 8.34k tok/s` (`+5.9%`)
    - step-0 latency: `583 ms -> 582 ms` (flat)
    - peak memory: `4298.7 MB -> 4294.2 MB` (effectively flat)
  - ultra-short startup probe (`0.01s`) was noisy and favored the baseline in total wall-clock time, so the longer fixed-budget run is the more reliable comparison
- Confirmed behavior:
  - the smoke path still trains and evaluates end-to-end
  - the `m5-xlarge` path completed with the lazily prewarmed `2048`-token cache setup instead of relying on the old eager `10x` RoPE allocation strategy

### March 9, 2026 — `2be14fe` — Add changelog score parser — score `4`

**Human-directed, AI-shaped (4)**

- Requested a changelog parser for future autonomy plotting and later tightened the requirement to include explicit commit-count tracking for daily per-commit views.
  - Added `tools/changelog_scores.py`.
  - Added per-entry and per-day output.
  - Added CSV/JSON/TSV formats, score verification, and daily `commit_count` plus per-commit score fields.

**Grounding**

- Files:
  - `tools/changelog_scores.py`
  - `CHANGELOG.md`
  - `README.md`
  - `program_mlx.md`
- Validation:
  - `python3 -m py_compile tools/changelog_scores.py`
  - `python3 tools/changelog_scores.py --group-by entry --format csv --include-unreleased --verify`
  - `python3 tools/changelog_scores.py --group-by day --format csv --include-unreleased`
- Historical score corrections surfaced by the parser:
  - `9a4ef79`: `39 -> 40`
  - `e06f85c`: `35 -> 36`

### March 9, 2026 — `2bcdc0c` — Add optional prepacked row caches — score `6`

**Human-directed, AI-shaped (4)**

- Implemented split-and-sequence-length keyed prepacked row caches, an opt-in `prepare_mlx.py` build path, runtime preference with fallback, and a trainer flag to disable the caches for ablations.

**AI-identified within brief, human-approved (2)**

- Requested work on optional prepacked caches after landing the optimizer checkpoint.

**Grounding**

- Files:
  - `autoresearch_mlx/constants.py`
  - `autoresearch_mlx/data.py`
  - `prepare_mlx.py`
  - `train_mlx.py`
  - `README.md`
  - `program_mlx.md`
  - `docs/mlx-port-architecture.md`
- Validation:
  - `python3 -m py_compile prepare_mlx.py train_mlx.py autoresearch_mlx/data.py autoresearch_mlx/constants.py`
  - `./.venv/bin/python prepare_mlx.py --num-shards 1 --build-prepacked-cache --prepacked-seq-lens 256,512`
  - `./.venv/bin/python train_mlx.py --smoke`
  - matched A/B benchmark on `m5-fast` against the live token-cache packing path using `--no-prepacked-cache`
- Measured effect on `m5-fast` (`2s`, matched settings):
  - step-0 latency: `48 ms -> 37 ms` (`-22.9%`)
  - completed updates in budget: `233 -> 255`
  - fixed-budget throughput: `59.65k tok/s -> 65.28k tok/s` (`+9.4%`)
  - peak memory: `147.1 MB -> 147.1 MB` (flat)
  - validation metrics: effectively unchanged within short-run noise
  - confirmed runtime behavior: train and val loaders both switched to `prepacked cache` when the matching cache existed

### March 9, 2026 — `137ba69` — Remove optimizer tree churn — score `3`

**AI-identified within brief, human-shaped (3)**

- Surfaced optimizer tree churn as the next high-value optimization after streamed gradient accumulation and, after human selection, removed it from the hot path.
  - Reworked `MuonAdamW` to cache stable parameter slots on the model.
  - Fetched gradients by cached path tokens and wrote updated arrays back directly instead of flattening and unflattening the full parameter tree on every step.
  - Cached per-group slot lists so the Muon path no longer rebuilds them inside the hot loop.

**Grounding**

- Files:
  - `autoresearch_mlx/optim.py`
- Validation:
  - `python3 -m py_compile autoresearch_mlx/optim.py`
  - matched A/B benchmark on `m5-large` against the last committed optimizer from `9a4ef79`
- Measured effect on `m5-large` (`5s`, matched settings):
  - step-0 latency: `558 ms -> 365 ms` (`-34.6%`)
  - completed updates in budget: `17 -> 18`
  - fixed-budget throughput: `13.65k tok/s -> 14.18k tok/s` (`+3.8%`)
  - peak memory: `1946.6 MB -> 1946.6 MB` (flat)
  - steady-state per-step throughput: roughly flat within run-to-run noise

### March 9, 2026 — `9a4ef79` — Add changelog and provenance policy — score `17`

**Human-driven (5)**

- Requested a grounded `CHANGELOG.md` rather than a lightweight release log, with explicit autonomy distinctions and progressively finer provenance tiers.
- Corrected provenance overclaims with a deliberate under-claiming bias and required a branch policy that fully human-authored code changes happen in a fork.

**Human-directed, AI-shaped (4)**

- Added `CHANGELOG.md` and integrated it into the repo workflow.
  - Seeded it with grounded history for the MLX port, evaluation split, preset/token-cache work, and the streamed-accumulation change.
  - Linked the changelog from `README.md`.
  - Updated `program_mlx.md` so future experiment loops keep the changelog current.

**AI-identified within brief, human-shaped (3)**

- Proposed intermediate provenance ladders and wording variants; the human materially reshaped them into the current six-tier taxonomy and policy wording.

**Grounding**

- Files:
  - `CHANGELOG.md`
  - `README.md`
  - `program_mlx.md`
- Validation:
  - docs/process only; no code-path tests were needed

### March 9, 2026 — `077a187` — Stream gradient accumulation in MLX trainer — score `3`

**AI-identified within brief, human-shaped (3)**

- Surfaced streamed gradient accumulation as a likely next optimization target and, after human selection, replaced stacked microbatch gathering with a streamed training loop.
  - Split the train step into a compiled per-microbatch gradient pass plus a compiled gradient-application pass.
  - Updated the architecture report to reflect the new training flow.

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

### March 9, 2026 — `e06f85c` — Add token caching and calibrate M5 presets — score `19`

**Human-driven (5)**

- Requested M5-oriented practical defaults rather than retaining H100-shaped settings.
- Requested measured investigation and documentation of the reference-machine presets, including upstream-scale behavior and the README baseline figures.

**Human-directed, AI-shaped (4)**

- Requested an `m5-xlarge` or "upstream-ish" preset for the tested M5 while leaving the exact shape to the agent, then used fixed-budget measurements to document the concrete throughput, memory, and baseline figures.

**AI-identified within brief, human-shaped (3)**

- Surfaced the time-budget bug during benchmarking and, after the human requested that specific fix, implemented the budget-stop correction for slow presets.

**AI-identified within brief, human-approved (2)**

- Added prepare-time token caches and cache-aware loading so training no longer has to re-tokenize parquet text on every run.

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

### March 9, 2026 — `eee26f5` — Separate canonical eval and demote local tooling — score `9`

**Human-directed, AI-shaped (4)**

- Requested that the fork stay centered on the core MLX research loop and approved demoting optional workstation automation so it would not define the repo.

**AI-identified within brief, human-shaped (3)**

- Surfaced tooling demotion as part of keeping the core MLX path front and center, then moved the overnight automation under `tools/` and rewrote docs to describe it as optional local tooling rather than core architecture.

**AI-identified within brief, human-approved (2)**

- Surfaced canonical-vs-proxy evaluation separation as part of the MLX optimization plan, then split evaluation into canonical `val_bpb` and preset-shaped `proxy_val_bpb` and updated the sweep runner to keep or discard runs using canonical `val_bpb`.

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

### March 9, 2026 — `c3b3d8d` — Add initial MLX port for Apple Silicon — score `4`

**Human-directed, AI-shaped (4)**

- Requested a clean, feature-complete, idiomatic, maintainable MLX reimplementation of upstream `karpathy/autoresearch` for Apple Silicon rather than the PyTorch/MPS SDPA path, then implemented the core MLX path and its Apple-Silicon-first docs.
  - Implemented the MLX data path, model, optimizer, and evaluation path.
  - Kept the upstream CUDA path in-tree for reference while making the MLX path the primary workflow.
  - Added an MLX-specific program file and smoke-test path.

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
