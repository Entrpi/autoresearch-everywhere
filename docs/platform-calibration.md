# Platform Calibration

This note describes the implemented one-button platform bring-up flow for new hardware, plus the closely related post-change revalidation path for meaningful autoresearch changes. The bring-up stack now sits on a shared training-engine boundary so the same workflow can host MLX first, CUDA second, and later ROCm or ANE without each backend becoming its own orchestration fork.

## User Story

A new user clones the repo onto unfamiliar hardware. They should be able to run one long calibration command and get:

- a recommended starting preset and operating point
- a candidate new default for the autoresearch stage on that hardware
- lower and upper bounding zones for that hardware
- a clear comparison to the checked-in M5 reference
- a clear comparison to the upstream-style reference shape
- a report that says what is measured, what is inferred, and what is still weakly calibrated

The point is not just to benchmark the machine. The point is to find the best initial research posture for that machine and explain why.

The same system should also answer a second story:

- the hardware is unchanged
- but the researcher changed something substantial in the training stack
- for example: SwiGLU, a different attention implementation, or another shape-affecting runtime improvement

In that case the tool should behave as a revalidation path, not just a first-time bring-up path.

## Primary Trigger: Judgment First

The intended trigger for rerunning platform calibration is agent or researcher judgment first, static signatures second.

The policy should be:

- if the findings suggest the change could alter the best operating point across preset shapes or hardware classes, rerun platform calibration proactively
- if the change appears local and non-generalizing, rely on the lighter preset-calibration path or ordinary grounding
- if the signatures no longer match, treat that as a conservative forced fallback, not as the only reason to recalibrate

Examples that should normally trigger platform revalidation even on the same machine:

- a new MLP block such as SwiGLU
- a more efficient or more memory-hungry attention implementation
- changes to compile strategy, kernel path, checkpointing behavior, or batching behavior that may shift throughput or memory pressure across preset families
- changes to eval semantics that alter what canonical quality means

The signatures are intentionally simpler than this judgment. They exist to stop the runtime from silently over-trusting stale rows, not to replace explicit reasoning about which findings are likely to generalize.

## Outcome

The current command is:

```bash
uv run calibrate.py
```

By default, that command means `--engine mlx` and sweeps only the practical MLX preset families. The slower `upstream` preset remains available as an explicit opt-in reference, but it is too expensive on typical local MLX hardware to belong in the default one-button path.

The same front door now also accepts `--engine cuda`. That path already shares:

- hardware fingerprinting
- comparable train probes
- coarse envelope / ranking / zone reporting
- candidate-default reporting

but it does not yet share the full MLX-only checkpoint-backed eval-calibration and promotion path. That is intentional: the training-engine boundary is real first, feature parity second.

That command may take a long time, but it now leaves behind:

- a complete bring-up report
- enough machine-readable artifacts to support later runtime policy decisions
- a concrete candidate default for the autoresearch stage on that hardware
- reusable phase artifacts so a later rerun can resume from the same output directory instead of starting over

Those outputs are now also stamped with runtime and eval signatures, so the report can be interpreted as "valid for this code state on this hardware," not just "valid for this hardware forever." But those signatures are only the conservative floor. The real policy is still to rerun proactively when the findings indicate likely cross-shape or cross-hardware impact.

## Relationship To The Current Tooling

The current stack already provides most of the underlying pieces:

- `autoresearch_platform/`
  - shared engine contract and backend registry
- `autoresearch_cuda/`
  - safe CUDA defaults and architecture/runtime metadata
  - explicit reference-family handling for A100/SM80, Ada RTX 40xx, Ada L40S-class, Hopper/SM90, RTX 50xx-class consumer Blackwell, B200-class Blackwell, a GB10/DGX Spark carve-out, and an anticipated Vera Rubin slot
- `tools/calibrate_eval_policy.py`
  - `train-grid`
  - `eval-batch`
  - `eval-rungs`
  - `telemetry-summary`
- `autoresearch_mlx/eval_policy.py`
  - checked-in preset x hardware eval tradeoff tables
- `autoresearch_mlx/eval_telemetry.py`
  - passive evidence from ordinary runs
- `autoresearch_mlx/train.py`
  - conservative runtime selector that exposes calibration status, confidence, freshness, and coverage

The missing layer used to be orchestration and reporting. It is now the engine-specific depth of implementation. MLX currently exercises the whole stack; CUDA is already beyond the original upstream path and lives on the same boundary with shared entrypoints, reporting, and architecture-aware runtime policy, even though it still trails MLX in calibration depth; ROCm and ANE should follow the same pattern rather than introducing new top-level orchestration.

The same principle now also applies to kernel work. `kernel-lab.py` and `autoresearch_lab/` are the parallel boundary for backend-specific kernel experimentation, so future Triton/CUDA, ROCm, or ANE kernel labs can reuse one outer workflow without being confused with the platform-calibration path itself.

The important point is that this boundary is not just for the calibration command itself. It is meant to be the shared contract for the important training-stack features that future engines need to plug into:

- hardware fingerprinting
- train probes and local search
- checkpoint minting
- eval calibration
- runtime capability reporting
- promotion-ready output for new defaults

## Bring-Up Phases

### 1. Hardware Fingerprint

Detect and record the machine identity used for calibration:

- hardware key
- unified memory size
- GPU core count if available
- relevant MLX / Python / macOS version markers

The report should say whether this is:

- an exact match to an existing checked-in hardware row
- a near neighbor of a known row
- a completely new platform

### 2. Coarse Operating Envelope

Run a bounded training sweep across shipped preset families to find the broad working region.

This phase should answer:

- what clearly fits and trains comfortably
- what clearly runs but is too slow or memory-heavy to recommend
- what fails or is obviously out of bounds

The important output is not just the fastest point. It is the zone structure.

### 3. Candidate Family Selection

Inside the viable region, the tool now runs comparable short training probes with eval enabled and ranks the feasible preset families with a frontier-based selector plus pressure-aware memory shaping.

The current selector measures:

- short-run validation quality
- steady-state throughput
- estimated eval overhead
- then keeps the non-dominated candidates on the primary Pareto front
- then chooses the candidate closest to the ideal point in that measured space
- then applies a pressure-aware memory cost:
  - below `50%` of total unified memory, memory only contributes a small tie-break cost
  - above `50%`, memory pressure increasingly shapes the selection
  - near capacity, memory pressure becomes a real penalty
- then uses telemetry maturity only as a tie-break and confidence signal

The result is still a policy choice, but it is no longer a hand-tuned weighted sum hidden in code, and it now treats memory more like a comfort and pressure signal than a co-equal objective at all usage levels.

### 4. Starting-Point Search

Inside the viable region, run constrained operating-point sweeps to identify a practical default for the new machine.

This is where the existing `train-grid` work should be reused:

- `device_batch_size`
- `total_batch_size`
- `seq_len`
- `window_pattern`

This stage should stop well short of a full architecture search. It is a hardware-fit search on top of shipped preset families.

The intended outcome is not only "the best point we measured." It is "the default starting point we would hand to a new autoresearch loop on this hardware unless the user asks for something else."

### 5. Eval Calibration

Once a candidate starting point exists, calibrate its eval behavior:

- fast batch for each canonical eval sequence length
- cheap / reference / full rung timings
- error against a full upstream-style baseline
- overhead at representative train budgets such as `5m`, `30m`, and `8h`

This is where the current eval-policy machinery plugs in.

### 6. Reference Comparison

The report should compare the new hardware to two checked-in reference stories:

- `M5 reference`
  - the current local Apple Silicon baseline
- `upstream-style reference`
  - the upstream/H100-shaped operating point, used as a semantic reference even when it is not a practical local default

We should say "upstream-style reference" unless we later add real measured H100 rows.

### 7. Final Report

The final output should include both Markdown and JSON.

The Markdown report should read like a bring-up memo. The JSON should be structured enough to support future automation and dashboards.

## Bounding Zones

The report should classify results into four zones.

### Lower Bound

Preset / operating-point choices that run comfortably but are clearly dominated for quality or scaling value. These are useful for smoke and debugging, not as the main starting point.

### Recommended Zone

The best local research posture on the machine:

- strong throughput for its class
- sane memory headroom
- acceptable eval overhead
- enough headroom to support longer autonomous experimentation

### Upper Bound

Shapes that still run, but are too slow, too memory-tight, or too operationally fragile to recommend as the default.

### Reference Zone

The upstream-style point used for comparison, even when it is not a practical daily driver on the new hardware.

## What The Report Should Say

At minimum, the one-button report should answer:

- Which preset family is the recommended starting point?
- What should become the new default for the autoresearch stage on this hardware?
- What are the lower and upper bounding zones?
- How does the recommended point compare to the M5 reference?
- How does it compare to the upstream-style reference?
- What eval rung policy should be used by default on this hardware?
- Which parts of that policy are well-supported by evidence, and which are still thin?

The report should treat "recommended starting point" and "candidate new default" as the same thing unless it has a concrete reason to separate them.

## Confidence And Drift

The bring-up tool should not just emit a starting point. It should also say how trustworthy the recommendation is.

That means surfacing:

- exact vs unmatched hardware identity
- how much of the recommendation comes from fresh measured data
- where it is borrowing from nearby references
- what still needs broader validation

This is the same semantic-drift problem the runtime selector now handles, but at the platform level instead of just the eval-rung level. The signatures are useful guardrails. They are not meant to be the whole revalidation policy.

That trust question is no longer only about hardware identity. It is also about code identity:

- does the current eval implementation still match the measured eval calibration?
- does the current runtime/model shape still match the measured platform default?

Those are now tracked as separate signatures.

## Expected Artifacts

The first implementation should write:

- a Markdown report under `results/analysis/`
- a JSON artifact with all raw phase outputs
- a machine-readable candidate default block that can later be promoted into checked-in preset / hardware defaults
- optional candidate calibration rows for manual review

It should also stamp those artifacts with:

- eval-semantics signature
- runtime-shape signature

It should not silently rewrite checked-in policy files on the first pass.

## Current Implementation

`calibrate.py` is the public bring-up entrypoint, with `tools/calibrate_platform.py` as the implementation module underneath. It is an orchestrator, not a separate measurement engine. It reuses the existing training and eval calibration machinery and writes a stable bring-up bundle under `results/analysis/` or the caller's chosen output directory.

It now has two stage-aware operating modes:

- `--mode fast`
  - a quicker first usable default
  - smaller local search expansion
  - `cheap,reference` eval-rung calibration
- `--mode full`
  - the broader bring-up path
  - expanded local search defaults
  - `cheap,reference,full` eval-rung calibration

It also treats the output directory as a resumable workspace. Each major phase writes a stable JSON artifact keyed by its inputs. A rerun with the same output directory reuses finished phases unless `--force` is supplied.

The current phase order is:

1. fingerprint the hardware
2. run a coarse preset envelope using short benchmark-skip-eval probes
3. run short comparable training probes across feasible non-reference presets
4. choose a candidate preset family using the measured frontier distance plus memory-pressure shaping
5. run a local operating-point search inside that family
6. mint a checkpoint at the chosen point
7. run final eval-rung calibration on that checkpoint
8. emit a Markdown report, JSON artifact, logs, a machine-readable candidate default block, and a promotion bundle with default/calibration artifacts

The current report now includes:

- measured `lower` / `recommended` / `upper` / `reference` zones
- zone reasons
- the candidate family ranking table with selection components, memory-fraction bands, and pressure-aware distance
- the tuned default block
- the promotion bundle paths and readiness state
- the current eval/runtime signatures
- selection confidence and supporting telemetry
- M5 train-side and eval-side reference comparisons
- upstream-style reference comparison

## Known Limitations

The current implementation is intentionally conservative.

- The coarse envelope still uses shipped preset families as the outer search shell.
- The local operating-point search now expands `seq_len` and `window_pattern` by mode, but it is still a constrained calibration grid rather than a full preset search.
- The report compares to the checked-in M5 eval reference and the local upstream-style preset behavior, but it does not yet emit a checked-in policy row automatically.
- The tool now emits promotion-ready artifacts by default, but a `fast`-mode run will usually leave the eval-calibration artifact marked incomplete until a `full` audit adds the missing `full` rung.
- The strongest validation story still comes from letting the tool run a full rung audit; quick smoke invocations should be treated as pipeline checks, not as final platform calibration.
- The selector is more defensible than the earlier weighted score, but it is still only as good as the measured candidate set the bring-up run explores.
