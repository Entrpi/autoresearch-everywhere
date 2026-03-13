from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path

from autoresearch_mlx.calibration_signature import current_eval_semantics_signature, current_runtime_shape_signature
from autoresearch_mlx.constants import MAX_SEQ_LEN
from autoresearch_mlx.eval_policy import detect_current_hardware_key
from autoresearch_mlx.train import PRESETS
from autoresearch_platform.summary import parse_summary
from tools.calibrate_eval_policy import default_eval_batch_size, run_eval_rungs

from .engines import EngineCapabilities, EnginePreset, HardwareFingerprint, ProbeResult


REPO_ROOT = Path(__file__).resolve().parents[1]

PRESET_ORDER = ("m5-tiny", "m5-small", "m5-balanced", "m5-large", "m5-xlarge", "upstream")
DEFAULT_PLATFORM_PRESETS = ("m5-tiny", "m5-small", "m5-balanced", "m5-large", "m5-xlarge")


class MLXEngine:
    name = "mlx"
    backend_family = "mlx"
    reference_preset = "upstream"
    capabilities = EngineCapabilities(
        supports_platform_bringup=True,
        supports_local_search=True,
        supports_eval_calibration=True,
        supports_checkpoint_mint=True,
        supports_runtime_eval_policy=True,
        mutable_axes=("seq_len", "window_pattern", "device_batch_size", "total_batch_size"),
    )

    def preset_catalog(self) -> dict[str, EnginePreset]:
        return {
            key: EnginePreset(
                key=key,
                description=value.description,
                seq_len=value.seq_len,
                depth=value.depth,
                window_pattern=value.window_pattern,
                device_batch_size=value.device_batch_size,
                total_batch_size=value.total_batch_size,
                canonical_eval_seq_len=value.canonical_eval_seq_len,
                canonical_eval_tokens=value.canonical_eval_tokens,
                canonical_eval_batch_size=value.canonical_eval_batch_size,
                tags=("reference",) if key == "upstream" else (),
            )
            for key, value in PRESETS.items()
        }

    def preset_order(self) -> tuple[str, ...]:
        return tuple(key for key in PRESET_ORDER if key in PRESETS)

    def default_platform_presets(self) -> tuple[str, ...]:
        return tuple(key for key in DEFAULT_PLATFORM_PRESETS if key in PRESETS)

    def detect_hardware_fingerprint(self) -> HardwareFingerprint:
        hardware_key = detect_current_hardware_key()
        os_version = None
        runtime_version = None
        memsize = None
        chip_model = None
        gpu_cores = None

        if sys.platform == "darwin":
            try:
                os_version = subprocess.run(
                    ["sw_vers", "-productVersion"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except Exception:
                os_version = None
            try:
                memsize = int(
                    subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                )
            except Exception:
                memsize = None
            try:
                displays = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                import re

                chip_match = re.search(r"Chipset Model:\s*(Apple\s+[A-Za-z0-9]+)", displays)
                if chip_match is None:
                    chip_match = re.search(r"^(Apple\s+[A-Za-z0-9]+):\s*$", displays, re.MULTILINE)
                gpu_match = re.search(r"Total Number of Cores:\s*(\d+)", displays)
                if chip_match is not None:
                    chip_model = chip_match.group(1)
                if gpu_match is not None:
                    gpu_cores = int(gpu_match.group(1))
            except Exception:
                pass

        try:
            import importlib.metadata

            runtime_version = importlib.metadata.version("mlx")
        except Exception:
            runtime_version = None

        memory_gb = memsize / (1024**3) if memsize is not None else None
        architecture = None
        if chip_model is not None:
            architecture = chip_model.lower().replace(" ", "-")

        return HardwareFingerprint(
            engine=self.name,
            backend_family=self.backend_family,
            hardware_key=hardware_key,
            platform=platform.platform(),
            machine=platform.machine(),
            processor=platform.processor(),
            python_version=platform.python_version(),
            os_version=os_version,
            runtime_version=runtime_version,
            accelerator_vendor="Apple",
            accelerator_model=chip_model,
            accelerator_architecture=architecture,
            accelerator_cores=gpu_cores,
            memory_bytes=memsize,
            memory_gb=memory_gb,
            attention_backend="mlx-fast-sdpa",
        )

    def run_train_probe(
        self,
        *,
        preset: str,
        time_budget: float,
        logs_dir: Path,
        stage: str,
        benchmark_skip_eval: bool,
        checkpoint_path: Path | None = None,
        seq_len: int | None = None,
        window_pattern: str | None = None,
        device_batch_size: int | None = None,
        total_batch_size: int | None = None,
        curve_eval_seconds: tuple[float, ...] | None = None,
        eval_seq_len: int | None = None,
        eval_tokens: int | None = None,
        eval_batch_size: int | None = None,
        no_checkpoint: bool = True,
    ) -> ProbeResult:
        del curve_eval_seconds, eval_seq_len, eval_tokens, eval_batch_size
        preset_config = PRESETS[preset]
        resolved_seq_len = seq_len if seq_len is not None else preset_config.seq_len
        resolved_window = window_pattern if window_pattern is not None else preset_config.window_pattern
        resolved_device_batch = device_batch_size if device_batch_size is not None else preset_config.device_batch_size
        resolved_total_batch = total_batch_size if total_batch_size is not None else preset_config.total_batch_size
        resolved_depth = preset_config.depth
        grad_accum_steps = self._infer_grad_accum(resolved_seq_len, resolved_device_batch, resolved_total_batch)
        if grad_accum_steps is None:
            raise ValueError(
                f"Invalid total batch {resolved_total_batch} for seq_len={resolved_seq_len}, "
                f"device_batch_size={resolved_device_batch}"
            )

        cmd = [
            sys.executable,
            "-m",
            "autoresearch_mlx.train",
            "--preset",
            preset,
            "--time-budget",
            str(time_budget),
            "--seq-len",
            str(resolved_seq_len),
            "--window-pattern",
            resolved_window,
            "--device-batch-size",
            str(resolved_device_batch),
            "--total-batch-size",
            str(resolved_total_batch),
        ]
        if benchmark_skip_eval:
            cmd.append("--benchmark-skip-eval")
        if no_checkpoint:
            cmd.append("--no-checkpoint")
        if checkpoint_path is not None:
            cmd.extend(["--checkpoint-path", str(checkpoint_path)])

        label = self._command_label(
            [
                stage,
                preset,
                f"seq{resolved_seq_len}",
                f"db{resolved_device_batch}",
                f"tb{resolved_total_batch}",
                resolved_window,
            ]
        )
        completed, wall_seconds, stdout_path, stderr_path = self._run_command(cmd, logs_dir=logs_dir, label=label)
        summary = parse_summary(completed.stdout) if completed.returncode == 0 else {}
        error_tail = None
        if completed.returncode != 0:
            error_tail = "\n".join(completed.stderr.splitlines()[-12:])

        return ProbeResult(
            preset=preset,
            stage=stage,
            seq_len=resolved_seq_len,
            depth=resolved_depth,
            window_pattern=resolved_window,
            device_batch_size=resolved_device_batch,
            total_batch_size=resolved_total_batch,
            grad_accum_steps=grad_accum_steps,
            status="ok" if completed.returncode == 0 else "error",
            returncode=completed.returncode,
            wall_seconds=wall_seconds,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            val_bpb=self._get_float(summary, "val_bpb"),
            proxy_val_bpb=self._get_float(summary, "proxy_val_bpb"),
            steady_state_tok_per_sec=self._get_float(summary, "steady_state_tok_per_sec"),
            peak_vram_mb=self._get_float(summary, "peak_vram_mb"),
            training_seconds=self._get_float(summary, "training_seconds"),
            total_seconds=self._get_float(summary, "total_seconds"),
            eval_percent=self._get_float(summary, "eval_percent"),
            optimizer_percent=self._get_float(summary, "optimizer_percent"),
            accum_percent=self._get_float(summary, "accum_percent"),
            control_overhead_percent=self._control_overhead_percent(summary),
            canonical_rung=self._get_str(summary, "canonical_rung"),
            canonical_seq_len=self._get_int(summary, "canonical_eval_seq_len") or self._get_int(summary, "canonical_seq_len"),
            canonical_tokens=self._get_int(summary, "canonical_eval_tokens") or self._get_int(summary, "canonical_tokens"),
            canonical_batch=self._get_int(summary, "canonical_eval_batch_size") or self._get_int(summary, "canonical_batch"),
            canonical_slices=self._get_int(summary, "canonical_eval_slices") or self._get_int(summary, "canonical_slices"),
            eval_calibration_status=self._get_str(summary, "eval_calibration_status"),
            eval_calibration_effective_confidence=self._get_str(summary, "eval_calibration_effective_confidence"),
            eval_calibration_freshness=self._get_str(summary, "eval_calibration_freshness"),
            eval_calibration_limited_by=self._get_str(summary, "eval_calibration_limited_by"),
            error_tail=error_tail,
        )

    def default_local_seq_lens(self, preset: str, *, mode: str) -> list[int]:
        preset_config = PRESETS[preset]
        if mode == "fast":
            return [preset_config.seq_len]
        candidates = [preset_config.seq_len]
        doubled = min(MAX_SEQ_LEN, preset_config.seq_len * 2)
        if doubled != preset_config.seq_len:
            candidates.append(doubled)
        return sorted(set(candidates))

    def default_local_window_patterns(self, preset: str, *, mode: str) -> list[str]:
        preset_config = PRESETS[preset]
        patterns = [preset_config.window_pattern]
        if mode == "full" and preset_config.window_pattern == "L" and preset_config.seq_len >= 1024:
            patterns.append("SSSL")
        return list(dict.fromkeys(patterns))

    def local_batch_candidates(self, preset: str, *, seq_len: int) -> list[tuple[int, int]]:
        preset_config = PRESETS[preset]
        base_device_batch = preset_config.device_batch_size
        base_tokens = base_device_batch * preset_config.seq_len
        base_grad_accum = max(1, preset_config.total_batch_size // base_tokens)

        device_batches = sorted({max(1, base_device_batch // 2), base_device_batch, base_device_batch * 2})
        grad_accum_candidates = sorted({max(1, base_grad_accum // 2), base_grad_accum, base_grad_accum * 2})

        combos: list[tuple[int, int]] = []
        for device_batch in device_batches:
            for grad_accum in grad_accum_candidates:
                total_batch = device_batch * seq_len * grad_accum
                combos.append((device_batch, total_batch))
        return sorted(set(combos))

    def batch_profile_candidates(self, preset: str, *, seq_len: int) -> list[tuple[int, int]]:
        return self.local_batch_candidates(preset, seq_len=seq_len)

    def batch_profile_refinement_candidates(
        self,
        preset: str,
        *,
        seq_len: int,
        coarse_winner: tuple[int, int],
    ) -> list[tuple[int, int]]:
        winner_device_batch, winner_total_batch = coarse_winner
        tokens_per_fwdbwd = seq_len * winner_device_batch
        if tokens_per_fwdbwd <= 0:
            return []
        winner_grad_accum = max(1, winner_total_batch // tokens_per_fwdbwd)
        device_batches = {
            max(1, winner_device_batch // 2),
            winner_device_batch,
            winner_device_batch * 2,
        }
        grad_accum_candidates = {
            max(1, winner_grad_accum - 2),
            max(1, winner_grad_accum - 1),
            winner_grad_accum,
            winner_grad_accum + 1,
            winner_grad_accum + 2,
            max(1, round(winner_grad_accum * 1.5)),
            winner_grad_accum * 2,
        }
        combos: list[tuple[int, int]] = []
        for device_batch in sorted(device_batches):
            tokens_per_fwdbwd = seq_len * device_batch
            if tokens_per_fwdbwd <= 0:
                continue
            for grad_accum in sorted(acc for acc in grad_accum_candidates if acc <= 16):
                combos.append((device_batch, tokens_per_fwdbwd * grad_accum))
        return sorted(set(combos))

    def calibration_signatures(self) -> dict[str, str | None]:
        return {
            "eval_semantics_signature": current_eval_semantics_signature(),
            "runtime_shape_signature": current_runtime_shape_signature(),
        }

    def run_eval_calibration(
        self,
        *,
        preset: str,
        checkpoint_dir: Path,
        hardware_key: str,
        rungs: list[str],
        budget_seconds: list[float],
        markdown_path: Path | None = None,
    ) -> dict:
        preset_config = PRESETS[preset]
        args = type(
            "EvalCalibrationArgs",
            (),
            {
                "preset": preset,
                "checkpoint": str(checkpoint_dir),
                "seq_len": preset_config.canonical_eval_seq_len,
                "batch_size": default_eval_batch_size(preset_config.canonical_eval_seq_len),
                "rungs": rungs,
                "budget_seconds": budget_seconds,
                "hardware_key": hardware_key,
                "no_prepacked_cache": False,
                "markdown_out": str(markdown_path) if markdown_path is not None else None,
            },
        )()
        return run_eval_rungs(args)

    @staticmethod
    def _command_label(parts: list[str]) -> str:
        return "_".join(
            part.replace("--", "").replace("/", "_").replace("=", "_").replace(",", "_")
            for part in parts
        )

    @staticmethod
    def _run_command(cmd: list[str], *, logs_dir: Path, label: str):
        stdout_path = logs_dir / f"{label}.stdout.log"
        stderr_path = logs_dir / f"{label}.stderr.log"
        started = time.perf_counter()
        completed = subprocess.run(cmd, capture_output=True, text=True)
        wall_seconds = time.perf_counter() - started
        stdout_path.write_text(completed.stdout)
        stderr_path.write_text(completed.stderr)
        return completed, wall_seconds, stdout_path, stderr_path

    @staticmethod
    def _infer_grad_accum(seq_len: int, device_batch_size: int, total_batch_size: int) -> int | None:
        tokens_per_fwdbwd = seq_len * device_batch_size
        if tokens_per_fwdbwd <= 0 or total_batch_size % tokens_per_fwdbwd != 0:
            return None
        return total_batch_size // tokens_per_fwdbwd

    @staticmethod
    def _get_float(summary: dict, key: str) -> float | None:
        value = summary.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_int(summary: dict, key: str) -> int | None:
        value = summary.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_str(summary: dict, key: str) -> str | None:
        value = summary.get(key)
        if value is None:
            return None
        text = str(value)
        if text == "None":
            return None
        return text

    @classmethod
    def _control_overhead_percent(cls, summary: dict) -> float | None:
        optimizer = cls._get_float(summary, "optimizer_percent")
        accum = cls._get_float(summary, "accum_percent")
        if optimizer is None and accum is None:
            return None
        return float(optimizer or 0.0) + float(accum or 0.0)
