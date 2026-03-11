from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path

from autoresearch_cuda.config import CUDA_PRESETS, resolve_run_preset
from autoresearch_cuda.runtime import detect_cuda_runtime_profile, query_nvidia_driver_version
from tools.calibrate_eval_policy import parse_summary

from .engines import EngineCapabilities, EnginePreset, HardwareFingerprint, ProbeResult


class CUDAEngine:
    name = "cuda"
    backend_family = "cuda"
    reference_preset = "upstream"
    capabilities = EngineCapabilities(
        supports_platform_bringup=True,
        supports_local_search=False,
        supports_eval_calibration=False,
        supports_checkpoint_mint=False,
        supports_runtime_eval_policy=False,
        mutable_axes=("seq_len", "window_pattern", "device_batch_size", "total_batch_size", "depth"),
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
                tags=("reference",),
            )
            for key, value in CUDA_PRESETS.items()
        }

    def preset_order(self) -> tuple[str, ...]:
        return tuple(CUDA_PRESETS.keys())

    def default_platform_presets(self) -> tuple[str, ...]:
        return ("upstream",)

    def detect_hardware_fingerprint(self) -> HardwareFingerprint:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("CUDA engine requires torch to inspect CUDA hardware.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA engine selected but no CUDA device is available.")

        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        capability = torch.cuda.get_device_capability(device_index)
        runtime = detect_cuda_runtime_profile(capability, device_name=props.name)
        total_memory_bytes = int(props.total_memory)
        memory_gb = total_memory_bytes / (1024**3)
        driver_version = query_nvidia_driver_version()
        hardware_key = f"nvidia-{runtime.reference_family.family_key}-{int(round(memory_gb))}gb"

        return HardwareFingerprint(
            engine=self.name,
            backend_family=self.backend_family,
            hardware_key=hardware_key,
            platform=platform.platform(),
            machine=platform.machine(),
            processor=platform.processor(),
            python_version=platform.python_version(),
            runtime_version=getattr(torch.version, "cuda", None),
            driver_version=driver_version,
            accelerator_vendor="NVIDIA",
            accelerator_model=props.name,
            accelerator_architecture=runtime.reference_family.family_key,
            accelerator_compute_capability=f"{capability[0]}.{capability[1]}",
            accelerator_cores=getattr(props, "multi_processor_count", None),
            memory_bytes=total_memory_bytes,
            memory_gb=memory_gb,
            attention_backend=runtime.selected_attention_backend,
            flash_attention_generation=runtime.preferred_flash_attention_generation,
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
        no_checkpoint: bool = True,
    ) -> ProbeResult:
        if checkpoint_path is not None:
            raise RuntimeError("CUDA engine does not support checkpoint minting through calibrate_platform yet.")
        if not no_checkpoint:
            raise RuntimeError("CUDA engine does not support checkpoint minting through calibrate_platform yet.")

        resolved = resolve_run_preset(
            preset,
            time_budget=time_budget,
            seq_len=seq_len,
            window_pattern=window_pattern,
            total_batch_size=total_batch_size,
            depth=None,
            device_batch_size=device_batch_size,
        )
        grad_accum_steps = self._infer_grad_accum(
            resolved.seq_len,
            resolved.device_batch_size,
            resolved.total_batch_size,
        )
        if grad_accum_steps is None:
            raise ValueError(
                f"Invalid total batch {resolved.total_batch_size} for seq_len={resolved.seq_len}, "
                f"device_batch_size={resolved.device_batch_size}"
            )

        cmd = [
            sys.executable,
            "train.py",
            "--preset",
            preset,
            "--time-budget",
            str(resolved.time_budget),
            "--seq-len",
            str(resolved.seq_len),
            "--depth",
            str(resolved.depth),
            "--window-pattern",
            resolved.window_pattern,
            "--device-batch-size",
            str(resolved.device_batch_size),
            "--total-batch-size",
            str(resolved.total_batch_size),
        ]
        if benchmark_skip_eval:
            cmd.append("--benchmark-skip-eval")

        label = self._command_label(
            [
                stage,
                preset,
                f"seq{resolved.seq_len}",
                f"db{resolved.device_batch_size}",
                f"tb{resolved.total_batch_size}",
                resolved.window_pattern,
            ]
        )
        completed, wall_seconds, stdout_path, stderr_path = self._run_command(cmd, logs_dir=logs_dir, label=label)
        summary = parse_summary(completed.stdout) if completed.returncode == 0 else {}
        error_tail = None
        if completed.returncode != 0:
            error_tail = "\n".join(completed.stderr.splitlines()[-12:])

        steady_state_tok_per_sec = self._get_float(summary, "steady_state_tok_per_sec")
        if steady_state_tok_per_sec is None:
            training_seconds = self._get_float(summary, "training_seconds")
            total_tokens_m = self._get_float(summary, "total_tokens_M")
            if training_seconds and total_tokens_m is not None and training_seconds > 0:
                steady_state_tok_per_sec = (total_tokens_m * 1_000_000.0) / training_seconds

        return ProbeResult(
            preset=preset,
            stage=stage,
            seq_len=resolved.seq_len,
            depth=resolved.depth,
            window_pattern=resolved.window_pattern,
            device_batch_size=resolved.device_batch_size,
            total_batch_size=resolved.total_batch_size,
            grad_accum_steps=grad_accum_steps,
            status="ok" if completed.returncode == 0 else "error",
            returncode=completed.returncode,
            wall_seconds=wall_seconds,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            val_bpb=self._get_float(summary, "val_bpb"),
            steady_state_tok_per_sec=steady_state_tok_per_sec,
            peak_vram_mb=self._get_float(summary, "peak_vram_mb"),
            training_seconds=self._get_float(summary, "training_seconds"),
            total_seconds=self._get_float(summary, "total_seconds"),
            error_tail=error_tail,
        )

    def default_local_seq_lens(self, preset: str, *, mode: str) -> list[int]:
        return [CUDA_PRESETS[preset].seq_len]

    def default_local_window_patterns(self, preset: str, *, mode: str) -> list[str]:
        return [CUDA_PRESETS[preset].window_pattern]

    def local_batch_candidates(self, preset: str, *, seq_len: int) -> list[tuple[int, int]]:
        value = CUDA_PRESETS[preset]
        return [(value.device_batch_size, value.total_batch_size)]

    def calibration_signatures(self) -> dict[str, str | None]:
        return {
            "eval_semantics_signature": None,
            "runtime_shape_signature": None,
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
        raise RuntimeError(
            "CUDA engine does not support checkpoint-backed eval calibration through the shared engine boundary yet."
        )

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
