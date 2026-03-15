from __future__ import annotations

import threading
from typing import Callable, Generic, TypeVar


SnapshotT = TypeVar("SnapshotT")


class AsyncCheckpointWriter(Generic[SnapshotT]):
    def __init__(
        self,
        *,
        write_snapshot: Callable[[SnapshotT], float],
        error_prefix: str = "Asynchronous checkpoint writer failed",
    ) -> None:
        self._write_snapshot = write_snapshot
        self._error_prefix = error_prefix
        self._thread: threading.Thread | None = None
        self._error: RuntimeError | None = None
        self._lock = threading.Lock()
        self.completed_write_seconds = 0.0
        self.completed_count = 0

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error

    def _write_in_background(self, snapshot: SnapshotT) -> None:
        try:
            elapsed = self._write_snapshot(snapshot)
            with self._lock:
                self.completed_write_seconds += elapsed
                self.completed_count += 1
        except BaseException as exc:  # pragma: no cover - surfaced on join/poll
            with self._lock:
                self._error = RuntimeError(f"{self._error_prefix}: {exc!r}")

    def _collect_if_ready(self, *, block: bool) -> None:
        thread = self._thread
        if thread is None:
            self._raise_if_error()
            return
        if block:
            thread.join()
        elif thread.is_alive():
            return
        else:
            thread.join()
        self._thread = None
        self._raise_if_error()

    def submit(self, snapshot: SnapshotT) -> None:
        self._raise_if_error()
        self._collect_if_ready(block=False)
        if self._thread is not None:
            raise RuntimeError("Asynchronous checkpoint writer is still writing the previous snapshot.")
        thread = threading.Thread(
            target=self._write_in_background,
            args=(snapshot,),
            daemon=True,
        )
        thread.start()
        self._thread = thread

    def wait_until_idle(self) -> None:
        self._collect_if_ready(block=True)
        self._raise_if_error()

    def close(self) -> None:
        self.wait_until_idle()
        self._raise_if_error()

    def check_health(self) -> None:
        self._collect_if_ready(block=False)
        self._raise_if_error()
