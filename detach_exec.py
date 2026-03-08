#!/usr/bin/env python3
"""Detach a command from the current session and redirect output to a log file."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit("usage: detach_exec.py <pid-file> <log-file> <command> [args...]")

    cwd = Path.cwd()
    pid_file = Path(sys.argv[1]).resolve()
    log_file = Path(sys.argv[2]).resolve()
    command: list[str] = []
    for raw_arg in sys.argv[3:]:
        candidate = Path(raw_arg)
        if candidate.is_absolute():
            command.append(str(candidate))
        elif (cwd / candidate).exists():
            command.append(str((cwd / candidate).resolve()))
        else:
            command.append(raw_arg)

    first_pid = os.fork()
    if first_pid > 0:
        _, status = os.waitpid(first_pid, 0)
        raise SystemExit(os.waitstatus_to_exitcode(status))

    os.setsid()

    second_pid = os.fork()
    if second_pid > 0:
        os._exit(0)

    os.chdir("/")
    os.umask(0)

    with open(os.devnull, "rb", buffering=0) as null_in:
        os.dup2(null_in.fileno(), 0)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("ab", buffering=0) as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)

    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{os.getpid()}\n")

    os.execv(command[0], command)


if __name__ == "__main__":
    main()
