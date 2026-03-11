#!/usr/bin/env python3
from __future__ import annotations

from autoresearch_platform.entrypoints import dispatch_entrypoint


if __name__ == "__main__":
    dispatch_entrypoint("train")
