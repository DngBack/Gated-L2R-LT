"""Lightweight logging utilities for experiment tracking."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class MetricLogger:
    """Console + file logger for training metrics."""

    output_dir: str
    filename: str = "log.jsonl"
    _file: Optional[Any] = field(init=False, default=None)

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, self.filename)
        self._file = open(path, "a", encoding="utf-8")

    def log(self, step: int, **metrics: float) -> None:
        payload = {"step": step, "time": datetime.utcnow().isoformat(), **metrics}
        line = json.dumps(payload)
        print(line)
        assert self._file is not None
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:  # pragma: no cover - trivial destructor
        if self._file is not None:
            self._file.close()
            self._file = None


class Tee:
    """Redirect stderr/stdout to a log file while printing."""

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self  # type: ignore
        sys.stderr = self  # type: ignore

    def write(self, data: str) -> None:  # pragma: no cover - passthrough
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self) -> None:  # pragma: no cover - passthrough
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:  # pragma: no cover - trivial destructor
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._file.close()
