"""YAML configuration loader."""
from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict

import yaml


def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output", type=str, default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser
