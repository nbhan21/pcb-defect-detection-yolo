"""Shared paths and D-drive work-file setup for the manual Python stages."""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    training: Path
    evaluation: Path
    predictions: Path
    deployment: Path


def experiment_paths(project_root: Path, name: str) -> ExperimentPaths:
    """Keep one learner-run's outputs together without touching the baseline."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError("Nama eksperimen harus huruf kecil, angka, '-' atau '_'.")
    root = Path(project_root) / "experiments" / name
    return ExperimentPaths(
        root=root,
        training=root / "training",
        evaluation=root / "evaluation",
        predictions=root / "predictions",
        deployment=root / "deployment",
    )


def configure_runtime(project_root: Path = PROJECT_ROOT) -> None:
    """Point project-managed temporary, cache and tool files into this project."""
    local = Path(project_root) / ".local"
    directories = {
        "TEMP": local / "tmp",
        "TMP": local / "tmp",
        "TMPDIR": local / "tmp",
        "UV_CACHE_DIR": local / "cache" / "uv",
        "PIP_CACHE_DIR": local / "cache" / "pip",
        "TORCH_HOME": local / "cache" / "torch",
        "XDG_CACHE_HOME": local / "cache" / "xdg",
        "XDG_STATE_HOME": local / "state",
        "YOLO_CONFIG_DIR": local / "config" / "ultralytics",
        "MPLCONFIGDIR": local / "matplotlib",
    }
    for variable, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    sys.dont_write_bytecode = True
