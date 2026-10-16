"""Configuration, cohort, critical-point, and atomic I/O helpers."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from .model import StimulationConfig


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_CRITICAL_ROOT = ROOT / "critical_point"
DATASET_GROUPS = ("Controls", "Pats_t1", "Pats_t2")
PAIRED_PATIENT_COUNT = 34


def load_stimulation_config(path: Path) -> tuple[dict, StimulationConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if tuple(raw["groups"]) != DATASET_GROUPS:
        raise ValueError("stimulation groups do not match expected groups")
    if not (raw["analysis_start_ms"] <= raw["statistics_start_ms"]
            < raw["analysis_end_ms"]):
        raise ValueError("statistics_start_ms must lie inside saved window")
    if not np.isclose(raw["analysis_start_ms"], raw["transient_ms"]):
        raise ValueError("analysis_start_ms must equal transient_ms")
    if not np.isclose(raw["analysis_end_ms"], raw["total_ms"]):
        raise ValueError("analysis_end_ms must equal total_ms")
    if not np.isclose(raw["noise_strength"], 1.0e-5):
        raise ValueError("noise_strength must remain 1e-5")
    model = StimulationConfig(
        dt_ms=raw["dt_ms"], total_ms=raw["total_ms"],
        transient_ms=raw["transient_ms"],
        stimulus_start_ms=raw["stimulus_start_ms"],
        stimulus_end_ms=raw["stimulus_end_ms"],
        stimulus_intensity=raw["stimulus_intensity"],
        noise_strength=raw["noise_strength"],
        initial_e=raw["initial_e"], initial_i=raw["initial_i"])
    return raw, model


def load_group(data_dir: Path, group: str, delay_divisor: float):
    path = data_dir / f"Dataset_{group}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    dataset = np.load(path, allow_pickle=False, mmap_mode="r")
    connectivity = np.asarray(dataset["Connectivity"])
    distance = np.asarray(dataset["distance"], dtype=np.float64)
    if connectivity.ndim != 3 or connectivity.shape[:2] != distance.shape:
        raise ValueError(f"invalid Connectivity/distance shapes in {path}")
    return connectivity, distance / float(delay_divisor)


def validate_adjacency(weighted: np.ndarray, subject: int) -> np.ndarray:
    adjacency = np.asarray(weighted, dtype=np.float64).copy()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"subject {subject}: adjacency must be square")
    if not np.all(np.isfinite(adjacency)) or np.any(adjacency < 0):
        raise ValueError(f"subject {subject}: invalid weights")
    if not np.allclose(np.diag(adjacency), 0.0, atol=1.0e-14):
        raise ValueError(f"subject {subject}: diagonal weights must be zero")
    row_sums = adjacency.sum(axis=1)
    nonisolated = row_sums > np.finfo(float).tiny
    if np.any(nonisolated) and not np.allclose(
            row_sums[nonisolated], 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"subject {subject}: rows are not sum-normalized")
    return adjacency


def cohort_subjects(group: str, available: int,
                    requested: str | None) -> list[int]:
    limit = available if group == "Controls" else min(PAIRED_PATIENT_COUNT, available)
    allowed = set(range(limit))
    if requested is None:
        return list(range(limit))
    selected = sorted({int(x) for x in requested.split(",") if x.strip()})
    if not selected or not set(selected).issubset(allowed):
        raise ValueError(
            f"{group} subjects must be 0-based indices in [0, {limit - 1}]")
    return selected


def read_critical_table(path: Path, expected_subjects: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_subjects:
        raise ValueError(f"{path}: expected {expected_subjects}, found {len(rows)}")
    by_subject = {int(row["subject_index_0based"]): row for row in rows}
    if sorted(by_subject) != list(range(expected_subjects)):
        raise ValueError(f"{path}: incomplete subject indices")
    return [by_subject[index] for index in range(expected_subjects)]


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def dynamics_file(output_root: Path, group: str, subject: int) -> Path:
    return output_root / "dynamics" / group / f"subject_{subject:03d}.npz"


def classification_file(output_root: Path, group: str, subject: int) -> Path:
    return output_root / "classification" / group / f"subject_{subject:03d}.npz"
