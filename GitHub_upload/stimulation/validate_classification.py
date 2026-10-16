"""Validate V4 Patton classification files against saved dynamics data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import (DATASET_GROUPS, DEFAULT_DATA_DIR, ROOT,
                     classification_file, cohort_subjects, dynamics_file,
                     load_group, load_stimulation_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=DATASET_GROUPS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results" / "stimulation")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config" / "stimulation.json")
    parser.add_argument("--subjects")
    args = parser.parse_args()

    raw, _ = load_stimulation_config(args.config)
    connectivity, _ = load_group(
        args.data_dir, args.group, raw["delay_speed_divisor"])
    subjects = cohort_subjects(args.group, connectivity.shape[2], args.subjects)
    thresholds = np.asarray(raw["robustness_thresholds"], dtype=float)
    node_count = connectivity.shape[0]
    errors: list[str] = []
    for subject in subjects:
        path = classification_file(args.output_root, args.group, subject)
        dynamics_path = dynamics_file(args.output_root, args.group, subject)
        prefix = f"{args.group} subject {subject}"
        if not path.exists():
            errors.append(f"{prefix}: missing classification file")
            continue
        with np.load(dynamics_path, allow_pickle=False) as dynamics, \
                np.load(path, allow_pickle=False) as result:
            required_shapes = {
                "state_flags": (len(thresholds), node_count),
                "partitions": (len(thresholds), node_count, 13),
                "community_count": (len(thresholds), node_count),
                "singleton_count": (len(thresholds), node_count),
                "entries_above": (len(thresholds), node_count),
                "state_proportions": (len(thresholds), 3),
                "eligible_stimulus_mask": (node_count,),
            }
            for key, shape in required_shapes.items():
                if key not in result or result[key].shape != shape:
                    errors.append(f"{prefix}: {key} shape mismatch")
            if not np.array_equal(result["thresholds"], thresholds):
                errors.append(f"{prefix}: threshold mismatch")
            if str(result["classification_rule"]) != \
                    "author_code_unique_community_count":
                errors.append(f"{prefix}: classification rule mismatch")
            if not np.isclose(float(result["louvain_gamma"]), 0.7):
                errors.append(f"{prefix}: gamma mismatch")
            if not np.isclose(float(result["source_statistics_start_ms"]),
                              float(dynamics["statistics_start_ms"])):
                errors.append(f"{prefix}: statistics window mismatch")
            eligible = np.asarray(result["eligible_stimulus_mask"], bool)
            expected_eligible = np.asarray(dynamics["degree"]) > 0
            if not np.array_equal(eligible, expected_eligible):
                errors.append(f"{prefix}: eligible mask mismatch")
            states = np.asarray(result["state_flags"])
            if np.any(~np.isin(states[:, eligible], (0, 1, 2))):
                errors.append(f"{prefix}: invalid eligible state code")
            if np.any(states[:, ~eligible] != 255):
                errors.append(f"{prefix}: isolated stimuli not excluded")
            expected_proportions = np.stack([
                np.mean(states[:, eligible] == state, axis=1)
                for state in (0, 1, 2)], axis=1)
            if not np.allclose(result["state_proportions"], expected_proportions):
                errors.append(f"{prefix}: state proportions mismatch")
            if not np.allclose(np.sum(result["state_proportions"], axis=1), 1.0):
                errors.append(f"{prefix}: state proportions do not sum to one")
    if errors:
        raise RuntimeError("Classification validation failed:\n" + "\n".join(errors))
    print(f"PASS: {args.group}, classification subjects={len(subjects)}, "
          f"thresholds={len(thresholds)}, nodes={node_count}")


if __name__ == "__main__":
    main()
