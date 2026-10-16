"""Classify saved V4 system_pair_rho matrices at all configured thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import (DATASET_GROUPS, DEFAULT_DATA_DIR, ROOT, atomic_savez,
                     classification_file, cohort_subjects, dynamics_file,
                     load_group, load_stimulation_config)
from .patton import classify_matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=DATASET_GROUPS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results" / "stimulation")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config" / "stimulation.json")
    parser.add_argument("--subjects")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw, _ = load_stimulation_config(args.config)
    connectivity, _ = load_group(
        args.data_dir, args.group, raw["delay_speed_divisor"])
    subjects = cohort_subjects(args.group, connectivity.shape[2], args.subjects)
    thresholds = np.asarray(raw["robustness_thresholds"], dtype=float)
    if not np.any(np.isclose(thresholds, raw["primary_threshold"])):
        raise ValueError("primary_threshold is absent from robustness_thresholds")
    gamma = float(raw["louvain_gamma"])
    if not np.isclose(gamma, 0.7):
        raise ValueError("Patton author-code objective requires gamma=0.7")

    for subject in subjects:
        input_path = dynamics_file(args.output_root, args.group, subject)
        output_path = classification_file(args.output_root, args.group, subject)
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if output_path.exists() and not args.force:
            print(f"SKIP group={args.group} subject={subject}: {output_path}")
            continue
        with np.load(input_path, allow_pickle=False) as dynamics:
            completed = np.asarray(dynamics["completed_stimulus_mask"], dtype=bool)
            if not np.all(completed):
                raise ValueError(f"{input_path}: incomplete stimulus run")
            degree = np.asarray(dynamics["degree"])
            if raw.get("exclude_isolated_stimuli_from_proportions", True):
                eligible = degree > 0
            else:
                eligible = np.ones(len(degree), dtype=bool)
            matrices = np.asarray(dynamics["system_pair_rho"], dtype=float)
            statistics_start_ms = float(dynamics["statistics_start_ms"])

        threshold_results = []
        for threshold_index, threshold in enumerate(thresholds):
            threshold_results.append(classify_matrices(
                matrices, eligible, float(threshold), gamma,
                int(raw["louvain_seed"]) + 1000003 * threshold_index))
        atomic_savez(
            output_path,
            group=np.array(args.group),
            subject_index=np.array(subject, dtype=np.int32),
            thresholds=thresholds,
            primary_threshold=np.array(raw["primary_threshold"]),
            eligible_stimulus_mask=eligible,
            state_flags=np.stack([item["state_flags"] for item in threshold_results]),
            partitions=np.stack([item["partitions"] for item in threshold_results]),
            community_count=np.stack(
                [item["community_count"] for item in threshold_results]),
            singleton_count=np.stack(
                [item["singleton_count"] for item in threshold_results]),
            entries_above=np.stack(
                [item["entries_above"] for item in threshold_results]),
            state_proportions=np.stack(
                [item["state_proportions"] for item in threshold_results]),
            state_names=np.array(["coherent", "chimera", "incoherent"]),
            classification_rule=np.array("author_code_unique_community_count"),
            louvain_gamma=np.array(gamma),
            excluded_state_code=np.array(255, dtype=np.uint8),
            source_statistics_start_ms=np.array(statistics_start_ms),
        )
        print(f"SAVED group={args.group} subject={subject}: {output_path}", flush=True)


if __name__ == "__main__":
    main()
