"""Validate V4 synchrony and propagation-subnetwork subject files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import (DATASET_GROUPS, DEFAULT_CRITICAL_ROOT, DEFAULT_DATA_DIR,
                     ROOT, cohort_subjects, dynamics_file, load_group,
                     load_stimulation_config, read_critical_table)


def _validate_cycles(data, errors: list[str], prefix: str) -> None:
    stimuli = np.asarray(data["propagation_cycle_stimulus"], dtype=int)
    lengths = np.asarray(data["propagation_cycle_length"], dtype=int)
    nodes = np.asarray(data["propagation_cycle_nodes"], dtype=int)
    indptr = np.asarray(data["propagation_cycle_indptr"], dtype=int)
    parent = np.asarray(data["propagation_parent"], dtype=int)
    if len(stimuli) != len(lengths) or len(indptr) != len(lengths) + 1:
        errors.append(f"{prefix}: malformed cycle arrays")
        return
    if len(indptr) == 0 or indptr[0] != 0 or indptr[-1] != len(nodes) \
            or np.any(np.diff(indptr) < 0):
        errors.append(f"{prefix}: invalid cycle indptr")
        return
    if not np.array_equal(np.diff(indptr), lengths):
        errors.append(f"{prefix}: cycle lengths disagree with indptr")
    for index, stimulus in enumerate(stimuli):
        cycle = nodes[indptr[index]:indptr[index + 1]]
        if len(cycle) < 2 or len(np.unique(cycle)) != len(cycle):
            errors.append(f"{prefix}: invalid cycle {index}")
            continue
        for edge_index, source in enumerate(cycle):
            target = cycle[(edge_index + 1) % len(cycle)]
            if parent[stimulus, target] != source:
                errors.append(f"{prefix}: cycle {index} edge mismatch")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=DATASET_GROUPS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--critical-root", type=Path,
                        default=DEFAULT_CRITICAL_ROOT)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results" / "stimulation")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config" / "stimulation.json")
    parser.add_argument("--subjects")
    parser.add_argument("--allow-partial-stimuli", action="store_true")
    args = parser.parse_args()
    raw, _ = load_stimulation_config(args.config)
    connectivity, _ = load_group(
        args.data_dir, args.group, raw["delay_speed_divisor"])
    subjects = cohort_subjects(args.group, connectivity.shape[2], args.subjects)
    critical = read_critical_table(
        args.critical_root / "aggregate" / f"critical_points_{args.group}.csv",
        connectivity.shape[2])
    node_count = connectivity.shape[0]
    errors: list[str] = []
    for subject in subjects:
        path = dynamics_file(args.output_root, args.group, subject)
        prefix = f"subject {subject}"
        if not path.exists():
            errors.append(f"{prefix}: missing {path}")
            continue
        with np.load(path) as data:
            completed = np.asarray(data["completed_stimulus_mask"], dtype=bool)
            if completed.shape != (node_count,):
                errors.append(f"{prefix}: completed mask shape mismatch")
                continue
            if not args.allow_partial_stimuli and not np.all(completed):
                errors.append(f"{prefix}: incomplete 324-node run")
            expected_shapes = {
                "system_pair_rho": (13, 13, node_count),
                "global_mean_rho": (node_count,),
                "chimera_index": (node_count,),
                "metastability_index": (node_count,),
                "propagation_responsive_mask": (node_count, node_count),
                "propagation_parent": (node_count, node_count),
                "propagation_drive_score": (node_count, node_count),
                "propagation_event_count": (node_count, node_count),
                "propagation_event_support": (node_count, node_count),
                "propagation_stimulus_reachable_mask": (node_count, node_count),
                "propagation_feedback_core_mask": (node_count, node_count),
            }
            for key, shape in expected_shapes.items():
                if key not in data or data[key].shape != shape:
                    errors.append(f"{prefix}: {key} shape mismatch")
            for key in ("global_mean_rho", "chimera_index",
                        "metastability_index"):
                values = np.asarray(data[key])
                if not np.all(np.isfinite(values[completed])):
                    errors.append(f"{prefix}: non-finite {key}")
            if not np.isclose(float(data["statistics_start_ms"]), 1200.0):
                errors.append(f"{prefix}: statistics do not start at 1200 ms")
            if not np.isclose(float(data["critical_c5"]),
                              float(critical[subject]["critical_c5"])):
                errors.append(f"{prefix}: wrong critical-point source/value")
            if not bool(data["baseline_qc_passed"]):
                errors.append(f"{prefix}: baseline QC failed")

            responsive = np.asarray(data["propagation_responsive_mask"], bool)
            parent = np.asarray(data["propagation_parent"], int)
            reachable = np.asarray(
                data["propagation_stimulus_reachable_mask"], bool)
            core = np.asarray(data["propagation_feedback_core_mask"], bool)
            adjacency = np.asarray(connectivity[:, :, subject])
            for stimulus in np.flatnonzero(completed):
                if not responsive[stimulus, stimulus]:
                    errors.append(f"{prefix}: stimulus {stimulus} not responsive")
                if parent[stimulus, stimulus] != -1:
                    errors.append(f"{prefix}: stimulus {stimulus} has parent")
                targets = np.flatnonzero(parent[stimulus] >= 0)
                sources = parent[stimulus, targets]
                if np.any(adjacency[sources, targets] <= 0):
                    errors.append(f"{prefix}: non-structural propagation edge")
                if np.any(~responsive[stimulus, targets]) \
                        or np.any(~responsive[stimulus, sources]):
                    errors.append(f"{prefix}: edge touches nonresponsive node")
                if reachable[stimulus, stimulus]:
                    errors.append(f"{prefix}: reachable mask includes root")
                if np.any(reachable[stimulus] & ~responsive[stimulus]):
                    errors.append(f"{prefix}: reachable nonresponsive node")
                if np.any(core[stimulus] & ~responsive[stimulus]):
                    errors.append(f"{prefix}: core nonresponsive node")
            count_checks = {
                "propagation_responsive_count": responsive.sum(axis=1),
                "propagation_edge_count": (parent >= 0).sum(axis=1),
                "propagation_stimulus_reachable_count": reachable.sum(axis=1),
                "propagation_feedback_core_count": core.sum(axis=1),
            }
            for key, expected in count_checks.items():
                if not np.array_equal(np.asarray(data[key])[completed],
                                      expected[completed]):
                    errors.append(f"{prefix}: inconsistent {key}")
            _validate_cycles(data, errors, prefix)
    if errors:
        raise RuntimeError("Validation failed:\n" + "\n".join(errors))
    print(f"PASS: {args.group}, subjects={len(subjects)}, nodes={node_count}")


if __name__ == "__main__":
    main()
