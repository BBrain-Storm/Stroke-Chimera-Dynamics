"""Aggregate subject-by-stimulus propagation metrics for later statistics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .common import (DATASET_GROUPS, DEFAULT_DATA_DIR, ROOT, cohort_subjects,
                     dynamics_file, load_group, load_stimulation_config)


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
    node_count = connectivity.shape[0]
    fields = (
        "propagation_responsive_count", "propagation_edge_count",
        "propagation_stimulus_reachable_count",
        "propagation_feedback_loop_count",
        "propagation_largest_loop_nodes",
        "propagation_feedback_core_count",
        "propagation_largest_component_nodes")
    arrays = {field: np.empty((len(subjects), node_count), np.int16)
              for field in fields}
    global_rho = np.empty((len(subjects), node_count), np.float64)
    chimera = np.empty_like(global_rho)
    metastability = np.empty_like(global_rho)
    system_pair = np.empty((len(subjects), 13, 13, node_count), np.float32)
    coupling = np.empty(len(subjects), np.float64)
    rows = []
    for row_index, subject in enumerate(subjects):
        with np.load(dynamics_file(args.output_root, args.group, subject)) as data:
            if not np.all(data["completed_stimulus_mask"]):
                raise ValueError(f"subject {subject}: incomplete stimulus run")
            for field in fields:
                arrays[field][row_index] = data[field]
            global_rho[row_index] = data["global_mean_rho"]
            chimera[row_index] = data["chimera_index"]
            metastability[row_index] = data["metastability_index"]
            system_pair[row_index] = data["system_pair_rho"]
            coupling[row_index] = data["coupling_c5"]
        rows.append({
            "group": args.group,
            "subject_index_0based": subject,
            "subject_id_1based": subject + 1,
            "coupling_c5": f"{coupling[row_index]:.3f}",
            "mean_global_rho": f"{global_rho[row_index].mean():.9f}",
            "mean_chimera_index": f"{chimera[row_index].mean():.9f}",
            "mean_metastability_index": f"{metastability[row_index].mean():.9f}",
            "mean_responsive_nodes": f"{arrays['propagation_responsive_count'][row_index].mean():.9f}",
            "mean_dominant_edges": f"{arrays['propagation_edge_count'][row_index].mean():.9f}",
            "mean_stimulus_reachable_nodes": f"{arrays['propagation_stimulus_reachable_count'][row_index].mean():.9f}",
            "total_feedback_loops": int(arrays[
                "propagation_feedback_loop_count"][row_index].sum()),
            "stimuli_with_feedback_loop": int(np.sum(arrays[
                "propagation_feedback_loop_count"][row_index] > 0)),
            "maximum_loop_nodes": int(arrays[
                "propagation_largest_loop_nodes"][row_index].max()),
        })
    aggregate = args.output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    np.save(aggregate / f"system_pair_rho_{args.group}.npy", system_pair)
    np.savez_compressed(
        aggregate / f"dynamical_indices_{args.group}.npz",
        subject_indices=np.asarray(subjects, dtype=np.int16),
        statistics_start_ms=np.array(raw["statistics_start_ms"]),
        global_mean_rho=global_rho, chimera_index=chimera,
        metastability_index=metastability, coupling_c5=coupling)
    np.savez_compressed(
        aggregate / f"propagation_indices_{args.group}.npz",
        subject_indices=np.asarray(subjects, dtype=np.int16),
        statistics_start_ms=np.array(raw["statistics_start_ms"]),
        global_mean_rho=global_rho, chimera_index=chimera,
        metastability_index=metastability, coupling_c5=coupling, **arrays)
    table = aggregate / f"subject_propagation_summary_{args.group}.csv"
    with table.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {table}")


if __name__ == "__main__":
    main()
