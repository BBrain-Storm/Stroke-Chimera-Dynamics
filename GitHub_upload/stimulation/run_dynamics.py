"""Run all node stimulations and save synchrony plus propagation subnetworks."""

from __future__ import annotations

import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .common import (DATASET_GROUPS, DEFAULT_CRITICAL_ROOT, DEFAULT_DATA_DIR,
                     ROOT, atomic_savez, cohort_subjects, dynamics_file,
                     load_group, load_stimulation_config, read_critical_table,
                     validate_adjacency)
from .metrics import calculate_synchronization_metrics
from .model import StimulationConfig, simulate_stimulated
from .propagation import calculate_dominant_propagation


_ADJACENCY: np.ndarray | None = None
_DELAYS_MS: np.ndarray | None = None
_DELAY_STEPS: np.ndarray | None = None
_TEMPLATE: np.ndarray | None = None
_INCLUDED: np.ndarray | None = None
_MODEL_CONFIG: StimulationConfig | None = None
_STATS_START_INDEX: int | None = None
_PROPAGATION_THRESHOLD: float | None = None
_PROPAGATION_MIN_EVENTS: int | None = None


def _initialize_subject(adjacency, delays_ms, delay_steps, template, included,
                        model_config, stats_start_index,
                        propagation_threshold, propagation_min_events):
    global _ADJACENCY, _DELAYS_MS, _DELAY_STEPS, _TEMPLATE, _INCLUDED
    global _MODEL_CONFIG, _STATS_START_INDEX
    global _PROPAGATION_THRESHOLD, _PROPAGATION_MIN_EVENTS
    _ADJACENCY = adjacency
    _DELAYS_MS = delays_ms
    _DELAY_STEPS = delay_steps
    _TEMPLATE = template
    _INCLUDED = included
    _MODEL_CONFIG = model_config
    _STATS_START_INDEX = int(stats_start_index)
    _PROPAGATION_THRESHOLD = float(propagation_threshold)
    _PROPAGATION_MIN_EVENTS = int(propagation_min_events)


def _simulate_node(node: int, c5: float, seed: int) -> dict:
    required = (_ADJACENCY, _DELAYS_MS, _DELAY_STEPS, _TEMPLATE, _INCLUDED,
                _MODEL_CONFIG, _STATS_START_INDEX, _PROPAGATION_THRESHOLD,
                _PROPAGATION_MIN_EVENTS)
    if any(value is None for value in required):
        raise RuntimeError("worker was not initialized")
    excitatory, inhibitory = simulate_stimulated(
        _ADJACENCY, _DELAYS_MS, c5, node, _MODEL_CONFIG, seed)
    metrics = calculate_synchronization_metrics(
        excitatory[_STATS_START_INDEX:], inhibitory[_STATS_START_INDEX:],
        _TEMPLATE, _INCLUDED)
    propagation = calculate_dominant_propagation(
        excitatory, inhibitory, _ADJACENCY, _DELAY_STEPS, c5, node,
        _STATS_START_INDEX, _PROPAGATION_THRESHOLD, _PROPAGATION_MIN_EVENTS)
    return {
        "node": node, **metrics,
        "responsive": propagation.responsive,
        "parent": propagation.parent,
        "drive_score": propagation.drive_score,
        "event_count": propagation.event_count,
        "event_support": propagation.event_support,
        "reachable": propagation.reachable,
        "core": propagation.core,
        "cycles": propagation.cycles,
        "largest_component_nodes": propagation.largest_component_nodes,
    }


def _persistent_mask(excitatory: np.ndarray, included: np.ndarray,
                     raw: dict) -> np.ndarray:
    amplitude = (np.quantile(excitatory, 0.99, axis=0)
                 - np.quantile(excitatory, 0.01, axis=0))
    half = len(excitatory) // 2
    first = (np.quantile(excitatory[:half], 0.99, axis=0)
             - np.quantile(excitatory[:half], 0.01, axis=0))
    second = (np.quantile(excitatory[half:], 0.99, axis=0)
              - np.quantile(excitatory[half:], 0.01, axis=0))
    ratio = np.divide(second, first, out=np.full(len(first), np.nan),
                      where=first > np.finfo(float).tiny)
    threshold = float(raw["baseline_amplitude_threshold"])
    return (included & (amplitude > threshold) & (first > threshold)
            & (second > threshold)
            & (ratio >= raw["baseline_window_ratio_min"])
            & (ratio <= raw["baseline_window_ratio_max"]))


def _compress_cycles(cycles_by_stimulus: list[list[list[int]]]):
    stimulus_ids: list[int] = []
    lengths: list[int] = []
    nodes: list[int] = []
    indptr = [0]
    for stimulus, cycles in enumerate(cycles_by_stimulus):
        for cycle in cycles:
            stimulus_ids.append(stimulus)
            lengths.append(len(cycle))
            nodes.extend(cycle)
            indptr.append(len(nodes))
    return {
        "propagation_cycle_stimulus": np.asarray(stimulus_ids, dtype=np.int16),
        "propagation_cycle_length": np.asarray(lengths, dtype=np.int16),
        "propagation_cycle_nodes": np.asarray(nodes, dtype=np.int16),
        "propagation_cycle_indptr": np.asarray(indptr, dtype=np.int32),
    }


def _completed_output_covers(path: Path, requested_stimuli: list[int],
                             statistics_start_ms: float) -> bool:
    """Return true only for a readable output covering all requested stimuli."""
    try:
        with np.load(path) as data:
            completed = np.asarray(data["completed_stimulus_mask"], dtype=bool)
            return (completed.ndim == 1
                    and np.all(completed[requested_stimuli])
                    and np.isclose(float(data["statistics_start_ms"]),
                                   statistics_start_ms))
    except (OSError, KeyError, ValueError):
        return False


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
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--subjects")
    parser.add_argument("--stimuli",
                        help="Optional comma-separated 0-based nodes for testing")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    raw, model_config = load_stimulation_config(args.config)
    if not 1 <= args.max_workers <= int(raw["max_physical_cores"]):
        parser.error("max-workers outside configured physical-core limit")

    connectivity, delays_ms = load_group(
        args.data_dir, args.group, raw["delay_speed_divisor"])
    dataset = np.load(args.data_dir / f"Dataset_{args.group}.npz",
                      allow_pickle=False)
    template = np.asarray(dataset["template"]).squeeze()
    if len(np.unique(template)) != 13:
        raise ValueError(f"{args.group}: expected 13 systems")
    critical_rows = read_critical_table(
        args.critical_root / "aggregate" / f"critical_points_{args.group}.csv",
        connectivity.shape[2])
    subjects = cohort_subjects(args.group, connectivity.shape[2], args.subjects)
    node_count = connectivity.shape[0]
    if args.stimuli is None:
        stimuli = list(range(node_count))
    else:
        stimuli = sorted({int(x) for x in args.stimuli.split(",") if x.strip()})
        if not stimuli or stimuli[0] < 0 or stimuli[-1] >= node_count:
            raise ValueError(f"stimuli must be in [0, {node_count - 1}]")
    partial_stimulus_run = len(stimuli) != node_count
    statistics_start_index = int(round(
        (float(raw["statistics_start_ms"]) - float(raw["analysis_start_ms"]))
        / float(raw["dt_ms"])))
    delay_steps = np.rint(delays_ms / float(raw["dt_ms"])).astype(np.int64)

    baseline_failures = []
    simulation_failures = []
    for subject in subjects:
        output_path = dynamics_file(args.output_root, args.group, subject)
        if output_path.exists() and not args.force:
            if _completed_output_covers(
                    output_path, stimuli, float(raw["statistics_start_ms"])):
                print(f"SKIP group={args.group} subject={subject}: {output_path}",
                      flush=True)
                continue
            print(f"RECOMPUTE incomplete/incompatible output: {output_path}",
                  flush=True)
        row = critical_rows[subject]
        if row["status"] != "ok":
            raise ValueError(
                f"{args.group} subject {subject}: critical status={row['status']}")
        critical_c5 = float(row["critical_c5"])
        largest_quiescent = float(row["largest_quiescent_c5"])
        initial_coupling = critical_c5 - float(raw["critical_offset"])
        coupling = initial_coupling
        if coupling < 0 or not np.isclose(
                coupling, largest_quiescent, atol=5.0e-10):
            raise ValueError(
                f"subject {subject}: critical offset is inconsistent with table")
        adjacency = validate_adjacency(connectivity[:, :, subject], subject)
        degree = (np.count_nonzero(adjacency, axis=0)
                  + np.count_nonzero(adjacency, axis=1))
        included = (degree > 0 if raw["exclude_isolated_nodes_from_metrics"]
                    else np.ones(node_count, dtype=bool))
        system_count = len(np.unique(template))
        common_seed = int(raw["noise_seed"]) + 100003 * subject

        baseline_tested_c5 = []
        baseline_tested_active_count = []
        baseline_tested_global_rho = []
        if raw["run_no_stimulus_baseline_qc"]:
            retreat_step = float(raw["baseline_retreat_step"])
            scale = int(round(1.0 / retreat_step))
            candidate_index = int(round(coupling * scale))
            minimum_index = int(round(float(raw["minimum_coupling_c5"]) * scale))
            baseline_active_count = -1
            while candidate_index >= minimum_index:
                coupling = candidate_index / scale
                baseline_e, baseline_i = simulate_stimulated(
                    adjacency, delays_ms, coupling, None,
                    model_config, common_seed)
                stats_e = baseline_e[statistics_start_index:]
                stats_i = baseline_i[statistics_start_index:]
                baseline_metrics = calculate_synchronization_metrics(
                    stats_e, stats_i, template, included)
                baseline_persistent = _persistent_mask(stats_e, included, raw)
                baseline_active_count = int(np.sum(baseline_persistent))
                baseline_tested_c5.append(coupling)
                baseline_tested_active_count.append(baseline_active_count)
                baseline_tested_global_rho.append(
                    baseline_metrics["global_mean_rho"])
                print(f"BASELINE_QC group={args.group} subject={subject} "
                      f"c5={coupling:.3f} active_nodes={baseline_active_count}",
                      flush=True)
                if baseline_active_count == 0:
                    break
                if not raw["adaptive_baseline_retreat"]:
                    break
                candidate_index -= 1
            if baseline_active_count != 0 and raw["abort_if_baseline_not_fixed"]:
                baseline_failures.append(subject)
                continue
        else:
            baseline_metrics = {
                "global_mean_rho": np.nan,
                "system_pair_rho": np.full((system_count, system_count), np.nan),
                "chimera_index": np.nan, "metastability_index": np.nan,
            }
            baseline_persistent = np.zeros(node_count, dtype=bool)
            baseline_active_count = -1

        shape = (node_count, node_count)
        system_pair = np.full((system_count, system_count, node_count),
                              np.nan, np.float32)
        global_rho = np.full(node_count, np.nan, np.float64)
        chimera = np.full(node_count, np.nan, np.float64)
        metastability = np.full(node_count, np.nan, np.float64)
        responsive = np.zeros(shape, dtype=bool)
        parent = np.full(shape, -1, dtype=np.int16)
        drive_score = np.zeros(shape, dtype=np.float32)
        event_count = np.zeros(shape, dtype=np.int16)
        event_support = np.zeros(shape, dtype=np.float32)
        reachable = np.zeros(shape, dtype=bool)
        core = np.zeros(shape, dtype=bool)
        responsive_count = np.zeros(node_count, dtype=np.int16)
        edge_count = np.zeros(node_count, dtype=np.int16)
        reachable_count = np.zeros(node_count, dtype=np.int16)
        loop_count = np.zeros(node_count, dtype=np.int16)
        largest_loop = np.zeros(node_count, dtype=np.int16)
        core_count = np.zeros(node_count, dtype=np.int16)
        largest_component = np.zeros(node_count, dtype=np.int16)
        cycles_by_stimulus: list[list[list[int]]] = [
            [] for _ in range(node_count)]
        completed_stimulus = np.zeros(node_count, dtype=bool)
        system_ids = None
        failures = []

        with ProcessPoolExecutor(
                max_workers=min(args.max_workers, len(stimuli)),
                initializer=_initialize_subject,
                initargs=(adjacency, delays_ms, delay_steps, template, included,
                          model_config, statistics_start_index,
                          raw["propagation_amplitude_threshold"],
                          raw["propagation_min_events"])) as pool:
            futures = {}
            for node in stimuli:
                seed = (common_seed if raw["common_noise_within_subject"]
                        else common_seed + 1009 * node)
                futures[pool.submit(_simulate_node, node, coupling, seed)] = node
            for future in as_completed(futures):
                node = futures[future]
                try:
                    result = future.result()
                    system_pair[:, :, node] = result["system_pair_rho"]
                    global_rho[node] = result["global_mean_rho"]
                    chimera[node] = result["chimera_index"]
                    metastability[node] = result["metastability_index"]
                    system_ids = result["system_ids"]
                    responsive[node] = result["responsive"]
                    parent[node] = result["parent"]
                    drive_score[node] = result["drive_score"]
                    event_count[node] = result["event_count"]
                    event_support[node] = result["event_support"]
                    reachable[node] = result["reachable"]
                    core[node] = result["core"]
                    cycles_by_stimulus[node] = result["cycles"]
                    responsive_count[node] = np.sum(result["responsive"])
                    edge_count[node] = np.sum(result["parent"] >= 0)
                    reachable_count[node] = np.sum(result["reachable"])
                    loop_count[node] = len(result["cycles"])
                    largest_loop[node] = max(
                        map(len, result["cycles"]), default=0)
                    core_count[node] = np.sum(result["core"])
                    largest_component[node] = result["largest_component_nodes"]
                    completed_stimulus[node] = True
                    print(f"DONE group={args.group} subject={subject} "
                          f"node={node}/{node_count - 1}", flush=True)
                except Exception:
                    failures.append(node)
                    print(f"FAILED group={args.group} subject={subject} "
                          f"node={node}\n{traceback.format_exc()}", flush=True)
        if failures:
            simulation_failures.append((subject, failures))
            continue
        cycle_arrays = _compress_cycles(cycles_by_stimulus)
        atomic_savez(
            output_path, group=np.array(args.group),
            subject_index=np.array(subject, dtype=np.int32),
            critical_c5=np.array(critical_c5), coupling_c5=np.array(coupling),
            initial_coupling_c5=np.array(initial_coupling),
            critical_offset=np.array(raw["critical_offset"]),
            actual_critical_offset=np.array(critical_c5 - coupling),
            adaptive_retreat_steps=np.array(
                max(0, len(baseline_tested_c5) - 1), dtype=np.int16),
            c6=np.array(coupling / 4.0), dt_ms=np.array(raw["dt_ms"]),
            transient_ms=np.array(raw["transient_ms"]),
            analysis_start_ms=np.array(raw["analysis_start_ms"]),
            statistics_start_ms=np.array(raw["statistics_start_ms"]),
            analysis_end_ms=np.array(raw["analysis_end_ms"]),
            statistics_start_index=np.array(statistics_start_index),
            stimulus_intensity=np.array(raw["stimulus_intensity"]),
            noise_strength=np.array(raw["noise_strength"]),
            common_noise_within_subject=np.array(
                raw["common_noise_within_subject"]),
            noise_seed=np.array(common_seed, dtype=np.int64),
            critical_results_root=np.array(str(args.critical_root.resolve())),
            paired_patient_count=np.array(34, dtype=np.int16),
            partial_stimulus_run=np.array(partial_stimulus_run),
            completed_stimulus_mask=completed_stimulus,
            baseline_qc_enabled=np.array(raw["run_no_stimulus_baseline_qc"]),
            baseline_qc_passed=np.array(baseline_active_count == 0),
            baseline_active_node_count=np.array(baseline_active_count),
            baseline_tested_c5=np.asarray(baseline_tested_c5),
            baseline_tested_active_node_count=np.asarray(
                baseline_tested_active_count, dtype=np.int16),
            baseline_tested_global_mean_rho=np.asarray(
                baseline_tested_global_rho),
            baseline_persistent_mask=baseline_persistent,
            baseline_global_mean_rho=np.array(
                baseline_metrics["global_mean_rho"]),
            baseline_system_pair_rho=np.asarray(
                baseline_metrics["system_pair_rho"], dtype=np.float32),
            baseline_chimera_index=np.array(
                baseline_metrics["chimera_index"]),
            baseline_metastability_index=np.array(
                baseline_metrics["metastability_index"]),
            degree=degree.astype(np.int16), included_node_mask=included,
            isolated_node_indices=np.flatnonzero(~included).astype(np.int16),
            system_ids=np.asarray(system_ids), system_pair_rho=system_pair,
            global_mean_rho=global_rho, chimera_index=chimera,
            metastability_index=metastability,
            propagation_amplitude_threshold=np.array(
                raw["propagation_amplitude_threshold"]),
            propagation_min_events=np.array(
                raw["propagation_min_events"], dtype=np.int16),
            propagation_responsive_mask=responsive,
            propagation_parent=parent,
            propagation_drive_score=drive_score,
            propagation_event_count=event_count,
            propagation_event_support=event_support,
            propagation_stimulus_reachable_mask=reachable,
            propagation_feedback_core_mask=core,
            propagation_responsive_count=responsive_count,
            propagation_edge_count=edge_count,
            propagation_stimulus_reachable_count=reachable_count,
            propagation_feedback_loop_count=loop_count,
            propagation_largest_loop_nodes=largest_loop,
            propagation_feedback_core_count=core_count,
            propagation_largest_component_nodes=largest_component,
            propagation_storage_convention=np.array(
                "row=stimulus,target-column; parent[target]=source; -1=no edge"),
            **cycle_arrays)
        print(f"SAVED group={args.group} subject={subject}: {output_path}",
              flush=True)
    if baseline_failures:
        raise RuntimeError(f"baseline QC failed: {baseline_failures}")
    if simulation_failures:
        raise RuntimeError(f"node simulations failed: {simulation_failures}")


if __name__ == "__main__":
    main()
