"""Delay-corrected, Wilson-Cowan-constrained dominant propagation network.

The returned parent vector is a lossless compact encoding of the directed
dominant graph: parent[target] is the single retained source, or -1.  The
stimulus-reachable mask selects the main propagation subnetwork rooted at the
stimulated node.  Directed cycles are enumerated without a length limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


INV_TAU = 1.0 / 8.0
C1, C2 = 16.0, 12.0
AE, THETA_E = 1.3, 4.0
SMAX_E = 1.0 - 1.0 / (1.0 + np.exp(AE * THETA_E))


@dataclass
class PropagationResult:
    responsive: np.ndarray
    parent: np.ndarray
    drive_score: np.ndarray
    event_count: np.ndarray
    event_support: np.ndarray
    reachable: np.ndarray
    core: np.ndarray
    cycles: list[list[int]]
    largest_component_nodes: int


def _sigmoid_derivative(x: float) -> float:
    z = np.exp(np.clip(-AE * (x - THETA_E), -700.0, 700.0))
    return float(AE * z / (1.0 + z) ** 2)


def _upward_crossings(signal: np.ndarray, start_index: int) -> np.ndarray:
    crossings = np.flatnonzero(
        (signal[:-1] <= 0.0) & (signal[1:] > 0.0)) + 1
    return crossings[crossings >= start_index]


def _cycles_from_parent(parent: np.ndarray) -> list[list[int]]:
    cycles: list[list[int]] = []
    completed = np.zeros(len(parent), dtype=bool)
    for start in range(len(parent)):
        if completed[start]:
            continue
        path: list[int] = []
        position: dict[int, int] = {}
        node = start
        while node >= 0 and not completed[node] and node not in position:
            position[node] = len(path)
            path.append(node)
            node = int(parent[node])
        if node >= 0 and node in position:
            cycle = path[position[node]:]
            if len(cycle) > 1:
                # parent edges point source -> target; reverse traversal order
                # into the actual edge direction for consistent reporting.
                cycles.append(list(reversed(cycle)))
        completed[path] = True
    cycles.sort(key=lambda cycle: (-len(cycle), tuple(cycle)))
    return cycles


def _reachable_from_stimulus(parent: np.ndarray, stimulus: int) -> np.ndarray:
    reachable = np.zeros(len(parent), dtype=bool)
    reachable[stimulus] = True
    changed = True
    while changed:
        previous = reachable.copy()
        valid = parent >= 0
        reachable[valid] |= reachable[parent[valid]]
        changed = not np.array_equal(previous, reachable)
    return reachable


def _largest_weak_component(parent: np.ndarray, responsive: np.ndarray) -> int:
    adjacency = [[] for _ in range(len(parent))]
    for target in np.flatnonzero(parent >= 0):
        source = int(parent[target])
        adjacency[source].append(int(target))
        adjacency[target].append(source)
    seen = np.zeros(len(parent), dtype=bool)
    largest = 0
    for start in np.flatnonzero(responsive):
        if seen[start]:
            continue
        stack = [int(start)]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        largest = max(largest, size)
    return largest


def calculate_dominant_propagation(
        excitatory: np.ndarray, inhibitory: np.ndarray,
        adjacency: np.ndarray, delay_steps: np.ndarray, c5: float,
        stimulated_node: int, analysis_start_index: int,
        amplitude_threshold: float = 1.0e-5,
        min_events: int = 5) -> PropagationResult:
    e = np.asarray(excitatory, dtype=np.float64)
    i = np.asarray(inhibitory, dtype=np.float64)
    graph = np.asarray(adjacency, dtype=np.float64)
    delays = np.asarray(delay_steps, dtype=np.int64)
    if e.shape != i.shape or e.ndim != 2:
        raise ValueError("E and I must have the same (time, nodes) shape")
    if graph.shape != (e.shape[1], e.shape[1]) or delays.shape != graph.shape:
        raise ValueError("adjacency/delay shape mismatch")
    if not 0 <= analysis_start_index < len(e) - 2:
        raise ValueError("invalid analysis_start_index")
    if not 0 <= stimulated_node < e.shape[1]:
        raise IndexError(stimulated_node)

    analysis = e[analysis_start_index:]
    means = analysis.mean(axis=0)
    centered = e - means
    amplitude = (np.quantile(analysis, 0.99, axis=0)
                 - np.quantile(analysis, 0.01, axis=0))
    nonisolated = ((graph > 0) | (graph.T > 0)).any(axis=1)
    responsive = (amplitude > amplitude_threshold) & nonisolated
    responsive[stimulated_node] = True
    node_count = e.shape[1]
    parent = np.full(node_count, -1, dtype=np.int16)
    score = np.zeros(node_count, dtype=np.float32)
    event_count = np.zeros(node_count, dtype=np.int16)
    support = np.zeros(node_count, dtype=np.float32)

    for target in np.flatnonzero(responsive):
        if target == stimulated_node:
            continue
        events = _upward_crossings(centered[:, target], analysis_start_index)
        if len(events) < min_events:
            continue
        sources = np.flatnonzero((graph[:, target] > 0) & responsive)
        if len(sources) == 0:
            continue
        all_sources = np.flatnonzero(graph[:, target] > 0)
        accumulated = np.zeros(len(sources), dtype=np.float64)
        winner_counts = np.zeros(len(sources), dtype=np.int32)
        usable = 0
        for event in events:
            source_indices = np.maximum(event - delays[sources, target], 0)
            delayed_delta = e[source_indices, sources] - means[sources]
            if not np.any(delayed_delta > 0.0):
                continue
            all_indices = np.maximum(event - delays[all_sources, target], 0)
            coupling_e = np.sum(
                graph[all_sources, target] * e[all_indices, all_sources])
            input_e = (C1 * e[event, target] - C2 * i[event, target]
                       + c5 * coupling_e)
            susceptibility = (INV_TAU * max(SMAX_E - e[event, target], 0.0)
                              * _sigmoid_derivative(input_e))
            contributions = (c5 * graph[sources, target]
                             * np.maximum(delayed_delta, 0.0)
                             * susceptibility)
            if not np.any(contributions > 0.0):
                continue
            accumulated += contributions
            winner_counts[int(np.argmax(contributions))] += 1
            usable += 1
        if usable < min_events or not np.any(accumulated > 0.0):
            continue
        best = int(np.argmax(accumulated))
        parent[target] = int(sources[best])
        score[target] = accumulated[best] / usable
        event_count[target] = usable
        support[target] = winner_counts[best] / usable

    cycles = _cycles_from_parent(parent)
    core = np.zeros(node_count, dtype=bool)
    for cycle in cycles:
        core[cycle] = True
    reachable = _reachable_from_stimulus(parent, stimulated_node)
    reachable &= responsive
    # Match the validation convention: descendants only; the known root is
    # added explicitly when reconstructing the rooted propagation subnetwork.
    reachable[stimulated_node] = False
    return PropagationResult(
        responsive=responsive, parent=parent, drive_score=score,
        event_count=event_count, event_support=support,
        reachable=reachable, core=core, cycles=cycles,
        largest_component_nodes=_largest_weak_component(parent, responsive))
