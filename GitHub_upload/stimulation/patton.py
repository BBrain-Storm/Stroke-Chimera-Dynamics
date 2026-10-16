"""Patton objective with seeded single-run Louvain and strict states."""

from __future__ import annotations

import numpy as np

try:
    import bct
except ImportError as error:  # pragma: no cover
    bct = None
    _IMPORT_ERROR = error


COHERENT = np.uint8(0)
CHIMERA = np.uint8(1)
INCOHERENT = np.uint8(2)
EXCLUDED = np.uint8(255)
STATE_NAMES = ("coherent", "chimera", "incoherent")


def _canonical_partition(partition: np.ndarray) -> np.ndarray:
    labels: dict[int, int] = {}
    result = np.empty(len(partition), dtype=np.int16)
    for index, raw_label in enumerate(np.asarray(partition).ravel()):
        label = int(raw_label)
        if label not in labels:
            labels[label] = len(labels)
        result[index] = labels[label]
    return result


def author_louvain_partition(
        binary_matrix: np.ndarray, gamma: float, seed: int) -> np.ndarray:
    """Apply the Patton author-code modularity objective once."""
    if bct is None:
        raise ImportError("bctpy is required") from _IMPORT_ERROR
    matrix = np.asarray(binary_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("binary matrix must be square")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("binary matrix must be symmetric")
    system_count = matrix.shape[0]
    twom = float(np.sum(matrix))
    if twom == 0.0:
        return np.arange(system_count, dtype=np.int16)
    mean_degree = twom / system_count
    objective = (matrix
                 - float(gamma) * (mean_degree ** 2 / twom)
                 * np.ones_like(matrix))
    partition, _ = bct.community_louvain(
        matrix, B=objective.tolist(), seed=int(seed))
    return _canonical_partition(partition)


def author_code_state(partition: np.ndarray) -> tuple[np.uint8, int, int]:
    """Classify strictly by the number of unique communities."""
    _, counts = np.unique(np.asarray(partition), return_counts=True)
    community_count = len(counts)
    if community_count == 1:
        state = COHERENT
    elif community_count == len(partition):
        state = INCOHERENT
    else:
        state = CHIMERA
    return state, community_count, int(np.sum(counts == 1))


def classify_matrices(
        matrices: np.ndarray, eligible: np.ndarray, threshold: float,
        gamma: float, seed: int) -> dict[str, np.ndarray]:
    """Classify all eligible stimuli while caching identical binary patterns."""
    matrices = np.asarray(matrices, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if matrices.ndim != 3 or matrices.shape[0] != matrices.shape[1]:
        raise ValueError("expected (systems, systems, stimulated_nodes)")
    system_count, _, node_count = matrices.shape
    if len(eligible) != node_count:
        raise ValueError("eligible mask length mismatch")
    indices = np.flatnonzero(eligible)
    if not len(indices):
        raise ValueError("no eligible stimulated nodes")
    if not np.all(np.isfinite(matrices[:, :, indices])):
        raise ValueError("eligible system_pair_rho matrices contain non-finite values")

    states = np.full(node_count, EXCLUDED, dtype=np.uint8)
    partitions = np.full((node_count, system_count), -1, dtype=np.int16)
    community_counts = np.full(node_count, -1, dtype=np.int16)
    singleton_counts = np.full(node_count, -1, dtype=np.int16)
    entries_above = np.full(node_count, -1, dtype=np.int16)

    upper = np.triu_indices(system_count)
    patterns = np.transpose(matrices[:, :, indices], (2, 0, 1))
    upper_binary = patterns[:, upper[0], upper[1]] >= threshold
    packed = np.packbits(upper_binary, axis=1)
    unique_packed, inverse = np.unique(packed, axis=0, return_inverse=True)
    unique_partitions = np.empty((len(unique_packed), system_count), dtype=np.int16)
    upper_count = len(upper[0])
    for pattern_index, pattern in enumerate(unique_packed):
        bits = np.unpackbits(pattern)[:upper_count].astype(float)
        binary = np.zeros((system_count, system_count), dtype=float)
        binary[upper] = bits
        binary = binary + binary.T - np.diag(np.diag(binary))
        unique_partitions[pattern_index] = author_louvain_partition(
            binary, gamma=gamma, seed=seed + pattern_index)

    for local_index, node in enumerate(indices):
        partition = unique_partitions[inverse[local_index]]
        state, communities, singletons = author_code_state(partition)
        states[node] = state
        partitions[node] = partition
        community_counts[node] = communities
        singleton_counts[node] = singletons
        entries_above[node] = int(np.sum(matrices[:, :, node] >= threshold))
    proportions = np.array([
        np.mean(states[eligible] == COHERENT),
        np.mean(states[eligible] == CHIMERA),
        np.mean(states[eligible] == INCOHERENT),
    ])
    return {
        "state_flags": states,
        "partitions": partitions,
        "community_count": community_counts,
        "singleton_count": singleton_counts,
        "entries_above": entries_above,
        "state_proportions": proportions,
    }
