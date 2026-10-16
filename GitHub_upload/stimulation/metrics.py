"""Synchronization metrics evaluated only after the configured transient mask."""

from __future__ import annotations

import numpy as np


def calculate_synchronization_metrics(
        excitatory: np.ndarray, inhibitory: np.ndarray,
        template: np.ndarray, included_nodes: np.ndarray) -> dict:
    e = np.asarray(excitatory, dtype=np.float64)
    i = np.asarray(inhibitory, dtype=np.float64)
    labels = np.asarray(template).squeeze()
    included = np.asarray(included_nodes, dtype=bool)
    if e.shape != i.shape or e.ndim != 2:
        raise ValueError("E and I must have the same (time, nodes) shape")
    if len(labels) != e.shape[1] or len(included) != e.shape[1]:
        raise ValueError("template/included mask length mismatch")
    phase = np.arctan2(i, e)
    phasor = np.exp(1j * phase)
    global_rho_t = np.abs(np.mean(phasor[:, included], axis=1))
    system_ids = np.unique(labels)
    masks = [(labels == system_id) & included for system_id in system_ids]
    if any(not np.any(mask) for mask in masks):
        raise ValueError("one or more systems contain no included nodes")
    system_rho_t = np.column_stack(
        [np.abs(np.mean(phasor[:, mask], axis=1)) for mask in masks])
    system_pair = np.empty((len(masks), len(masks)), dtype=np.float64)
    for row in range(len(masks)):
        for col in range(row, len(masks)):
            mask = masks[row] | masks[col]
            value = np.mean(np.abs(np.mean(phasor[:, mask], axis=1)))
            system_pair[row, col] = system_pair[col, row] = value
    return {
        "global_mean_rho": float(np.mean(global_rho_t)),
        "system_pair_rho": system_pair,
        "chimera_index": float(np.mean(np.var(system_rho_t, axis=1, ddof=1))),
        "metastability_index": float(np.mean(np.var(system_rho_t, axis=0, ddof=1))),
        "system_ids": system_ids,
    }
