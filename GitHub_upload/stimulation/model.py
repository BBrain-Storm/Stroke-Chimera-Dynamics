"""Stochastic stimulated delayed Wilson-Cowan model (Heun/RK2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StimulationConfig:
    dt_ms: float
    total_ms: float
    transient_ms: float
    stimulus_start_ms: float
    stimulus_end_ms: float
    stimulus_intensity: float
    noise_strength: float
    initial_e: float = 0.1
    initial_i: float = 0.1


def _sigmoid(x: np.ndarray, slope: float, threshold: float) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-slope * (x - threshold)))
            - 1.0 / (1.0 + np.exp(slope * threshold)))


def simulate_stimulated(adjacency: np.ndarray, delays_ms: np.ndarray,
                        c5: float, stimulated_node: int | None,
                        config: StimulationConfig,
                        seed: int) -> tuple[np.ndarray, np.ndarray]:
    graph = np.asarray(adjacency, dtype=np.float64).copy()
    delays = np.asarray(delays_ms, dtype=np.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("adjacency must be square")
    if delays.shape != graph.shape:
        raise ValueError("delay shape mismatch")
    np.fill_diagonal(graph, 0.0)
    node_count = graph.shape[0]
    if stimulated_node is not None and not 0 <= stimulated_node < node_count:
        raise IndexError(stimulated_node)
    dt = float(config.dt_ms)
    total_steps = int(round(config.total_ms / dt)) + 1
    transient_steps = int(round(config.transient_ms / dt))
    analysis_steps = total_steps - transient_steps
    stimulus_start = int(round(config.stimulus_start_ms / dt))
    stimulus_end = min(int(round(config.stimulus_end_ms / dt)), total_steps - 1)
    if transient_steps < 1 or analysis_steps < 2:
        raise ValueError("invalid transient/analysis duration")
    if stimulus_start != transient_steps:
        raise ValueError("stimulation must begin at the saved-window boundary")

    delay_steps = np.rint(delays / dt).astype(np.int64)
    buffer_size = int(delay_steps.max()) + 2
    current_e = np.full(node_count, config.initial_e, dtype=np.float64)
    current_i = np.full(node_count, config.initial_i, dtype=np.float64)
    e_buffer = np.full((buffer_size, node_count), current_e, dtype=np.float64)
    i_buffer = np.full((buffer_size, node_count), current_i, dtype=np.float64)
    e_output = np.empty((analysis_steps, node_count), dtype=np.float64)
    i_output = np.empty_like(e_output)

    inv_tau = 1.0 / 8.0
    c1, c2, c3, c4 = 16.0, 12.0, 15.0, 3.0
    ae, ai, theta_e, theta_i = 1.3, 2.0, 4.0, 3.7
    smax_e = 1.0 - 1.0 / (1.0 + np.exp(ae * theta_e))
    smax_i = 1.0 - 1.0 / (1.0 + np.exp(ai * theta_i))
    c6 = float(c5) / 4.0
    source_columns = np.broadcast_to(np.arange(node_count),
                                     (node_count, node_count))
    rng = np.random.default_rng(int(seed))
    noise_scale = config.noise_strength * np.sqrt(dt)
    current_index = 0
    saved = 0

    for step in range(1, total_steps):
        previous_index = current_index
        current_index = (current_index + 1) % buffer_size
        delayed_rows = (previous_index - delay_steps).T % buffer_size
        delayed_e = e_buffer[delayed_rows, source_columns]
        delayed_i = i_buffer[delayed_rows, source_columns]
        coupling_e = np.sum(graph.T * delayed_e, axis=1)
        coupling_i = np.sum(graph.T * delayed_i, axis=1)
        stimulus_active = stimulus_start <= (step - 1) <= stimulus_end

        input_e1 = c1 * current_e - c2 * current_i + c5 * coupling_e
        if stimulus_active and stimulated_node is not None:
            input_e1[stimulated_node] += config.stimulus_intensity
        input_i1 = c3 * current_e - c4 * current_i + c6 * coupling_i
        response_e1 = _sigmoid(input_e1, ae, theta_e)
        response_i1 = _sigmoid(input_i1, ai, theta_i)
        drift_e1 = inv_tau * (-current_e + (smax_e - current_e) * response_e1)
        drift_i1 = inv_tau * (-current_i + (smax_i - current_i) * response_i1)
        dw_e = noise_scale * rng.standard_normal(node_count)
        dw_i = noise_scale * rng.standard_normal(node_count)
        predictor_e = current_e + drift_e1 * dt + inv_tau * dw_e
        predictor_i = current_i + drift_i1 * dt + inv_tau * dw_i

        input_e2 = c1 * predictor_e - c2 * predictor_i + c5 * coupling_e
        if stimulus_active and stimulated_node is not None:
            input_e2[stimulated_node] += config.stimulus_intensity
        input_i2 = c3 * predictor_e - c4 * predictor_i + c6 * coupling_i
        response_e2 = _sigmoid(input_e2, ae, theta_e)
        response_i2 = _sigmoid(input_i2, ai, theta_i)
        drift_e2 = inv_tau * (-predictor_e + (smax_e - predictor_e) * response_e2)
        drift_i2 = inv_tau * (-predictor_i + (smax_i - predictor_i) * response_i2)
        current_e = (current_e + 0.5 * (drift_e1 + drift_e2) * dt
                     + inv_tau * dw_e)
        current_i = (current_i + 0.5 * (drift_i1 + drift_i2) * dt
                     + inv_tau * dw_i)
        e_buffer[current_index] = current_e
        i_buffer[current_index] = current_i
        if step >= transient_steps:
            e_output[saved] = current_e
            i_output[saved] = current_i
            saved += 1
    if saved != analysis_steps:
        raise RuntimeError(f"saved {saved} samples; expected {analysis_steps}")
    if not np.all(np.isfinite(e_output)) or not np.all(np.isfinite(i_output)):
        raise FloatingPointError("non-finite stimulated trajectory")
    return e_output, i_output
