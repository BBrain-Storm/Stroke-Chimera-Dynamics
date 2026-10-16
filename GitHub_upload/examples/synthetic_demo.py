import json
import numpy as np

from stimulation.model import StimulationConfig, simulate_stimulated
from stimulation.metrics import calculate_synchronization_metrics
from stimulation.patton import classify_matrices
from stimulation.propagation import calculate_dominant_propagation


def main():
    seed = 42
    node_count = 26
    random = np.random.default_rng(seed)
    weights = random.uniform(0.1, 1.0, (node_count, node_count))
    np.fill_diagonal(weights, 0)
    weights /= weights.sum(axis=1, keepdims=True)
    delays_ms = random.uniform(1, 5, weights.shape)
    np.fill_diagonal(delays_ms, 0)
    config = StimulationConfig(dt_ms=0.1, total_ms=2000, transient_ms=1000,
                              stimulus_start_ms=1000, stimulus_end_ms=2000,
                              stimulus_intensity=1.25, noise_strength=1e-5)
    coupling = 1.2
    excitatory, inhibitory = simulate_stimulated(weights, delays_ms, coupling, 0, config, seed)
    start = int(round((1200 - 1000) / config.dt_ms))
    template = np.repeat(np.arange(13), 2)
    metrics = calculate_synchronization_metrics(excitatory[start:], inhibitory[start:], template,
                                                np.ones(node_count, dtype=bool))
    classification = classify_matrices(metrics['system_pair_rho'][:, :, None],
                                       np.ones(1, dtype=bool), 0.65, 0.7, seed)
    propagation = calculate_dominant_propagation(excitatory, inhibitory, weights,
                                                np.rint(delays_ms / config.dt_ms).astype(int),
                                                coupling, 0, start)
    names = ['coherent', 'chimera-like', 'incoherent']
    result = dict(seed=seed, synthetic_nodes=node_count, systems=13,
                  global_synchronization=metrics['global_mean_rho'],
                  state=names[int(classification['state_flags'][0])],
                  retained_edges=int(np.sum(propagation.parent >= 0)),
                  reached_descendants=int(propagation.reachable.sum()),
                  cycles=len(propagation.cycles))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
