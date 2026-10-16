import unittest
import numpy as np

from stimulation.patton import author_code_state, classify_matrices
from stimulation.propagation import calculate_dominant_propagation


class SyntheticIntegrationTests(unittest.TestCase):
    def test_delayed_drivers_reconstruct_chain(self):
        time = np.arange(0, 400, 0.5)
        phase = 2 * np.pi * time[:, None] / 40 + np.array([1.2, 0.6, 0.0])
        excitatory = 0.2 + 0.05 * np.sin(phase)
        inhibitory = np.full_like(excitatory, 0.1)
        weights = np.array([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
        result = calculate_dominant_propagation(excitatory, inhibitory, weights,
                                               np.full((3, 3), 2), 1.0, 0, 100)
        np.testing.assert_array_equal(result.parent, [-1, 0, 1])
        np.testing.assert_array_equal(result.reachable, [False, True, True])
        self.assertEqual(result.cycles, [])

    def test_three_operational_states(self):
        self.assertEqual(int(author_code_state(np.zeros(13, dtype=int))[0]), 0)
        self.assertEqual(int(author_code_state(np.arange(13))[0]), 2)
        self.assertEqual(int(author_code_state(np.repeat([0, 1], [6, 7]))[0]), 1)

    def test_uniform_synchrony_classifies_coherent(self):
        result = classify_matrices(np.ones((13, 13, 1)), np.ones(1, dtype=bool),
                                   threshold=0.65, gamma=0.7, seed=42)
        self.assertEqual(int(result['state_flags'][0]), 0)
        self.assertEqual(int(result['community_count'][0]), 1)


if __name__ == '__main__':
    unittest.main()
