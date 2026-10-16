from __future__ import annotations

import unittest

import numpy as np

from stimulation.common import cohort_subjects
from stimulation.propagation import (_cycles_from_parent,
                                     _reachable_from_stimulus)


class PropagationGraphTests(unittest.TestCase):
    def test_arbitrary_length_cycle(self):
        # Edges 1->2, 2->3, 3->1 encoded as parent[target]=source.
        parent = np.array([-1, 3, 1, 2, -1], dtype=np.int16)
        cycles = _cycles_from_parent(parent)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(len(cycles[0]), 3)
        cycle = cycles[0]
        for index, source in enumerate(cycle):
            target = cycle[(index + 1) % len(cycle)]
            self.assertEqual(parent[target], source)

    def test_stimulus_reachability(self):
        # 0->1->2 and disconnected 3->4.
        parent = np.array([-1, 0, 1, -1, 3], dtype=np.int16)
        expected = np.array([True, True, True, False, False])
        self.assertTrue(np.array_equal(
            _reachable_from_stimulus(parent, 0), expected))

    def test_patient_cohort_is_first_34(self):
        self.assertEqual(cohort_subjects("Pats_t1", 54, None), list(range(34)))
        self.assertEqual(cohort_subjects("Pats_t2", 59, None), list(range(34)))
        with self.assertRaises(ValueError):
            cohort_subjects("Pats_t1", 54, "34")

    def test_controls_use_all_available_subjects(self):
        self.assertEqual(cohort_subjects("Controls", 46, None), list(range(46)))


if __name__ == "__main__":
    unittest.main()
