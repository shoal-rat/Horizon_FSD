import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_utils import (  # noqa: E402
    exclusive_pedals,
    model_to_physical_action,
    pedal_to_model,
    pedal_to_physical,
    physical_to_model_action,
)


class TestActionUtils(unittest.TestCase):
    def test_pedal_mapping(self):
        self.assertAlmostEqual(pedal_to_model(0.0), -1.0)
        self.assertAlmostEqual(pedal_to_model(0.5), 0.0)
        self.assertAlmostEqual(pedal_to_model(1.0), 1.0)
        self.assertAlmostEqual(pedal_to_physical(-1.0), 0.0)
        self.assertAlmostEqual(pedal_to_physical(0.0), 0.5)
        self.assertAlmostEqual(pedal_to_physical(1.0), 1.0)

    def test_action_roundtrip(self):
        physical = np.array([-0.25, 0.75, 0.10], dtype=np.float32)
        model = physical_to_model_action(physical)
        np.testing.assert_allclose(model, [-0.25, 0.5, -0.8], atol=1e-6)
        np.testing.assert_allclose(model_to_physical_action(model), physical, atol=1e-6)

    def test_exclusive_pedals(self):
        throttle, brake = exclusive_pedals(0.8, 0.2)
        self.assertAlmostEqual(throttle, 0.6)
        self.assertAlmostEqual(brake, 0.0)
        throttle, brake = exclusive_pedals(0.2, 0.8)
        self.assertAlmostEqual(throttle, 0.0)
        self.assertAlmostEqual(brake, 0.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
