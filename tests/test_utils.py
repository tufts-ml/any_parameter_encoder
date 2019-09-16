import unittest
import numpy as np
from utils import softmax_1d, softmax


class TestUtils(unittest.TestCase):
    def test_softmax(self):
        np.testing.assert_array_almost_equal(
            np.array([0.25949646034242, 0.70538451269824, 0.03511902695934]),
            softmax_1d(np.array([3, 4, 1])))
        np.testing.assert_array_almost_equal(
            np.array(
                [[0.25949646034242, 0.70538451269824, 0.03511902695934],
                [7.471972337343E-43, 1.6889118802245E-48, 1]]),
            softmax(np.array([[3, 4, 1], [3, -10, 100]])))


if __name__ == "__main__":
    unittest.main()