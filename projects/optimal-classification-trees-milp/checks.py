import unittest

import numpy as np
from study import catalog, exhaustive_error, fit, predict


class TreeTests(unittest.TestCase):
    def test_xor(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        m = fit(X, y)
        self.assertTrue(m["proven_optimal"])
        np.testing.assert_array_equal(predict(m, X), y)

    def test_enumeration(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            X = rng.integers(0, 2, (8, 3))
            y = rng.integers(0, 2, 8)
            self.assertAlmostEqual(fit(X, y)["train_error"], exhaustive_error(X, y))

    def test_constant(self):
        m = fit(np.ones((5, 2)), np.ones(5))
        self.assertEqual(m["train_error"], 0)

    def test_stump(self):
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])
        self.assertEqual(fit(X, y, depth=1)["train_error"], 0)

    def test_catalog_train_only(self):
        self.assertEqual(catalog(np.array([[0], [2], [4]])), [(0, 1.0), (0, 3.0)])

    def test_invalid(self):
        for X, y in [(np.array([[np.nan]]), [1]), (np.ones((2, 1)), [0, 2])]:
            with self.assertRaises(ValueError):
                fit(X, y)

    def test_prediction_shape(self):
        m = fit(np.ones((2, 1)), [0, 1])
        with self.assertRaises(ValueError):
            predict(m, np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
