"""Checks for split isolation, validation, and uncertainty calculations."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mnist_assignment.data import flatten, load_split, training_fingerprint
from mnist_assignment.evaluate import wilson_interval
from mnist_assignment.train import model_specs
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


class ProtocolTests(unittest.TestCase):
    def test_training_loader_needs_no_test_arrays(self):
        images = np.zeros((10, 28, 28), dtype=np.uint8)
        labels = np.array([1, 7] * 5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "training_only.npz"
            np.savez(path, train_data=images, train_targets=labels)
            actual, target = load_split(path, "train")
        np.testing.assert_array_equal(actual, images)
        np.testing.assert_array_equal(target, labels)

    def test_invalid_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.npz"
            images = np.zeros((2, 28, 28))
            images[0, 0, 0] = np.nan
            np.savez(path, train_data=images, train_targets=[1, 7])
            with self.assertRaises(ValueError):
                load_split(path, "train")

    def test_fixed_scaling_preserves_endpoints(self):
        x = np.zeros((2, 28, 28), dtype=np.uint8)
        x[1] = 255
        actual = flatten(x)
        self.assertEqual(actual.shape, (2, 784))
        np.testing.assert_array_equal(actual[:, 0], [0, 1])

    def test_both_models_fit_folds_with_constant_features(self):
        rng = np.random.default_rng(5)
        x = np.column_stack([rng.normal(size=(20, 5)), np.zeros((20, 2))])
        y = np.array([1, 7] * 10)
        for estimator, _ in model_specs().values():
            for training, validation in StratifiedKFold(2).split(x, y):
                model = Pipeline([("filter", VarianceThreshold()), ("model", estimator)])
                model.fit(x[training], y[training])
                self.assertEqual(len(model.predict(x[validation])), len(validation))
                self.assertEqual(model.named_steps["filter"].n_features_in_, 7)
                self.assertEqual(model.named_steps["filter"].get_support().sum(), 5)

    def test_wilson_interval_boundaries(self):
        low, high = wilson_interval(0, 100)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, 0)
        low, high = wilson_interval(100, 100)
        self.assertLess(low, 1)
        self.assertAlmostEqual(high, 1)

    def test_fingerprint_changes_with_labels(self):
        images = np.zeros((2, 28, 28), dtype=np.uint8)
        self.assertNotEqual(training_fingerprint(images, np.array([1, 7])),
                            training_fingerprint(images, np.array([7, 1])))


if __name__ == "__main__":
    unittest.main()
