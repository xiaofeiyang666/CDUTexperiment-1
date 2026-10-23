import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_utils import (
    binary_confusion_matrix,
    metrics_from_confusion_matrix,
    stratified_train_val_indices,
)


class StratifiedSplitTests(unittest.TestCase):
    def test_split_is_deterministic_complete_and_class_balanced(self):
        labels = [0] * 20 + [1] * 20

        first = stratified_train_val_indices(labels, seed=7)
        second = stratified_train_val_indices(labels, seed=7)

        self.assertEqual(first, second)
        train, val = first
        self.assertEqual(sorted(train + val), list(range(len(labels))))
        self.assertEqual(len(set(train) & set(val)), 0)
        for indices, expected in ((train, 16), (val, 4)):
            self.assertEqual(sum(labels[i] == 0 for i in indices), expected)
            self.assertEqual(sum(labels[i] == 1 for i in indices), expected)

    def test_split_rejects_classes_too_small_for_both_subsets(self):
        with self.assertRaises(ValueError):
            stratified_train_val_indices([0, 0, 0, 1], seed=3)


class ConfusionMatrixTests(unittest.TestCase):
    def test_matrix_uses_true_rows_and_predicted_columns(self):
        matrix = binary_confusion_matrix([0, 0, 1, 1, 1], [0, 1, 1, 0, 1])

        self.assertEqual(matrix, [[1, 1], [1, 2]])

    def test_matrix_rejects_mismatched_label_lengths(self):
        with self.assertRaises(ValueError):
            binary_confusion_matrix([0, 1], [1])

    def test_metrics_match_a_hand_checked_fixture(self):
        metrics = metrics_from_confusion_matrix([[1, 1], [1, 2]])

        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["macro_precision"], 7 / 12)
        self.assertAlmostEqual(metrics["macro_recall"], 7 / 12)
        self.assertAlmostEqual(metrics["macro_f1"], 7 / 12)


if __name__ == "__main__":
    unittest.main()
