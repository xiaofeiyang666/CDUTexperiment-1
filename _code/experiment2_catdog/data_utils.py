"""数据切分与二分类评估工具。"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence


def stratified_train_val_indices(
    labels: Sequence[int],
    seed: int = 42,
    train_ratio: float = 0.80,
) -> tuple[list[int], list[int]]:
    """在原始训练集内分层切分，测试集保持原样独立。"""
    if not labels:
        raise ValueError("labels 不能为空")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须介于 0 和 1 之间")
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        label = int(label)
        if label not in (0, 1):
            raise ValueError("二分类标签只能是 0 或 1")
        grouped[label].append(index)
    if set(grouped) != {0, 1}:
        raise ValueError("数据必须同时包含两个类别")

    rng = random.Random(seed)
    train: list[int] = []
    val: list[int] = []
    for class_indices in grouped.values():
        if len(class_indices) < 2:
            raise ValueError("每个类别至少需要 2 个样本")
        shuffled = list(class_indices)
        rng.shuffle(shuffled)
        n_train = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_ratio)))
        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def binary_confusion_matrix(
    true_labels: Sequence[int], predicted_labels: Sequence[int]
) -> list[list[int]]:
    """返回以真实类别为行、预测类别为列的 2x2 混淆矩阵。"""
    if len(true_labels) != len(predicted_labels):
        raise ValueError("真实标签和预测标签长度必须相同")
    matrix = [[0, 0], [0, 0]]
    for true, predicted in zip(true_labels, predicted_labels):
        if int(true) not in (0, 1) or int(predicted) not in (0, 1):
            raise ValueError("二分类标签只能是 0 或 1")
        matrix[int(true)][int(predicted)] += 1
    return matrix


def metrics_from_confusion_matrix(matrix: Sequence[Sequence[int]]) -> dict[str, float]:
    """根据 2x2 混淆矩阵计算准确率及宏平均 precision、recall、F1。"""
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("混淆矩阵必须为 2x2")
    total = sum(int(value) for row in matrix for value in row)
    if total <= 0:
        raise ValueError("混淆矩阵必须至少包含一个样本")

    per_class = []
    for class_id in range(2):
        true_positive = int(matrix[class_id][class_id])
        false_positive = sum(int(matrix[row][class_id]) for row in range(2)) - true_positive
        false_negative = sum(int(value) for value in matrix[class_id]) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append((precision, recall, f1))

    return {
        "accuracy": sum(int(matrix[i][i]) for i in range(2)) / total,
        "macro_precision": sum(values[0] for values in per_class) / 2,
        "macro_recall": sum(values[1] for values in per_class) / 2,
        "macro_f1": sum(values[2] for values in per_class) / 2,
    }
