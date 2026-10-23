"""Evaluate a saved checkpoint on the official, untouched Kaggle test set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

from data_utils import binary_confusion_matrix, metrics_from_confusion_matrix
from train import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR, IMAGE_SIZE, MEAN, STD, SEED


def read_training_log(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"训练日志为空：{path}")
    return [{key: float(value) for key, value in row.items()} for row in rows]


def save_training_process(rows: list[dict[str, float]], output_dir: Path) -> None:
    columns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    display_headers = ["Epoch", "Train loss", "Train acc", "Val loss", "Val acc"]
    cell_rows = []
    for row in rows:
        cell_rows.append(
            [
                str(int(row["epoch"])),
                f"{row['train_loss']:.4f}",
                f"{row['train_acc']:.2%}",
                f"{row['val_loss']:.4f}",
                f"{row['val_acc']:.2%}",
            ]
        )
    figure, axis = plt.subplots(figsize=(11, max(3.0, 0.48 * (len(cell_rows) + 1))))
    axis.axis("off")
    axis.set_title("Experiment 2 - Cats vs Dogs Training Log", fontsize=15, pad=14)
    table = axis.table(
        cellText=cell_rows,
        colLabels=display_headers,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.10, 0.19, 0.19, 0.19, 0.19],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dde5")
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f5f9")
    figure.tight_layout()
    figure.savefig(output_dir / "training_process.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_training_curves(rows: list[dict[str, float]], output_dir: Path) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o", label="Train loss")
    axes[0].plot(epochs, [row["val_loss"] for row in rows], marker="o", label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] * 100 for row in rows], marker="o", label="Train accuracy")
    axes[1].plot(epochs, [row["val_acc"] * 100 for row in rows], marker="o", label="Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Cats vs Dogs - Training Curves")
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix(matrix: list[list[int]], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks([0, 1], labels=["cat", "dog"])
    axis.set_yticks([0, 1], labels=["cat", "dog"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Confusion Matrix")
    threshold = max(max(row) for row in matrix) / 2
    for true_label in range(2):
        for predicted_label in range(2):
            value = matrix[true_label][predicted_label]
            axis.text(
                predicted_label,
                true_label,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=13,
            )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def select_prediction_examples(
    records: list[dict], seed: int = SEED, examples_per_group: int = 3
) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for correct in (True, False):
        for true_label in (0, 1):
            group = [
                record
                for record in records
                if record["true_id"] == true_label and record["correct"] is correct
            ]
            rng.shuffle(group)
            selected.extend(group[:examples_per_group])
    return selected


def save_prediction_examples(records: list[dict], output_dir: Path) -> None:
    selected = select_prediction_examples(records)
    if not selected:
        raise ValueError("没有可绘制的测试样本")
    columns = 4
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.25 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    names = ["cat", "dog"]
    for axis, record in zip(axes.flat, selected):
        with Image.open(record["path"]) as image:
            axis.imshow(image.convert("RGB"))
        color = "#16803c" if record["correct"] else "#c62828"
        confidence = record["probability"] * 100
        axis.set_title(
            f"True: {names[record['true_id']]} | Pred: {names[record['pred_id']]}\n"
            f"Confidence: {confidence:.1f}%",
            color=color,
            fontsize=9,
        )
        axis.axis("off")
    figure.suptitle("Held-out Test Predictions (green: correct, red: incorrect)", y=1.01)
    figure.tight_layout()
    figure.savefig(output_dir / "prediction_examples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size 必须为正")

    output_dir = args.output_dir
    checkpoint_path = output_dir / "best_model.pth"
    if not checkpoint_path.is_file():
        parser.error(f"找不到模型权重：{checkpoint_path}")
    test_root = args.data_root / "test"
    test_dataset = datasets.ImageFolder(
        test_root,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        ),
    )
    expected_classes = {"cat": 0, "dog": 1}
    if test_dataset.class_to_idx != expected_classes:
        raise ValueError(f"测试集类别映射不符：{test_dataset.class_to_idx}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["class_to_idx"] != expected_classes:
        raise ValueError("模型和测试集类别映射不一致")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    true_labels: list[int] = []
    predicted_labels: list[int] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            probs = logits.softmax(dim=1).cpu()
            preds = probs.argmax(dim=1)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(preds.tolist())
            probabilities.extend(probs.tolist())

    matrix = binary_confusion_matrix(true_labels, predicted_labels)
    metrics = metrics_from_confusion_matrix(matrix)
    names = ["cat", "dog"]
    records = []
    for index, ((path, true_id), predicted_id, probs) in enumerate(
        zip(test_dataset.samples, predicted_labels, probabilities)
    ):
        records.append(
            {
                "index": index,
                "path": path,
                "filename": Path(path).name,
                "true_id": true_id,
                "pred_id": predicted_id,
                "true_label": names[true_id],
                "predicted_label": names[predicted_id],
                "cat_probability": probs[0],
                "dog_probability": probs[1],
                "probability": probs[predicted_id],
                "correct": true_id == predicted_id,
            }
        )

    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "filename",
                "true_label",
                "predicted_label",
                "cat_probability",
                "dog_probability",
                "correct",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in writer.fieldnames})

    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    result = {
        "test_count": len(test_dataset),
        "correct_count": sum(true == predicted for true, predicted in zip(true_labels, predicted_labels)),
        "incorrect_count": sum(true != predicted for true, predicted in zip(true_labels, predicted_labels)),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "confusion_matrix_true_rows_predicted_columns": matrix,
        "classes": names,
        "best_epoch": checkpoint["best_epoch"],
        "best_validation_accuracy": checkpoint["best_val_accuracy"],
        "test_device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "training_device": metadata.get("device"),
        "architecture": checkpoint["architecture"],
        "pretrained_weights": checkpoint["pretrained_weights"],
        "pretrained_sha256": checkpoint["pretrained_sha256"],
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    correct_percent = metrics["accuracy"] * 100
    metrics_text = [
        "猫狗分类实验测试结果",
        f"测试集样本数：{result['test_count']}",
        f"预测正确：{result['correct_count']}",
        f"预测错误：{result['incorrect_count']}",
        f"测试准确率：{correct_percent:.2f}%",
        f"宏平均 Precision：{metrics['macro_precision']:.4f}",
        f"宏平均 Recall：{metrics['macro_recall']:.4f}",
        f"宏平均 F1：{metrics['macro_f1']:.4f}",
        f"混淆矩阵（行是真实类别，列是预测类别）：{matrix}",
        f"最佳验证轮次：{checkpoint['best_epoch']}",
        f"最佳验证准确率：{checkpoint['best_val_accuracy'] * 100:.2f}%",
    ]
    (output_dir / "test_metrics.txt").write_text("\n".join(metrics_text) + "\n", encoding="utf-8")

    history = read_training_log(output_dir / "training_log.csv")
    save_training_process(history, output_dir)
    save_training_curves(history, output_dir)
    save_confusion_matrix(matrix, output_dir)
    save_prediction_examples(records, output_dir)
    print("\n".join(metrics_text))
    print(f"预测示例图：{output_dir / 'prediction_examples.png'}")


if __name__ == "__main__":
    main()
