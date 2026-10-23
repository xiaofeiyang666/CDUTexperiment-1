"""Train a ResNet18 cat/dog classifier using the Kaggle split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

from data_utils import stratified_train_val_indices


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset" / "cats_dogs" / "dc_2000"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "报告" / "实验二证据"
SEED = 42
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 32
EPOCHS = 8


class IndexedImageFolder(Dataset):
    """Apply a transform to an ImageFolder without changing its labels."""

    def __init__(self, image_folder: datasets.ImageFolder, indices: list[int], transform):
        self.image_folder = image_folder
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        source_index = self.indices[item]
        path, label = self.image_folder.samples[source_index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_model(device: torch.device) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 2)
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model.to(device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    grad_context = torch.enable_grad if training else torch.no_grad
    with grad_context():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += logits.argmax(dim=1).eq(labels).sum().item()
            total += batch_size
    return total_loss / total, correct / total


def verify_images(image_folder: datasets.ImageFolder) -> None:
    invalid = []
    for path, _ in image_folder.samples:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            invalid.append(path)
    if invalid:
        sample = "\n".join(invalid[:5])
        raise RuntimeError(f"发现 {len(invalid)} 张无法读取的训练图片，例如：\n{sample}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("epochs、batch-size 必须为正，workers 不能小于 0")
    train_root = args.data_root / "train"
    if not train_root.is_dir() or not (args.data_root / "test").is_dir():
        parser.error(f"数据目录应包含 train 与 test 子目录：{args.data_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_seed(args.seed)
    train_folder = datasets.ImageFolder(train_root)
    test_folder = datasets.ImageFolder(args.data_root / "test")
    if train_folder.class_to_idx != {"cat": 0, "dog": 1}:
        raise ValueError(f"类别目录应为 cat、dog，当前映射：{train_folder.class_to_idx}")
    if test_folder.class_to_idx != train_folder.class_to_idx:
        raise ValueError("训练集和测试集类别映射不一致")
    if args.verify_images:
        print("正在检查所有训练图片 ...", flush=True)
        verify_images(train_folder)
        verify_images(test_folder)
        print("训练集与测试集图片完整性检查通过。", flush=True)

    labels = [label for _, label in train_folder.samples]
    train_indices, val_indices = stratified_train_val_indices(labels, seed=args.seed, train_ratio=0.8)
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    if sorted(set(train_labels)) != [0, 1] or sorted(set(val_labels)) != [0, 1]:
        raise ValueError("训练集和验证集都必须包含猫、狗两类")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.78, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        IndexedImageFolder(train_folder, train_indices, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        IndexedImageFolder(train_folder, val_indices, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    weights_sha256 = sha256_file(weights_path) if weights_path.is_file() else "loaded-by-torchvision-cache"

    split_manifest = {
        "seed": args.seed,
        "class_to_idx": train_folder.class_to_idx,
        "train": [
            {"path": Path(train_folder.samples[i][0]).relative_to(args.data_root).as_posix(), "label": labels[i]}
            for i in train_indices
        ],
        "validation": [
            {"path": Path(train_folder.samples[i][0]).relative_to(args.data_root).as_posix(), "label": labels[i]}
            for i in val_indices
        ],
        "test": [
            {"path": Path(path).relative_to(args.data_root).as_posix(), "label": label}
            for path, label in test_folder.samples
        ],
    }
    with (args.output_dir / "split_manifest.json").open("w", encoding="utf-8") as target:
        json.dump(split_manifest, target, ensure_ascii=False, indent=2)

    history_path = args.output_dir / "training_log.csv"
    checkpoint_path = args.output_dir / "best_model.pth"
    best_val_accuracy = -1.0
    best_epoch = 0
    start = datetime.now()
    header = (
        f"设备: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} | "
        f"类别: {train_folder.class_to_idx} | 数据: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_folder)} | seed={args.seed}"
    )
    text_log = (args.output_dir / "training_log.txt").open("w", encoding="utf-8")

    def emit(message: str) -> None:
        print(message, flush=True)
        text_log.write(message + "\n")
        text_log.flush()

    emit(header)
    emit(
        f"Epochs={args.epochs} batch_size={args.batch_size} | "
        "ResNet18 ImageNet1K_V1 (only layer4 + fc trainable) | "
        "Adam lr(layer4)=0.0001 lr(fc)=0.001 | CrossEntropyLoss"
    )
    emit("Epoch | train_loss | train_acc | val_loss | val_acc | seconds")

    with history_path.open("w", newline="", encoding="utf-8-sig") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"],
        )
        writer.writeheader()
        log_file.flush()
        for epoch in range(1, args.epochs + 1):
            epoch_start = datetime.now()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            seconds = (datetime.now() - epoch_start).total_seconds()
            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
                "seconds": f"{seconds:.1f}",
            }
            writer.writerow(row)
            log_file.flush()
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_epoch = epoch
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "class_to_idx": train_folder.class_to_idx,
                        "architecture": "torchvision.models.resnet18",
                        "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1",
                        "pretrained_sha256": weights_sha256,
                        "best_epoch": best_epoch,
                        "best_val_accuracy": best_val_accuracy,
                        "image_size": IMAGE_SIZE,
                        "mean": MEAN,
                        "std": STD,
                        "seed": args.seed,
                    },
                    checkpoint_path,
                )
            emit(
                f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>9.2%} | "
                f"{val_loss:>8.4f} | {val_acc:>7.2%} | {seconds:>7.1f}"
            )
    metadata = {
        "started_at": start.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "train_images": len(train_indices),
        "validation_images": len(val_indices),
        "test_images": len(test_folder),
        "class_to_idx": train_folder.class_to_idx,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "pretrained_sha256": weights_sha256,
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as target:
        json.dump(metadata, target, ensure_ascii=False, indent=2)
    emit(
        f"训练完成；最佳验证准确率 {best_val_accuracy:.2%}（epoch {best_epoch}）。"
        f"模型已保存：{checkpoint_path}"
    )
    text_log.close()


if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    main()
