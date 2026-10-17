"""生成实验报告所需的测试指标、预测样本图和 UTF-8 文本。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader

from model import Residual, ResNet18


ROOT = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=224),
    torchvision.transforms.ToTensor(),
])
dataset = torchvision.datasets.MNIST(
    root=str(ROOT / "dataset"), train=False, download=False, transform=transform
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

model = ResNet18(Residual).to(device)
model.load_state_dict(torch.load(ROOT / "ResNet_model" / "best_model.pth", map_location=device, weights_only=True))
model.eval()

correct = 0
total = 0
correct_examples = []
wrong_examples = []
all_lines = []
with torch.no_grad():
    for index, (image, label) in enumerate(dataloader):
        output = model(image.to(device))
        prediction = int(output.argmax(dim=1).item())
        true_label = int(label.item())
        total += 1
        correct += int(prediction == true_label)
        all_lines.append(f"样本 {index + 1}: 预测值={prediction}, 真实值={true_label}, "
                         f"结果={'正确' if prediction == true_label else '错误'}")
        item = (image.squeeze().numpy(), prediction, true_label, index + 1)
        if prediction == true_label and len(correct_examples) < 5:
            correct_examples.append(item)
        elif prediction != true_label and len(wrong_examples) < 5:
            wrong_examples.append(item)

accuracy = correct / total
(ROOT / "test_metrics.txt").write_text(
    f"测试集样本数: {total}\n正确样本数: {correct}\n错误样本数: {total - correct}\n"
    f"最终测试准确率: {accuracy * 100:.2f}%\n\n" +
    "\n".join(all_lines[:20]) + "\n\n" +
    "报告样本（正确）:\n" + "\n".join(all_lines[item[3] - 1] for item in correct_examples) + "\n" +
    "报告样本（错误）:\n" + "\n".join(all_lines[item[3] - 1] for item in wrong_examples) + "\n",
    encoding="utf-8",
)

examples = correct_examples + wrong_examples
fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
for axis, (image, prediction, true_label, index) in zip(axes.ravel(), examples):
    axis.imshow(image, cmap="gray")
    is_correct = prediction == true_label
    axis.set_title(
        f"sample {index}\npred: {prediction}  true: {true_label}",
        color="green" if is_correct else "red",
        fontsize=10,
    )
    axis.axis("off")
fig.suptitle("MNIST test prediction examples (green=correct, red=wrong)", fontsize=14)
fig.tight_layout()
fig.savefig(ROOT / "test_prediction_samples.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(f"测试集准确率: {accuracy * 100:.2f}%")
print(f"正确: {correct}, 错误: {total - correct}")
print("已生成 test_metrics.txt 和 test_prediction_samples.png")
