# CDUTexperiment-1 深度学习实验

本仓库包含数字识别和猫狗分类两个 PyTorch 实验的源码。

## 实验结果

- 训练轮数：5 epochs
- 最佳验证准确率：99.10%
- 测试集准确率：99.14%（9914/10000）
- 运行设备：NVIDIA GeForce RTX 4060 Laptop GPU，CUDA

## 运行环境

```text
Python（Conda env）
PyTorch 2.4.1
torchvision 0.20.0
numpy 1.24.3
pandas 2.0.3
matplotlib 3.7.2
```

## 使用方法

```powershell
cd F:\深度学习实验\CDUTexperiment-1
F:\Anoconda\envs\env\python.exe model_train.py
F:\Anoconda\envs\env\python.exe model_test.py
F:\Anoconda\envs\env\python.exe generate_report_evidence.py
```

训练脚本会生成 `training_log.txt`、`training_log.csv`、`training_curves.png` 和 `misclassified_samples.png`。测试证据脚本会生成 `test_metrics.txt` 和 `test_prediction_samples.png`。

## 实验二 猫狗分类

实验二使用 2000 张训练图片和 1000 张独立测试图片。训练脚本会从训练集分出 1600 张训练图片和 400 张验证图片，微调预训练 ResNet18 的 `layer4` 和分类层。

```powershell
cd F:\深度学习实验\CDUTexperiment-1
F:\Anoconda\envs\env\python.exe _code\experiment2_catdog\train.py --epochs 8 --batch-size 32 --verify-images
F:\Anoconda\envs\env\python.exe _code\experiment2_catdog\evaluate.py --batch-size 64
```

实验二脚本会生成训练日志、训练曲线、混淆矩阵、逐样本预测表和预测示例图。数据集需要放在 `dataset/cats_dogs/dc_2000`，该目录不上传到仓库。

预训练 ResNet18 权重由 torchvision 自动下载：
`https://download.pytorch.org/models/resnet18-f37072fd.pth`

## 模型架构

输入图像先 Resize 到 224×224，然后经过初始卷积模块、四个残差阶段、自适应平均池化和全连接层，输出 10 个数字类别。

## GitHub 地址

源码仓库：`https://github.com/xiaofeiyang666/CDUTexperiment-1`

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `model.py` | ResNet18 和残差块定义 |
| `model_train.py` | 数据加载、训练、验证和曲线生成 |
| `model_test.py` | 测试集评估和逐样本预测 |
| `generate_report_evidence.py` | 生成报告指标和预测样本图 |
| `_code/experiment2_catdog/data_utils.py` | 数据划分和分类指标 |
| `_code/experiment2_catdog/train.py` | 猫狗分类训练和模型保存 |
| `_code/experiment2_catdog/evaluate.py` | 猫狗分类测试和证据图生成 |
