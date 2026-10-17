# CDUTexperiment-1 数字识别实验

基于 MNIST 数据集和自定义 ResNet18 的手写数字识别实验。

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

## 模型架构

输入图像先 Resize 到 224×224，然后经过初始卷积模块、四个残差阶段、自适应平均池化和全连接层，输出 10 个数字类别。

## GitHub 地址

本地工程目前尚未绑定远程 GitHub 仓库。创建仓库后，将以下占位地址替换成真实地址：

- 源码仓库：`https://github.com/<用户名>/CDUTexperiment-1`
- 预训练模型：`https://github.com/<用户名>/CDUTexperiment-1/releases/download/v1.0/best_model.pth`

建议将 `ResNet_model/best_model.pth` 上传到 GitHub Release v1.0 的附件中，不建议将大模型文件直接放入普通 Git 提交。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `model.py` | ResNet18 和残差块定义 |
| `model_train.py` | 数据加载、训练、验证和曲线生成 |
| `model_test.py` | 测试集评估和逐样本预测 |
| `generate_report_evidence.py` | 生成报告指标和预测样本图 |
| `ResNet_model/best_model.pth` | 最佳模型权重 |
| `数字识别实验报告.docx` | 实验报告 |
