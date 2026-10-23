# CDUTexperiment-1 深度学习实验

本仓库只保留源码，两个实验使用独立的顶级文件夹，文件不会相互覆盖：

```text
experiment1_digit_recognition/  # 实验一：数字识别
experiment2_catdog/             # 实验二：猫狗分类
```

## 运行环境

```text
Python（Conda env）
PyTorch 2.4.1
torchvision 0.20.0
numpy 1.24.3
pandas 2.0.3
matplotlib 3.7.2
```

## 实验一：数字识别

```powershell
cd F:\深度学习实验\CDUTexperiment-1\experiment1_digit_recognition
F:\Anoconda\envs\env\python.exe model_train.py
F:\Anoconda\envs\env\python.exe model_test.py
F:\Anoconda\envs\env\python.exe generate_report_evidence.py
```

脚本会在当前实验目录生成训练日志、曲线、测试指标和预测样本图。MNIST 数据集和模型权重保存在本地，不上传到仓库。

## 实验二：猫狗分类

实验二使用 2000 张训练图片和 1000 张独立测试图片，从训练集分出 1600 张训练图片和 400 张验证图片，微调预训练 ResNet18 的 `layer4` 和分类层。

```powershell
cd F:\深度学习实验\CDUTexperiment-1\experiment2_catdog
F:\Anoconda\envs\env\python.exe train.py --epochs 8 --batch-size 32 --verify-images
F:\Anoconda\envs\env\python.exe evaluate.py --batch-size 64
F:\Anoconda\envs\env\python.exe -m unittest discover -s tests -v
```

数据集需要放在 `F:\深度学习实验\CDUTexperiment-1\dataset\cats_dogs\dc_2000`，该目录不上传。预训练 ResNet18 权重由 torchvision 自动下载：
`https://download.pytorch.org/models/resnet18-f37072fd.pth`

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `experiment1_digit_recognition/model.py` | 数字识别 ResNet18 和残差块定义 |
| `experiment1_digit_recognition/model_train.py` | 数字识别训练、验证和曲线生成 |
| `experiment1_digit_recognition/model_test.py` | 数字识别测试集评估 |
| `experiment1_digit_recognition/generate_report_evidence.py` | 数字识别报告证据生成 |
| `experiment1_digit_recognition/create_report.py` | 数字识别报告生成脚本 |
| `experiment2_catdog/data_utils.py` | 数据切分和二分类指标 |
| `experiment2_catdog/train.py` | 猫狗分类训练和模型保存 |
| `experiment2_catdog/evaluate.py` | 猫狗分类测试、曲线和证据图生成 |
| `experiment2_catdog/tests/test_data_metrics.py` | 数据切分与指标单元测试 |

## GitHub 地址

源码仓库：`https://github.com/xiaofeiyang666/CDUTexperiment-1`
