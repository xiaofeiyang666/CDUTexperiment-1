# GitHub 发布清单

当前工程没有 Git 远程仓库。按下面步骤上传后，即可获得报告要求的两个真实链接。

## 1 创建仓库并上传源码

在 GitHub 新建仓库 `CDUTexperiment-1`，然后在工程目录执行：

```powershell
cd F:\深度学习实验\CDUTexperiment-1
git init
git add model.py model_train.py model_test.py generate_report_evidence.py README.md dataset ResNet_model/best_model.pth
git commit -m "Add MNIST ResNet18 experiment"
git branch -M main
git remote add origin https://github.com/<用户名>/CDUTexperiment-1.git
git push -u origin main
```

如果模型文件超过 GitHub 普通文件限制，使用 Git LFS：

```powershell
git lfs install
git lfs track "ResNet_model/best_model.pth"
git add .gitattributes ResNet_model/best_model.pth
git commit -m "Track pretrained model with Git LFS"
git push
```

## 2 发布预训练模型

在 GitHub 仓库的 Releases 页面创建 `v1.0`，上传 `ResNet_model/best_model.pth`。模型下载链接通常为：

```text
https://github.com/<用户名>/CDUTexperiment-1/releases/download/v1.0/best_model.pth
```

## 3 替换报告中的占位链接

将 `数字识别实验报告.docx` 和 `README.md` 中的两个占位地址替换为实际用户名和仓库地址即可。
