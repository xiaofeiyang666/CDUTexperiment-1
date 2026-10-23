import copy
import time
import pandas as pd
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as Data
from model import ResNet18, Residual
import torch
import matplotlib
import numpy as np
import os
matplotlib.use('Agg')  # 保存图片，避免训练脚本等待图形窗口
import matplotlib.pyplot as plt

def train_val_data_process():
    train_data = torchvision.datasets.MNIST(root='./dataset', train=True, download=True,
                                            transform=torchvision.transforms.Compose
                                            ([torchvision.transforms.Resize(size=224),
                                              torchvision.transforms.ToTensor()]
                                             ))
    # 计算数据集长度
    train_data_len = len(train_data)
    print(train_data_len)
    # 划分数据集和训练集
    train_data, val_data = Data.random_split(train_data,
                                             lengths=[round(0.8 * train_data_len), round(0.2 * train_data_len)])
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)
    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"使用设备: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    # 复制当前模型参数,wts=weigths
    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0  # 最高准确度
    train_loss_all = []  # 训练集损失列表
    val_loss_all = []  # 验证集损失列表
    train_acc_all = []  # 训练集准确度列表
    val_acc_all = []  # 验证集准确度列表
    since = time.time()  # 当前时间

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-' * 10)

        # 初始化参数
        train_loss = 0.0
        train_corrents = 0.0
        val_loss = 0.0
        val_corrents = 0.0
        train_num = 0
        val_num = 0

        model.train()  # 设置模型为训练模式，有Dropout层和BatchNorm层才需调用
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            pre_label = torch.argmax(output, dim=1)  # 查找每行最大值对应的行标
            loss = loss_fn(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)  # 对损失函数进行累加
            train_corrents += torch.sum(pre_label == b_y.data)  # 如果预测正确，则精确度＋1
            train_num += b_x.size(0)  # 当前用于训练的样本数量

        model.eval()  # 设置模型为验证模式，有Dropout层和BatchNorm层才需调用
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(val_dataloader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                output = model(b_x)
                pre_label = torch.argmax(output, dim=1)
                loss = loss_fn(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrents += torch.sum(pre_label == b_y.data)  # 如果预测正确，则精确度＋1
                val_num += b_x.size(0)  # 当前用于验证的样本数量

        train_loss_all.append(train_loss / train_num)  # 计算并保存每一次训练集迭代的loss值和准确率
        train_acc_all.append(train_corrents.double().item() / train_num)  # 计算并保存训练集准确率
        val_loss_all.append(val_loss / val_num)  # 计算并保存每一次验证集迭代的loss值和准确率
        val_acc_all.append(val_corrents.double().item() / val_num)  # 计算并保存验证集准确率

        print("{} Train loss: {:.4f},Train Acc: {:.4f}".format(epoch + 1, train_loss_all[-1], train_acc_all[-1]))
        print("{} Val loss: {:.4f},Val Acc: {:.4f}".format(epoch + 1, val_loss_all[-1], val_acc_all[-1]))

        # 寻找最高准确度
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]  # 保存当前的最高准确度
            best_model_wts = copy.deepcopy(model.state_dict())  # 保存当前的最高准确度的模型参数
        # 训练耗时
        time_use = time.time() - since
        print("训练耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数，保存最优参数的模型
    torch.save(best_model_wts, "./ResNet_model/best_model.pth")

    train_process = pd.DataFrame(data={"epoch": range(num_epoch),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})
    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "bs-", label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=180)
    plt.close()

def show_misclassified_images(model, val_dataloader, num_images=20):
    #显示预测错误的图片
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    misclassified_images=[]
    misclassified_labels=[]
    misclassified_predictions=[]

    with torch.no_grad():
        for batch_x, batch_y in val_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            # 找出预测错误的样本
            wrong_indices = (predicted != batch_y).nonzero(as_tuple=True)[0]
            # 当找到的预测错误图片数量达到 num_images 时停止收集
            for idx in wrong_indices:
                if len(misclassified_images) < num_images:# 检查是否已达到最大数量
                    # 收集错误样本，将图像从GPU转到CPU并转换为numpy
                    image = batch_x[idx].cpu().squeeze().numpy()
                    true_label = batch_y[idx].cpu().item()
                    pred_label = predicted[idx].cpu().item()
                    misclassified_images.append(image)
                    misclassified_labels.append(true_label)
                    misclassified_predictions.append(pred_label)
                else:
                    break
            if len(misclassified_images) >= num_images:
                break

    # 显示预测错误的图片
    if misclassified_images:
        print(f"\n显示 {len(misclassified_images)} 个预测错误的样本:")

        # 计算网格大小
        cols = 5
        rows = (len(misclassified_images) + cols - 1) // cols

        plt.figure(figsize=(15, 3 * rows))
        for i in range(len(misclassified_images)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(misclassified_images[i], cmap='gray')
            plt.title(f'True: {misclassified_labels[i]}, Pred: {misclassified_predictions[i]}',
                      color='red', fontsize=10)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig("misclassified_samples.png", dpi=180)
        plt.close()

        # 打印统计信息
        print(f"\n预测错误统计:")
        for i in range(10):  # MNIST有0-9共10个类别
            true_count = misclassified_labels.count(i)
            if true_count > 0:
                print(f"真实标签为 {i} 但被预测错误的样本数: {true_count}")
    else:
        print("没有找到预测错误的样本！")

def analyze_predictions(model, val_dataloader):
    """
    在验证集上进行预测并分析结果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_predictions = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for batch_x, batch_y in val_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)

            # 保存预测结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_images.extend(batch_x.cpu().numpy())

    # 计算准确率
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_predictions == all_labels)

    print(f"\n验证集总体准确率: {accuracy:.4f}")
    print(f"总样本数: {len(all_labels)}")
    print(f"预测正确的样本数: {np.sum(all_predictions == all_labels)}")
    print(f"预测错误的样本数: {np.sum(all_predictions != all_labels)}")

    return all_predictions, all_labels, all_images


if __name__ == '__main__':
    # 将模型实例化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    resnet = ResNet18(Residual)
    resnet = resnet.to(device)
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(resnet, train_dataloader, val_dataloader, num_epoch=5)
    train_process.to_csv("training_log.csv", index=False, encoding="utf-8-sig")
    with open("training_log.txt", "w", encoding="utf-8") as log_file:
        for _, row in train_process.iterrows():
            log_file.write(
                f"Epoch {int(row['epoch']) + 1}/5 | "
                f"Train loss: {row['train_loss_all']:.4f} | "
                f"Val loss: {row['val_loss_all']:.4f} | "
                f"Val Acc: {row['val_acc_all']:.4f}\n"
            )
    matplot_acc_loss(train_process)

    # ========== 新增代码：加载最佳模型并进行预测分析 ==========
    print("\n" + "=" * 50)
    print("开始分析预测结果...")
    print("=" * 50)
    # 加载训练好的最佳模型
    resnet.load_state_dict(torch.load("./ResNet_model/best_model.pth"))
    resnet.eval()
    # 分析预测结果
    all_predictions, all_labels, all_images = analyze_predictions(resnet, val_dataloader)
    # 显示预测错误的图片
    show_misclassified_images(resnet, val_dataloader, num_images=20)
