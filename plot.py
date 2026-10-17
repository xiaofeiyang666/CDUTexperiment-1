import numpy as np
import torchvision
import matplotlib
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

train_data=torchvision.datasets.FashionMNIST(root='./dataset', train=True,download=True,
                                             transform=torchvision.transforms.Compose
                                                 ([torchvision.transforms.Resize(size=224),
                                                   torchvision.transforms.ToTensor()]
                                             ))
train_data_len=len(train_data)
print(train_data_len)

train_dataloader=DataLoader(train_data, batch_size=64, shuffle=True)

#获得一个Batch的数据
for step,(b_x,b_y) in enumerate(train_dataloader):
    if step>0:
        break
batch_x=b_x.squeeze().numpy() # 将四维张量移除第一维，并转换成Numpy数组
batch_y=b_y.numpy() # 将张量转换成Numpy数组
class_label=train_data.classes # 训练集的标签
print(class_label)

#可视化一个Batch的图像
plt.figure(figsize=(12,5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4,16,ii+1)
    plt.imshow(batch_x[ii,:,:],cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]],size=10)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05)
plt.show()
