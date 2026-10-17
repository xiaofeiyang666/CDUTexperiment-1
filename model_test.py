import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as Data

from model import Residual,ResNet18


def test_val_data_process():
    test_data = torchvision.datasets.MNIST(root='./dataset', train=False, download=False,
                                                   transform=torchvision.transforms.Compose
                                                   ([torchvision.transforms.Resize(size=224),
                                                     torchvision.transforms.ToTensor()]
                                                    ))
    test_data_len = len(test_data)
    print(test_data_len)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    return test_dataloader

def test_model_process(model,test_dataloader):
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    #初始化参数
    test_corrects = 0
    test_num = 0
    print(f"使用设备: {device}")
    with torch.no_grad():
        for test_data_x,test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()
            output = model(test_data_x)
            pre_label = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_label == test_data_y.data)
            test_num += test_data_x.size(0)#将所有的测试样本进行累加

    test_acc = test_corrects.double() / test_num #计算准确率
    print("测试的准确率为：{}%".format(test_acc*100))

if __name__ == '__main__':
    #加载模型
    model = ResNet18(Residual)
    model.load_state_dict(torch.load("./ResNet_model/best_model.pth",weights_only=False))
    #加载测试数据
    test_dataloader = test_val_data_process()
    #加载模型测试的函数
    test_model_process(model,test_dataloader)
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_label = torch.argmax(output, dim=1)
            result=pre_label.item()
            label=b_y.item()
            print("预测值：",classes[result],'--------',"真实值",classes[label])
