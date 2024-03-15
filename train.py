import torch
import numpy as np
import torchvision
import os
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F

from model import LR

# 特征缩放 数据集转换为tensor格式
transform = transforms.Compose([transforms.Resize((320, 320)),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 数据集路径
train_data_dir = r'E:\Python\pythonProject\train'
test_data_dir = r'E:\Python\pythonProject\val-1\val'

batch_size = 4
num_epochs = 10
num_classes = 10
learning_rate = 0.01

# 获取数据集
train_dataset = datasets.ImageFolder(root = train_data_dir, transform = transform)
test_dataset = datasets.ImageFolder(root = test_data_dir, transform = transform)

# 输出类别标签与数字的映射结果
for class_idx, class_label in enumerate(train_dataset.classes):
    print(f'Class Label: {class_label}, Numeric Label: {class_idx}')
# Result:
#     Class Label: bus, Numeric Label: 0
#     Class Label: fuel, Numeric Label: 1
#     Class Label: mpv, Numeric Label: 2
#     Class Label: pickup, Numeric Label: 3
#     Class Label: sedan, Numeric Label: 4
#     Class Label: suv, Numeric Label: 5
#     Class Label: truck, Numeric Label: 6

# 数据集加载
# 数据加载器（dataloader ）实例化一个dataset后，然后用Dataloader包起来，即载入数据集。shuffle=True即打乱数据集，打乱训练集进行训练，而对测试集进行顺序测试。
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 参数设置
input_size = 102400 * 3
total_step = len(train_loader)

# 定义模型
LR_model = LR(input_size, num_classes)

# 定义逻辑回归的损失函数，采用nn.CrossEntropyLoss(),nn.CrossEntropyLoss()内部集成了softmax函数
criterion = nn.CrossEntropyLoss(reduction='mean')

# 定义优化器
# optimizer = torch.optim.SGD(LR_model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(LR_model.parameters(), lr=learning_rate)


# 训练模型-------------------------------------------------------------------------------------------------------------------------
# 定义保存路径和文件名的格式
save_path_format = 'runs/model_epoch_{}.ckpt'
os.makedirs('runs', exist_ok=True)
for epoch in range(num_epochs):
    LR_model.train()
    for i, (images, labels) in enumerate(train_loader):
        
        # 将图像序列转换至大小为 (batch_size, input_size)
        images = images.reshape(-1, 320 * 320 * 3)

        # forward
        y_pred = LR_model(images)
        # print(y_pred.size())
        # print(labels.size())
        loss = criterion(y_pred, labels)

        # backward()
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        if (i % (total_step-1) == 0):
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step,
                                                                     loss.item()))

            # 验证模型---------------------------------------------------------
            # PyTorch 默认每一次前向传播都会计算梯度
            with torch.no_grad():
                correct = 0
                total = 0
                val_loss = 0
                for images, labels in test_loader:
                    images = images.reshape(-1, 320 * 320 * 3)
                    outputs = LR_model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    # torch.max的输出：out (tuple, optional) – the result tuple of two output tensors (max, max_indices)
                    max, predicted = torch.max(outputs.data, 1)
                    #print(max.data)
                    #print(predicted)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()

                print('Accuracy of the model on test dataset: {} %'.format(100 * correct / total))
                # print('Average val loss of the model on test dataset: {} '.format(val_loss / len(test_loader)))

            ## 保存模型
            save_path = save_path_format.format(epoch + 1)
            torch.save(LR_model.state_dict(), save_path)
            # torch.save(LR_model.state_dict(), 'model.ckpt')
            
print("Training finished.")
