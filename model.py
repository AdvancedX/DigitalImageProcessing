import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义逻辑回归模型
class LR(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out
