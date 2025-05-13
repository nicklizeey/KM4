
import math
import torch
import torch.nn.functional as F
from torch import nn

class CNN(nn.Module):
    def __init__(self, p, num_classes, emb_dim=64, c1=64, c2=128, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(2*p, emb_dim)  # p 是整数最大值
        
        # 输入变成 (batch, 1, 2, emb_dim)，看作图像
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(c2)

        self.pool  = nn.AdaptiveAvgPool2d((1,1))
        self.fc1   = nn.Linear(c2, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # x: (batch, 2)
        x = self.embedding(x)  # → (batch, 2, emb_dim)
        x = x.unsqueeze(1)     # → (batch, 1, 2, emb_dim)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
