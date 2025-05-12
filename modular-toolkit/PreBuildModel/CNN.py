import math
import torch
import torch.nn.functional as F
from torch import nn
from ..datasets import dataloader

class CNN(nn.Module):
    def __init__(self, 
                 p: int, 
                 num_channels: int, 
                 kernel_sizes: list,    # [(h1,w1),(h2,w2)]
                 strides: list,         # [s1,s2]
                 paddings: list,        # [(ph1,pw1),(ph2,pw2)]
                 num_hiddens: list,     # [c1,c2]
                 num_outputs: int,
                 dropout_rate: float = 0.1,
                 hidden_dim: int = 256,
                 **kwargs):
        super().__init__()
        (h1, w1), (h2, w2) = kernel_sizes
        s1, s2             = strides
        (ph1, pw1), (ph2, pw2) = paddings
        c1, c2             = num_hiddens

        # 第一层卷积 + GroupNorm  
        self.conv1 = nn.Conv2d(num_channels, c1,
                               kernel_size=(h1, w1),
                               stride=s1,
                               padding=(ph1, pw1))
        self.gn1   = nn.GroupNorm(num_groups=8, num_channels=c1)

        # 第二层卷积 + GroupNorm  
        self.conv2 = nn.Conv2d(c1, c2,
                               kernel_size=(h2, w2),
                               stride=s2,
                               padding=(ph2, pw2))
        self.gn2   = nn.GroupNorm(num_groups=8, num_channels=c2)

        # 第三层 1×3 卷积 + GroupNorm  
        self.conv3 = nn.Conv2d(c2, c2,
                               kernel_size=(1, 3),
                               padding=(0, 1))           
        self.gn3   = nn.GroupNorm(num_groups=8, num_channels=c2)

        # 残差连线：将 conv1 输出升到 c2 通道
        self.res_conv = nn.Conv2d(c1, c2, kernel_size=1)

        # 全局池化 + MLP 头
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1     = nn.Linear(c2, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2     = nn.Linear(hidden_dim, num_outputs)

    def forward(self, X):
        out1 = F.gelu(self.gn1(self.conv1(X)))       # -> (batch, c1, 1, 1)
        out2 = F.gelu(self.gn2(self.conv2(out1)))    # -> (batch, c2, 1, 1)
        res  = self.res_conv(out1)                   # -> (batch, c2, 1, 1)
        out2 = out2 + res                            # 残差连接
        out3 = F.gelu(self.gn3(self.conv3(out2)))    # -> (batch, c2, 1, 1)
        feat = self.pool(out3).view(out3.size(0), -1)  # -> (batch, c2)
        h    = F.gelu(self.fc1(feat))
        h    = self.dropout(h)
        return self.fc2(h)                            # -> (batch, p)
