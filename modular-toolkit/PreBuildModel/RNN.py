import math
import torch
import torch.nn.functional as F
from torch import nn


class RNN(nn.Module):
    def __init__(self, p, embedding_dim, hidden_size, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.embedding = nn.Embedding(p + 2, embedding_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义输入到隐藏状态的权重矩阵
        self.W_xh = nn.Parameter(torch.randn(embedding_dim, hidden_size))
        # 定义隐藏状态到隐藏状态的权重矩阵
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        # 定义隐藏状态到输出的权重矩阵
        self.W_hy = nn.Parameter(torch.randn(hidden_size, p + 2))
        # 定义输入到隐藏状态的偏置
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        # 定义隐藏状态到输出的偏置
        self.b_y = nn.Parameter(torch.zeros(p + 2))

    def forward(self, X):
        X = self.embedding(X)
        batch_size, seq_length, _ = X.shape
        # 初始化多层隐藏状态
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(X.device)
        for t in range(seq_length):
            x_t = X[:, t, :]
            new_h = torch.tanh(torch.matmul(x_t, self.W_xh) + self.b_h)
            for layer in range(self.num_layers):
                new_h = torch.tanh(torch.matmul(new_h, self.W_hh) + self.b_h)
            h[layer] = new_h

        out = torch.matmul(new_h, self.W_hy) + self.b_y
        return out
