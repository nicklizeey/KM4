import math
import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_hiddens, **kwargs):
        super().__init__(**kwargs)
        self.vocab = vocab_size
        self.embedding = nn.Embedding(self.vocab, embedding_dim)
        self.num_hiddens = num_hiddens
        self.model = nn.Sequential()
        for i in range(len(num_hiddens)):
            if i == 0:
                self.model.add_module(f"linear{i}", nn.Linear(embedding_dim * 4, num_hiddens[i]))
                self.model.add_module(f"relu{i}", nn.ReLU())
            else:
                self.model.add_module(f"linear{i}", nn.Linear(num_hiddens[i-1], num_hiddens[i]))
                self.model.add_module(f"relu{i}", nn.ReLU())
        self.model.add_module(f"linear{len(num_hiddens)}", nn.Linear(num_hiddens[-1], self.vocab))
            
        
    def forward(self, X):
        X = self.embedding(X)
        X = X.reshape(X.shape[0], -1)
        X = self.model(X)
        return X