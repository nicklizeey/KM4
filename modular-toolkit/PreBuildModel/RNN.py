import torch
from torch import nn


class RNN(nn.Module):

    def __init__(self, p: int,
                       embedding_dim: int,
                       hidden_size: int,
                       num_layers: int = 1,
                       **kwargs):
        super().__init__(**kwargs)

    
        self.p             = p                
        self.vocab_size    = p + 2           
        self.embedding_dim = embedding_dim
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers    


        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)

        k = hidden_size ** 0.5
        self.W_xh = nn.Parameter(torch.randn(embedding_dim, hidden_size) / k)
        self.W_hh = nn.Parameter(torch.randn(hidden_size,  hidden_size)  / k)
        self.b_h  = nn.Parameter(torch.zeros(hidden_size))


        self.W_hy = nn.Parameter(torch.randn(hidden_size, self.vocab_size) / k)
        self.b_y  = nn.Parameter(torch.zeros(self.vocab_size))


    def forward(self, X: torch.Tensor) -> torch.Tensor:

        X = self.embedding(X)
        B, T, _ = X.shape

        h = torch.zeros(B, self.hidden_size, device=X.device)
        for t in range(T):
            x_t = X[:, t, :]
            h   = torch.tanh(x_t @ self.W_xh + h @ self.W_hh + self.b_h)

        logits = h @ self.W_hy + self.b_y   
        return logits
