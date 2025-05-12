import math
import torch
import torch.nn.functional as F
from torch import nn




class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
        
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=self.batch_first, dropout=dropout)

    def run(self, Q, K, V):
        seq_len = Q.shape[1]
        attn_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(Q.device)
        output, attn_weight = self.mha(Q, K, V, attn_mask=attn_mask)
        return output, attn_weight

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
        
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.num_hiddens = num_hiddens
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        self.nun_hiddens = num_hiddens
        X = torch.arange(max_len , dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X) #取偶索引赋值给P
        self.P[:, :, 1::2] = torch.cos(X) #取奇索引赋值给P
        
    def forward(self, X):
        X = X * math.sqrt(self.num_hiddens) + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)



class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, norm_shape, ffn_num_hiddens, num_heads, dropout, **kwargs):
        super().__init__(**kwargs)
        self.attention1 = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(embed_dim, ffn_num_hiddens, embed_dim)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X):
        X2, _ = self.attention1.run(X, X, X)
        Y = self.addnorm1(X, X2)
        Y2 = self.ffn(Y)
        Z = self.addnorm2(Y, Y2)
        return Z


class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, embed_dim, norm_shape, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f'decoder_block_{i}',
                                 DecoderBlock(embed_dim, norm_shape, ffn_num_hiddens, num_heads, dropout))
        self.dense = nn.Linear(embed_dim, vocab_size)
            
    def forward(self, X):
        X = self.embedding(X)
        X = self.pos_encoding(X)
        X = self.blks(X)
        return self.dense(X)