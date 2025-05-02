import math
import torch
import torch.nn.functional as F
from torch import nn
from .TransformerDecodeOnly import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, ffn_num_hiddens, num_heads, num_layers, dropout, device, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)

        self.model = nn.Transformer(d_model = embed_dim, nhead = num_heads, num_encoder_layers = num_layers,
                                    num_decoder_layers = num_layers, dim_feedforward = ffn_num_hiddens,
                                    dropout = dropout, norm_first = False,
                                    batch_first = True, device = device)
        
        self.dense = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        src = self.pos_encoding(self.embedding(src))
        tgt = self.pos_encoding(self.embedding(tgt))
        tgt_seq_len = tgt.shape[1]
        tgt_mask = Transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        out = self.model(src, tgt, tgt_mask=tgt_mask)
        out = self.dense(out)
        return out
    


if __name__ == '__main__':
    model = Transformer(99, 128, 256, 2, 2, 0.01, 'cuda')
    print(model)