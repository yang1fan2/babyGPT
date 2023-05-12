import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#TODO
# 1. scale the weight of residual layer by 1 / sqrt(n_layers)
# 2. NewGELU
# 3. weight initialization
# 4. weight deay for linear (not layernorm and embedding)
# 5. bug fix in position embedding
# 6. generate

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_embed):
        super().__init__()
        self.n_head = n_head
        self.n_embed = n_embed
        self.head_embed = n_embed // n_head
        self.att_proj = nn.linear(self.head_embed, 3*self.head_embed)
        self.output_project = nn.linear(n_embed, n_embed)
        self.att_dropout = nn.Dropout(0.1)

    def forward(self, x): #  x = [B, C, D]
        B = x.size(dim=0)
        C = x.size(dim=1)
        D = x.size(dim=2)
        assert D % self.n_head == 0
        x = x.reshape((B, C, self.n_head, D // self.n_head)).transpose(1, 2) # [B, n_head, C, D // head = head_embed]
        x = self.att_proj(x) # [B, n_head, C, 3*head_embed]
        Q, K, V = torch.split(x, self.head_embed, dim = 3) # [B, n_head, C, head_embed]
        att = Q @ K.transpose(2, 3) / math.sqrt(self.head_embed) # [B, n_head, C, C]
        mask = torch.tril(torch.ones(C, C), diagonal=-1) # TODO register as buffer
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_head, C, C) # TODO optimize this
        att = nn.Softmax(att * mask, dim=3)
        output = att @ V # [B, n_head, C, dk]
        output = output.transpose(1, 2).reshape(B, C, D) # [B, C, D]
        output = self.att_dropout(self.output_project(output))
        return output

class TransformerBlock(nn.Module):
    def __init__(self, n_head, n_embed):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_embed)
        self.attention = MultiHeadAttention(n_head, n_embed)

    def forward(self, x):
        x = self.layer_norm(x)
        return x + self.attention(x) # TODO one more MLP

class Transformer(nn.Module):
    def __init__(self, n_layers, n_head, n_embed, n_vocab, context_size):
        super().__init__()
        # position embedding
        self.blocks = nn.ModuleList([TransformerBlock(n_head, n_embed) for i in range(n_layers)])
        self.vocab_embed = nn.Embedding(n_vocab, n_embed)
        self.softmax_proj = nn.Linear(n_embed, n_vocab)
        self.layer_norm = nn.LayerNorm(n_embed)
        position = torch.arange(0, context_size).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_embed, 2).float() * -(math.log(10000.0) / n_embed))
        pe = torch.zeros(context_size, n_embed)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) # [C, D]
        pe = pe.unsqueeze(0) # [1, C, D]
        self.register_buffer('pe', pe)
        
    def forward(self, x): # [B, C]
        x = self.vocab_embed(x) # [B, C, D]
        x = x + self.pe
        # position
        for i, l in enumerate(self.blocks):
            x = l(x)
        x = self.layer_norm(x)
        x = self.softmax_proj(x) # [B, C, vocab]
        x = nn.Softmax(x, dim=-1)
        return x
