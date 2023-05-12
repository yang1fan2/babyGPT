import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        mask = torch.tril(torch.ones(C, C), diagonal=-1)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_head, C, C)
        att = nn.Softmax(att * mask, dim=3)
        output = att @ V # [B, n_head, C, dk]
        output = output.transpose(1, 2).reshape(B, C, D) # [B, C, D]
        output = self.att_dropout(self.output_project(output))
        return output

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        pass
