import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        cdf = 0.5 * (1.0 + torch.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
        return x * cdf

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, device):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.head_embed = d_model // n_head
        self.att_proj = nn.Linear(d_model, 3*d_model)
        self.residual_proj = nn.Linear(d_model, d_model)
        self.att_dropout = nn.Dropout(0.1)
        self.device = device    

    def forward(self, x): #  x = [B, C, D]
        B = x.size(dim=0)
        C = x.size(dim=1)
        D = x.size(dim=2)
        assert D % self.n_head == 0
        x = self.att_proj(x) # [B, C, 3*D]
        Q, K, V = torch.split(x, D, dim = 2) # [B, C, D]
        Q = Q.view(B, C, self.n_head, self.head_embed).transpose(1, 2) # [B, n_head, C, D // n_head]
        K = K.view(B, C, self.n_head, self.head_embed).transpose(1, 2)
        V = V.view(B, C, self.n_head, self.head_embed).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_embed)
        att = (Q @ K.transpose(2, 3)) * scale  # [B, n_head, C, C]
        mask = torch.tril(torch.ones(C, C), diagonal=-1).to(self.device)  # TODO register as buffer
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_head, C, C) # TODO optimize this
        att = F.softmax(att * mask, dim=3)
        output = att @ V # [B, n_head, C, dk]
        output = output.transpose(1, 2).reshape(B, C, D) # [B, C, D]
        output = self.att_dropout(self.residual_proj(output))
        return output

class TransformerBlock(nn.Module):
    def __init__(self, n_head, d_model, device):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(n_head, d_model, device)
        self.linear1 = nn.Linear(d_model, d_model*4)
        self.residual_proj = nn.Linear(d_model*4, d_model)
        self.dropout = nn.Dropout(0.1)
        self.gelu = GELU()

    def forward(self, x):
        x = self.layer_norm1(x)
        x = x + self.attention(x)
        x = self.layer_norm2(x)
        x = x + self.residual_proj(self.gelu(self.linear1(x)))
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, n_layers, n_head, d_model, n_vocab, context_size, device):
        super().__init__()
        # position embedding
        self.blocks = nn.ModuleList([TransformerBlock(n_head, d_model, device) for i in range(n_layers)])
        self.vocab_embed = nn.Embedding(n_vocab, d_model)
        self.position_model = nn.Embedding(context_size, d_model)
        self.softmax_proj = nn.Linear(d_model, n_vocab)
        self.layer_norm = nn.LayerNorm(d_model)
        self.position = torch.arange(0, context_size).unsqueeze(1).int().to(device)
        self.apply(Transformer.init_weights)
        for name, param in self.named_parameters():
            # scale the residual weight by sqrt(N)
            if 'residual_proj' in name and 'weight' in name:
                print("scale the weight by sqrt(n_layer): %s" % name)
                param = param / math.sqrt(n_layers)

    def forward(self, x): # [B, C]
        x = self.vocab_embed(x) # [B, C, D]
        pos = self.position_model(self.position).view((1, x.size(1), x.size(2))) # [1, C, D]
        x = x + pos # [B, C, D]
        # position
        for i, l in enumerate(self.blocks):
            x = l(x)
        x = self.layer_norm(x)
        x = self.softmax_proj(x) # [B, C, vocab]
        return x

    def get_optimizer(self):
        decay = []
        non_decay = []
        for module_name, m in self.named_modules():
            for name, param in m.named_parameters():
                if module_name:
                    full_name = "%s.%s" % (module_name, name)
                else:
                    full_name = name
                if isinstance(m, nn.Linear) and 'weight' in name:
                    decay.append(full_name)
                elif 'bias' in name or isinstance(m, nn.LayerNorm) or isinstance(m, nn.Embedding):
                    non_decay.append(full_name)
        key2parameter = {k:v for k,v in self.named_parameters()}

        assert len(set(decay) & set(non_decay)) == 0, set(decay) & set(non_decay)
        assert len(set(decay) | set(non_decay)) == len(key2parameter.keys())
        parameter_groups = [
            {"params": [key2parameter[k] for k in set(decay)], "weight_decay":0.1},
            {"params": [key2parameter[k] for k in set(non_decay)], "weight_decay":0}
        ]
        return torch.optim.AdamW(parameter_groups, betas=(0.9, 0.95), lr=1e-3)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def generate(self, x):
        pass
