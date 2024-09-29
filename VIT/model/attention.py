import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['n_heads']
        self.head_dim = config['head_dim']
        self.emb_dim = config['emb_dim']
        self.drop_prob = config['dropout'] if 'dropout' in config else 0.0
        self.att_dim = self.n_heads * self.head_dim

        self.qkv_proj = nn.Linear(self.emb_dim, 3 * self.att_dim, bias=False)
        self.output_proj = nn.Sequential(
            nn.Linear(self.att_dim, self.emb_dim),
            nn.Dropout(self.drop_prob))

        self.attn_dropout = nn.Dropout(self.drop_prob)

    def forward(self, x):
        B, N = x.shape[:2]
        q, k, v = self.qkv_proj(x).split(self.att_dim, dim=-1)
        q = rearrange(q, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        k = rearrange(k, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        v = rearrange(v, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)

        att = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** (-0.5))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        out = torch.matmul(att, v)
        out = rearrange(out, 'b n_h n h_dim -> b n (n_h h_dim)')

        out = self.output_proj(out)

        return out
