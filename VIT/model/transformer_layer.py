import torch.nn as nn
from VIT.model.attention import Attention

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        emb_dim = config['emb_dim']
        ff_hidden_dim = config['ff_dim'] if 'ff_dim' in config else 4 * emb_dim
        ff_drop_prob = config['ff_drop'] if 'ff_drop' in config else 0.0
        self.att_norm = nn.LayerNorm(emb_dim)
        self.attn_block = Attention(config)
        self.ff_norm = nn.LayerNorm(emb_dim)

        self.ff_block = nn.Sequential(
            nn.Linear(emb_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(ff_drop_prob),
            nn.Linear(ff_hidden_dim, emb_dim),
            nn.Dropout(ff_drop_prob)
        )

    def forward(self, x):
        out = x
        out = out + self.attn_block(self.att_norm(out))
        out = out + self.ff_block(self.ff_norm(out))
        return out