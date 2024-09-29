import torch
import torch.nn as nn
from einops import rearrange, repeat

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_height = config['image_height']
        image_width = config['image_width']
        im_channels = config['im_channels']
        emb_dim = config['emb_dim']
        patch_embd_drop = config['patch_emb_drop']

        self.patch_height = config['patch_height']
        self.patch_width = config['patch_width']

        num_patches = (image_height // self.patch_height) * (image_width // self.patch_width)

        patch_dim = im_channels * self.patch_height * self.patch_width

        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(emb_dim))
        self.patch_emb_dropout = nn.Dropout(patch_embd_drop)

    def forward(self, x):
        batch_size = x.shape[0]
        out = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)',
                        ph=self.patch_height,
                        pw=self.patch_width)
        out = self.patch_embed(out)

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=batch_size)
        out = torch.cat((cls_tokens, out), dim=1)

        out += self.pos_embed
        out = self.patch_emb_dropout(out)

        return out

