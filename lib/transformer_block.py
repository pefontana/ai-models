from multihead_attention import MultiHeadAttention
from feedforward import FeedForward
from feedforward import LayerNorm
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
            )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        #norm
        x = self.norm1(x)
        #mha
        x = self.att(x)
        #dropout
        x = self.drop_shortcut(x)
        x = x + shortcut

        #norm2
        shortcut = x
        x = self.norm2(x)
        #ff
        x = self.ff(x)
        #dropout
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


        
