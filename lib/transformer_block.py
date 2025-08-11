GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

from multihead_attention import MultiHeadAttention
from feedforward import FeedForward
from feedforward import LayerNorm
import torch.nn
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
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
        self.drop_shortcut = torch.nn.Dropout(cfg["drop_rate"])

    def foward(self,x):
        shortcut = x
        #norm
        x = self.norm1(x)
        #mha
        x= self.att(x)
        #dropout
        x = self.drop_shortcut(x)
        x = x + shortcut

        #norm2
        shortcut = x
        x = self.norm2(x)
        #ff
        x = self.ff(x)
        #droput
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


        
