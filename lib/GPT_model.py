import torch.nn
from transformer_block import TransformerBlock
from feedforward import LayerNorm

GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}



class GPTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = torch.nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]


        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.output =  torch.nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False),


    def forward(self, in_idx):

        # todo
        x = self.tok_emb(in_idx) + self.pos_emb(in_idx)
        x =  self.drop_emb(x)

        for transformer in self.trf_blocks:
            x = transformer(x)

        x = self.final_norm(x)
        x = self.output(x)

        return x 