import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from typing import Optional, Union, List


class TextDataset(Dataset):
    """A PyTorch Dataset for text data with tokenization and sliding window chunking."""
    
    def __init__(self, text: str, tokenizer, max_length: int = 256, stride: int = 128):
        """
        Initialize the dataset.
        
        Args:
            text: Input text string
            tokenizer: Tokenizer (e.g., tiktoken encoder)
            max_length: Maximum sequence length
            stride: Stride for sliding window
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        
        # Tokenize the entire text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
        # Use sliding window to create overlapping sequences
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    tokenizer_name: str = "gpt2"
) -> DataLoader:
    """
    Create a DataLoader for text data.
    
    Args:
        text: Input text string
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length
        stride: Stride for sliding window
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        num_workers: Number of worker processes
        tokenizer_name: Name of tokenizer encoding
    
    Returns:
        DataLoader instance
    """
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    
    # Create dataset
    dataset = TextDataset(text, tokenizer, max_length, stride)
    
    # Create and return dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
