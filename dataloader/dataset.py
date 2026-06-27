import os
import json
import torch
from torch.utils.data import Dataset

class DGCBERTDataset(Dataset):
    def __init__(self, contents, labels, indexes, vocab, tokenizer, max_seq_length=256):
        self.contents = contents
        self.labels = labels
        self.indexes = indexes
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.CLS = '[CLS]'
        self.SEP = '[SEP]'
        self.PAD = '[PAD]'
        self.UNK = '[UNK]'
    
    def __len__(self):
        return len(self.contents)
    
    def __getitem__(self, idx):
        content = self.contents[idx]
        label = self.labels[idx]
        index = self.indexes[idx]
        
        tokens = self.tokenizer.tokenize(content.strip())
        tokens = [self.CLS] + tokens + [self.SEP]
        seq_len = len(tokens)
        
        if len(tokens) < self.max_seq_length:
            tokens.extend([self.PAD] * (self.max_seq_length - len(tokens)))
            mask = [1] * seq_len + [0] * (self.max_seq_length - seq_len)
        else:
            tokens = tokens[:self.max_seq_length]
            tokens[-1] = self.SEP
            seq_len = self.max_seq_length
            mask = [1] * self.max_seq_length
        
        input_ids = torch.tensor([self.vocab[token] for token in tokens], dtype=torch.int64)
        attention_mask = torch.tensor(mask, dtype=torch.int64)
        label = torch.tensor(label, dtype=torch.int64)
        index = torch.tensor(index, dtype=torch.int64)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
            "index": index
        }