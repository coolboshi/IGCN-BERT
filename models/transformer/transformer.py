import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] 
                                for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)

        scale = K.size(-1) ** -0.5
        context, attention = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out, attention


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(dim_model, num_head, dropout)
        self.feed_forward = PositionWiseFeedForward(dim_model, hidden, dropout)

    def forward(self, x):
        out, attention = self.attention(x)
        out = self.feed_forward(out)
        return out, attention


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        
        self.num_class = config["num_class"]
        self.max_seq_length = config["max_seq_length"]
        self.dropout = config.get("dropout", 0.5)
        self.dim_model = config.get("dim_model", 256)
        self.num_head = config.get("num_head", 4)
        self.hidden = config.get("hidden", 512)
        self.num_encoder = config.get("num_encoder", 3)
        self.vocab_size = config.get("vocab_size", 10000)
        
        self.embedding = nn.Embedding(self.vocab_size, self.dim_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        self.positional_encoding = PositionalEncoding(
            self.dim_model, self.max_seq_length, self.dropout, self.device
        )
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(self.dim_model, self.num_head, self.hidden, self.dropout)
            for _ in range(self.num_encoder)
        ])
        
        self.fc = nn.Linear(self.dim_model, self.num_class)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        
        for encoder in self.encoder_layers:
            x, _ = encoder(x)
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = torch.sum(x * mask_expanded, dim=1) / torch.sum(mask_expanded, dim=1)
        else:
            pooled = torch.mean(x, dim=1)
        
        pooled = self.dropout_layer(pooled)
        logits = self.fc(pooled)
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}