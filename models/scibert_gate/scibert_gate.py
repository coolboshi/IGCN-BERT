import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class SciBERT_Gate(nn.Module):
    def __init__(self, config):
        super(SciBERT_Gate, self).__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        
        self.num_class = config["num_class"]
        self.max_seq_length = config["max_seq_length"]
        self.hidden_size = config.get("hidden_size", 768)
        self.num_layers = config.get("num_layers", 13)
        self.dropout = config.get("dropout", 0.3)
        self.model_type = config.get("model_type", "allenai/scibert-scivocab-uncased")
        
        self.bert = BertModel.from_pretrained(self.model_type)
        
        self.attention_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.num_class)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids, 
            attention_mask=attention_mask, 
            return_dict=True,
            output_hidden_states=True
        )
        
        hidden_states = outputs['hidden_states']
        
        cls_tokens = []
        for layer in hidden_states:
            cls_token = layer[:, 0, :]
            cls_tokens.append(cls_token)
        
        cls_stack = torch.stack(cls_tokens, dim=1)
        
        attention_scores = self.attention_layer(cls_stack)
        attention_scores = attention_scores.squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = attention_weights.unsqueeze(-1)
        
        gated_output = torch.sum(cls_stack * attention_weights, dim=1)
        
        gated_output = self.dropout(gated_output)
        logits = self.fc(gated_output)
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self