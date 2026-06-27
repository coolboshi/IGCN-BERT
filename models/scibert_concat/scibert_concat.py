import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class SciBERT_Concat(nn.Module):
    def __init__(self, config):
        super(SciBERT_Concat, self).__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        
        self.num_class = config["num_class"]
        self.max_seq_length = config["max_seq_length"]
        self.hidden_size = config.get("hidden_size", 768)
        self.num_layers = config.get("num_layers", 13)  # BERT has 13 layers (embedding + 12 encoders)
        self.dropout = config.get("dropout", 0.3)
        self.model_type = config.get("model_type", "allenai/scibert-scivocab-uncased")
        
        self.bert = BertModel.from_pretrained(self.model_type)
        
        self.concat_dim = self.hidden_size * self.num_layers
        
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.concat_dim, self.num_class)
        
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
        
        concat_cls = torch.cat(cls_tokens, dim=1)
        
        concat_cls = self.dropout(concat_cls)
        logits = self.fc(concat_cls)
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self