import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BERT_CLS(nn.Module):
    def __init__(self, config):
        super(BERT_CLS, self).__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        
        self.num_class = config["num_class"]
        self.max_seq_length = config["max_seq_length"]
        self.hidden_size = config.get("hidden_size", 768)
        self.dropout = config.get("dropout", 0.3)
        self.model_type = config.get("model_type", "bert-base-uncased")
        
        try:
            self.bert = BertModel.from_pretrained(self.model_type)
        except:
            self.bert = BertModel.from_pretrained(self.model_type, local_files_only=True)
        
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.num_class)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids, 
            attention_mask=attention_mask, 
            return_dict=True
        )
        
        cls_embedding = outputs['pooler_output']
        
        cls_embedding = self.dropout(cls_embedding)
        logits = self.fc(cls_embedding)
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self