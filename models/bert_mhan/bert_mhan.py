import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class WordAttention(nn.Module):
    def __init__(self, hidden_size):
        super(WordAttention, self).__init__()
        self.hidden_size = hidden_size
        self.linear_project = nn.Linear(hidden_size, hidden_size)
        self.representation = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.representation)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        u = self.linear_project(x.contiguous().view(-1, hidden_size)).view(batch_size, seq_len, -1)
        u = torch.tanh(u)
        atten_weights = torch.matmul(u, self.representation).squeeze(-1)
        atten_weights = F.softmax(atten_weights, dim=1)
        atten_weights = atten_weights.unsqueeze(1)
        s = torch.matmul(atten_weights, x).squeeze(1)
        return s, atten_weights


class SentenceAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SentenceAttention, self).__init__()
        self.hidden_size = hidden_size
        self.linear_project = nn.Linear(hidden_size, hidden_size)
        self.representation = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.xavier_uniform_(self.representation)

    def forward(self, x):
        batch_size, num_sentences, hidden_size = x.size()
        u = self.linear_project(x.contiguous().view(-1, hidden_size)).view(batch_size, num_sentences, -1)
        u = torch.tanh(u)
        atten_weights = torch.matmul(u, self.representation).squeeze(-1)
        atten_weights = F.softmax(atten_weights, dim=1)
        atten_weights = atten_weights.unsqueeze(1)
        s = torch.matmul(atten_weights, x).squeeze(1)
        return s, atten_weights


class BERT_MHAN(nn.Module):
    def __init__(self, config):
        super(BERT_MHAN, self).__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        
        self.num_class = config["num_class"]
        self.max_seq_length = config["max_seq_length"]
        self.max_sentences = config.get("max_sentences", 50)
        self.hidden_size = config.get("hidden_size", 768)
        self.dropout = config.get("dropout", 0.3)
        self.model_type = config.get("model_type", "bert-base-uncased")
        
        self.bert = BertModel.from_pretrained(self.model_type)
        for param in self.bert.parameters():
            param.requires_grad = config.get("bert_freeze", False)
        
        self.word_attention = WordAttention(self.hidden_size)
        self.sentence_attention = SentenceAttention(self.hidden_size)
        
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_size, self.num_class)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.size(0)
        
        if len(input_ids.shape) == 3:
            batch_size, num_sentences, seq_len = input_ids.size()
            input_ids = input_ids.view(-1, seq_len)
            if attention_mask is not None:
                attention_mask = attention_mask.view(-1, seq_len)
        
        outputs = self.bert(
            input_ids, 
            attention_mask=attention_mask, 
            return_dict=True
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        if len(input_ids.shape) == 3:
            last_hidden_state = last_hidden_state.view(batch_size, num_sentences, seq_len, self.hidden_size)
        
        if len(last_hidden_state.shape) == 4:
            word_attended = []
            for i in range(num_sentences):
                sentence_output = last_hidden_state[:, i, :, :]
                sent_vec, _ = self.word_attention(sentence_output)
                word_attended.append(sent_vec)
            
            sentence_embeddings = torch.stack(word_attended, dim=1)
            doc_vec, _ = self.sentence_attention(sentence_embeddings)
        else:
            doc_vec, _ = self.word_attention(last_hidden_state)
        
        doc_vec = self.dropout(doc_vec)
        logits = self.fc(doc_vec)
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self