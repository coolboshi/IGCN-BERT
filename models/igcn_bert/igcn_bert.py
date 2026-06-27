import os
os.environ['PYTORCH_ATTENTION_BACKEND'] = 'eager'

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class WANU(nn.Module):
    def __init__(self, hidden_dim, d_A):
        super(WANU, self).__init__()
        self.W_ea = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, adj_multi_channel, node_emb):
        n_token, _, d_A = adj_multi_channel.shape
        channel_outputs = []
        for i in range(d_A):
            adj_i = adj_multi_channel[:, :, i]
            h_i = torch.matmul(adj_i, node_emb)
            h_i = self.W_ea(h_i)
            channel_outputs.append(h_i)
        stack_h = torch.stack(channel_outputs, dim=0)
        avg_h = torch.mean(stack_h, dim=0)
        updated_node = F.relu(avg_h)
        return self.dropout(updated_node)


class NAWU(nn.Module):
    def __init__(self, hidden_dim, d_A):
        super(NAWU, self).__init__()
        self.W_na = nn.Linear(2 * hidden_dim + d_A, d_A)

    def forward(self, adj_multi_channel, node_emb):
        n_token = node_emb.shape[0]
        h_i = node_emb.unsqueeze(1).repeat(1, n_token, 1)
        h_j = node_emb.unsqueeze(0).repeat(n_token, 1, 1)
        adj_ori = adj_multi_channel
        concat_feat = torch.cat([adj_ori, h_i, h_j], dim=-1)
        new_adj = self.W_na(concat_feat)
        return new_adj


class IGCNLayer(nn.Module):
    def __init__(self, hidden_dim, d_A):
        super(IGCNLayer, self).__init__()
        self.WANU = WANU(hidden_dim, d_A)
        self.NAWU = NAWU(hidden_dim, d_A)

    def forward(self, adj_in, node_in):
        node_new = self.WANU(adj_in, node_in)
        adj_new = self.NAWU(adj_in, node_new)
        return adj_new, node_new


class IGCNBERT(nn.Module):
    def __init__(self, config):
        super(IGCNBERT, self).__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

        self.num_class = config["num_class"]
        self.max_seq_length = config["max_seq_length"]
        self.hidden_size = config.get("hidden_size", 768)
        self.dropout_rate = config.get("keep_prob", 0.3)
        self.model_type = config.get("model_type", "./data/bert/scibert-scivocab-uncased")
        self.l_layers = config.get("l_layers", 3)
        self.d_A = config.get("d_A", 32)
        self.M_igcn = config.get("M_igcn", 3)

        self.bert_config = BertConfig.from_pretrained(self.model_type)
        self.bert_config._attn_implementation = "eager"
        self.bert = BertModel.from_pretrained(self.model_type, config=self.bert_config)

        self.W_lex = nn.Linear(self.hidden_size, self.hidden_size)
        self.b_lex = nn.Parameter(torch.zeros(self.hidden_size))
        self.W_sem = nn.Linear(self.hidden_size, self.hidden_size)
        self.b_sem = nn.Parameter(torch.zeros(self.hidden_size))
        self.W_ls = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.b_ls = nn.Parameter(torch.zeros(self.hidden_size))

        self.W_fusion = nn.Linear(2 * self.l_layers, self.d_A)
        self.b_fusion = nn.Parameter(torch.zeros(self.d_A))

        self.igcn_blocks = nn.ModuleList([
            IGCNLayer(self.hidden_size, self.d_A) for _ in range(self.M_igcn)
        ])

        self.dropout = nn.Dropout(self.dropout_rate)
        self.cls_fc = nn.Linear(self.hidden_size, self.num_class)

    def extract_lex_sem_repr(self, bert_hidden_states):
        all_layers = torch.stack(bert_hidden_states[1:], dim=0)
        front_layers = all_layers[:self.l_layers]
        back_layers = all_layers[-self.l_layers:]

        front_proj = self.W_lex(front_layers) + self.b_lex
        H_lex = torch.max(front_proj, dim=0).values
        back_proj = self.W_sem(back_layers) + self.b_sem
        H_sem = torch.max(back_proj, dim=0).values

        H_ls = torch.cat([H_lex, H_sem], dim=-1)
        H_ls = self.W_ls(H_ls) + self.b_ls
        return H_ls

    def extract_lex_sem_adj(self, bert_attentions):
        all_att = torch.stack(bert_attentions, dim=0)
        att_front = all_att[:self.l_layers]
        att_back = all_att[-self.l_layers:]

        A_lex = torch.mean(att_front, dim=2)
        A_sem = torch.mean(att_back, dim=2)

        A_ls = torch.cat([A_lex, A_sem], dim=0)
        A_ls = A_ls.permute(1, 2, 3, 0)
        B, L1, L2, C = A_ls.shape
        A_flat = A_ls.reshape(B * L1 * L2, C)
        A_fused = self.W_fusion(A_flat) + self.b_fusion
        adj_multi = A_fused.reshape(B, L1, L2, self.d_A)
        return adj_multi

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape

        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True
        )
        hidden_states = bert_out.hidden_states
        attentions = bert_out.attentions

        node_emb = self.extract_lex_sem_repr(hidden_states)
        adj_mat = self.extract_lex_sem_adj(attentions)

        for igcn in self.igcn_blocks:
            new_adj_list = []
            new_node_list = []
            for b in range(batch_size):
                adj_b = adj_mat[b]
                node_b = node_emb[b]
                adj_b, node_b = igcn(adj_b, node_b)
                new_adj_list.append(adj_b)
                new_node_list.append(node_b)
            adj_mat = torch.stack(new_adj_list, dim=0)
            node_emb = torch.stack(new_node_list, dim=0)

        doc_emb = torch.mean(node_emb, dim=1)
        doc_emb = self.dropout(doc_emb)
        logits = self.cls_fc(doc_emb)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self
