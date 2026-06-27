import os
os.environ['PYTORCH_ATTENTION_BACKEND'] = 'eager'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

ACTIVATION = nn.Tanh
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FNN(nn.Module):
    def __init__(self, dim, keep_prob, activation):
        super(FNN, self).__init__()
        self.fc = nn.Linear(dim, int(dim / 4))
        self.activation = activation()
        self.dropout = nn.Dropout(p=keep_prob)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class TopAPPNPGNNModule(nn.Module):
    def __init__(self, dim_model, keep_prob, device, top_rate=0.1, reduce='mean', predict_dim=None, k=10, alpha=0.2):
        super(TopAPPNPGNNModule, self).__init__()
        self.dim_model = dim_model
        self.keep_prob = keep_prob
        self.device = device
        self.top_rate = top_rate
        self.reduce = reduce
        self.predict_dim = predict_dim if predict_dim else dim_model
        self.k = k
        self.alpha = alpha
        
        self.fc = nn.Linear(self.predict_dim, self.predict_dim)
        self.dropout = nn.Dropout(p=keep_prob)
        self.activation = ACTIVATION()
        self.ln = nn.LayerNorm(self.predict_dim)
    
    def seq_to_graph(self, topk_values, topk_indices, hidden_state, reduce, length):
        """纯 PyTorch 实现：将序列转换为图结构"""
        nodes = hidden_state[topk_indices[:length]]
        num_nodes = len(topk_indices[:length])
        
        if num_nodes <= 1:
            # 单节点，返回原始特征
            return nodes
        
        # 使用纯 PyTorch 创建邻接矩阵并进行图卷积
        # 创建全1邻接矩阵（不含自环）
        adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        # 度矩阵的逆平方根
        degree = adj.sum(dim=1)
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
        # 归一化邻接矩阵
        norm_adj = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
        # 图卷积
        node_feat = torch.matmul(norm_adj, nodes)
        return node_feat
    
    def forward(self, hidden_state, attention, lengths):
        batch_size = hidden_state.size(0)
        seq_len = hidden_state.size(1)
        
        dealt_atten = attention
        topk_result = torch.topk(dealt_atten, round(self.top_rate * hidden_state.size(1)), dim=-1)
        topk_indices = topk_result.indices
        
        # 计算选中的节点数
        num_selected_nodes = topk_result.indices.shape[-1]
        
        # 使用 torch_geometric 批量处理所有图
        all_graphs = []
        graph_node_counts = []  # 记录每个图对应的样本和位置
        
        for i in range(batch_size):
            length = lengths[i].item() if torch.is_tensor(lengths[i]) else int(lengths[i])
            
            for pos_idx in range(min(length, seq_len)):
                indices = topk_indices[i, pos_idx].long()
                selected_nodes = hidden_state[i, indices]  # [num_selected_nodes, dim_model]
                num_nodes = min(num_selected_nodes, length)
                graph_node_counts.append((i, pos_idx, num_nodes))
                
                if num_nodes > 1:
                    # 创建边索引（全连接图，不含自环）
                    edge_index = []
                    for j in range(num_nodes):
                        for k in range(num_nodes):
                            if j != k:
                                edge_index.append([j, k])
                    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                    
                    # 创建 PyG Data 对象
                    data = Data(x=selected_nodes[:num_nodes], edge_index=edge_index)
                    all_graphs.append(data)
                else:
                    # 单节点图
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    data = Data(x=selected_nodes[:1], edge_index=edge_index)
                    all_graphs.append(data)
        
        # 批量处理所有图
        if all_graphs:
            batched_data = Batch.from_data_list(all_graphs)
            
            # 使用 GCN 卷积（需要在 GPU 上）
            gcn_conv = GCNConv(self.dim_model, self.dim_model).to(self.device)
            
            # 执行图卷积
            x = gcn_conv(batched_data.x.to(self.device), batched_data.edge_index.to(self.device))
            
            # 按图分割结果
            node_features = x.cpu()
            
            # 重新组织输出
            node_embeddings = torch.zeros(batch_size, seq_len, self.predict_dim, device=self.device)
            feat_idx = 0
            for i, pos_idx, num_nodes in graph_node_counts:
                if num_nodes > 1:
                    node_feat = node_features[feat_idx:feat_idx + num_nodes]
                    feat_idx += num_nodes
                else:
                    node_feat = node_features[feat_idx:feat_idx + 1]
                    feat_idx += 1
                
                # 投影到 predict_dim
                if self.proj is not None:
                    node_feat = self.proj(node_feat.to(self.device))
                else:
                    node_feat = node_feat.to(self.device)
                
                node_embeddings[i, pos_idx] = node_feat.mean(dim=0)
        else:
            node_embeddings = torch.zeros(batch_size, seq_len, self.predict_dim, device=self.device)
        
        return node_embeddings


class TopGNN(nn.Module):
    def __init__(self, num_head, dim_model, output_dim, keep_prob, device, top_rate=0.1, agg=None):
        super(TopGNN, self).__init__()
        self.num_head = num_head
        self.dim_model = dim_model
        self.output_dim = output_dim
        self.keep_prob = keep_prob
        self.device = device
        self.top_rate = top_rate
        self.agg = agg
        self.activation = ACTIVATION()
        
        self.word_fc = nn.Linear(output_dim, output_dim)
        self.semantic_fc = nn.Linear(output_dim, output_dim)

class TopAttentionGNN(TopGNN):
    def __init__(self, num_head, dim_model, output_dim, keep_prob, device, top_rate=0.1, agg=None, reduce='mean', k=10, alpha=0.2):
        super(TopAttentionGNN, self).__init__(num_head, dim_model, output_dim, keep_prob, device, top_rate, agg)
        self.word_gnn = TopAPPNPAttentionGNNModule(dim_model, keep_prob, device, top_rate, reduce, output_dim, k, alpha)
        self.semantic_gnn = TopAPPNPAttentionGNNModule(dim_model, keep_prob, device, top_rate, reduce, output_dim, k, alpha)
        self.hs_word_trans = nn.Linear(dim_model, dim_model)
        self.hs_semantic_trans = nn.Linear(dim_model, dim_model)
    
    def forward(self, output, lengths):
        word_attention = torch.stack(output['attentions'][:3], dim=4).max(dim=4)[0].mean(dim=1)
        word_embed = self.hs_word_trans(torch.stack(output['hidden_states'][:3], dim=3).transpose(-2, -1)).transpose(-2, -1).max(dim=3)[0]
        
        semantic_attention = torch.stack(output['attentions'][-3:], dim=4).max(dim=4)[0].mean(dim=1)
        semantic_embed = self.hs_semantic_trans(torch.stack(output['hidden_states'][-4:-1], dim=3).transpose(-2, -1)).transpose(-2, -1).max(dim=3)[0]
        
        word_output = self.activation(self.word_fc(self.word_gnn(word_embed, word_attention, lengths)))
        semantic_output = self.activation(self.semantic_fc(self.semantic_gnn(semantic_embed, semantic_attention, lengths)))
        
        return word_output, semantic_output

class TopAPPNPAttentionGNNModule(TopAPPNPGNNModule):
    def __init__(self, dim_model, keep_prob, device, top_rate=0.1, reduce='mean', predict_dim=None, k=10, alpha=0.2):
        super(TopAPPNPAttentionGNNModule, self).__init__(dim_model, keep_prob, device, top_rate, reduce, predict_dim, k, alpha)
        # 添加投影层，将 dim_model 投影到 predict_dim
        if self.dim_model != self.predict_dim:
            self.proj = nn.Linear(self.dim_model, self.predict_dim)
        else:
            self.proj = None

class InteractionModule(nn.Module):
    def __init__(self, dim_model, attention_mode='normal'):
        super(InteractionModule, self).__init__()
        self.attention_mode = attention_mode
        self.dim_model = dim_model
        self.fc = nn.Linear(dim_model * 2, dim_model)
        self.output_trans = nn.Linear(dim_model, dim_model)
        self.context_trans = nn.Linear(dim_model, dim_model)
        self.activation = ACTIVATION()
        
        if self.attention_mode == 'biaffine':
            self.bilinear = nn.Linear(self.dim_model, self.dim_model, bias=False)
            self.U = nn.Linear(self.dim_model, 1)
            self.V = nn.Linear(self.dim_model, 1)
    
    def forward(self, output, context, attention_mask):
        lengths = torch.sum(attention_mask, dim=-1)
        batch_size = output.shape[0]
        context_len = output.shape[1]
        
        key_padding_mask = torch.tensor([[0] * length + [-1e9] * (context_len - length) for length in lengths]
                                       ).unsqueeze(dim=1).to(DEVICE)
        
        if self.attention_mode == 'biaffine':
            attn = self.bilinear(self.output_trans(output))
            attn = torch.bmm(attn, self.context_trans(context).transpose(1, 2))
            attn = attn + self.U(output).expand(attn.shape) + self.V(context).transpose(1, 2).expand(attn.shape)
            attn = attn + key_padding_mask
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.tanh(torch.bmm(self.output_trans(output), self.context_trans(context).transpose(1, 2)))
            attn = attn + key_padding_mask
            attn = F.softmax(attn.view(-1, context_len), dim=1).view(batch_size, -1, context_len)
        
        mix = torch.bmm(attn, context)
        combined = torch.cat((output, mix), dim=2)
        output = F.tanh(self.fc(combined.view(-1, 2 * self.dim_model))).view(batch_size, -1, self.dim_model)
        output = (torch.sum((output.transpose(1, 2) * attention_mask.unsqueeze(dim=1)).transpose(1, 2), dim=1).T / lengths).T
        
        return output

class GateModule(nn.Module):
    def __init__(self, dim_model):
        super(GateModule, self).__init__()
        self.bert_trans = nn.Linear(dim_model, dim_model)
        self.gnn_trans = nn.Linear(dim_model, dim_model)
        self.activation = nn.Sigmoid()
    
    def forward(self, bert, gnn):
        alpha = self.activation(self.bert_trans(bert) + self.gnn_trans(gnn))
        return alpha * gnn

class DGCBERT(nn.Module):
    def __init__(self, config):
        super(DGCBERT, self).__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        
        self.hidden_size = config["hidden_size"]
        self.predict_dim = config["predict_dim"]
        self.num_class = config["num_class"]
        self.keep_prob = config["keep_prob"]
        self.pad_size = config["max_seq_length"]
        self.mode = config["mode"]
        self.model_type = config["model_type"]
        
        self.k = config.get("k", 10)
        self.alpha = config.get("alpha", 0.2)
        self.top_rate = config.get("top_rate", 0.1)
        
        self.block_pooled = False
        self.reduce_method = 'mean'
        self.num_head = 12
        
        self.model_name = f"DGCBERT_{self.mode}_{self.model_type}"
        
        self.attention_mode = 'biaffine'
        if self.mode == 'top_normal':
            self.attention_mode = 'normal'
        elif self.mode == 'top_biaffine+softmax':
            self.reduce_method = 'softmax'
        
        self.gnn = TopAttentionGNN(
            self.num_head, self.hidden_size, self.predict_dim, 
            self.keep_prob, self.device, self.top_rate, 
            'APPNP', self.reduce_method, self.k, self.alpha
        )
        
        self.final_dim = 2 * self.predict_dim if self.block_pooled else 3 * self.predict_dim
        
        if self.predict_dim <= 512:
            self.bert_trans = nn.Sequential(
                FNN(self.hidden_size, self.keep_prob, ACTIVATION),
                ACTIVATION(),
                nn.Dropout(p=self.keep_prob),
                nn.Linear(int(self.hidden_size / 4), self.predict_dim),
                ACTIVATION()
            )
        elif self.predict_dim == 768:
            self.bert_trans = nn.Sequential(
                nn.Dropout(p=self.keep_prob),
                nn.Linear(self.hidden_size, self.hidden_size),
                ACTIVATION()
            )
        else:
            self.bert_trans = nn.Sequential(
                nn.Linear(self.hidden_size, int(self.hidden_size / 2)),
                ACTIVATION(),
                nn.Linear(int(self.hidden_size / 2), int(self.hidden_size / 2)),
                ACTIVATION(),
                nn.Dropout(p=self.keep_prob),
                nn.Linear(int(self.hidden_size / 2), self.predict_dim),
                ACTIVATION()
            )
        
        if self.attention_mode in ['normal', 'biaffine']:
            self.word_interaction = InteractionModule(self.predict_dim, self.attention_mode)
            self.semantic_interaction = InteractionModule(self.predict_dim, self.attention_mode)
        
        self.word_gate = GateModule(self.predict_dim)
        self.semantic_gate = GateModule(self.predict_dim)
        
        self.dropout = nn.Dropout(p=self.keep_prob)
        self.fc = nn.Sequential(
            FNN(self.final_dim, self.keep_prob, ACTIVATION),
            ACTIVATION(),
            nn.Linear(int(self.final_dim / 4), self.num_class),
        )
        
        # 加载配置并设置 eager attention 以支持 output_attentions
        self.bert_config = BertConfig.from_pretrained(self.model_type)
        self.bert_config._attn_implementation = "eager"
        self.bert = BertModel.from_pretrained(self.model_type, config=self.bert_config)
    
    def forward(self, input_ids, attention_mask, labels=None):
        lengths = torch.sum(attention_mask, dim=-1)
        content = input_ids
        
        output = self.bert(
            content, 
            attention_mask=attention_mask, 
            return_dict=True, 
            output_attentions=True,
            output_hidden_states=True
        )
        pooled = output['pooler_output']
        
        word_attention_gnn, semantic_attention_gnn = self.gnn(output, lengths)
        
        if not self.block_pooled:
            bert_out = self.bert_trans(pooled)
            word_attention_gnn_mix = self.word_interaction(word_attention_gnn, semantic_attention_gnn, attention_mask)
            semantic_attention_gnn_mix = self.semantic_interaction(semantic_attention_gnn, word_attention_gnn, attention_mask)
            
            word_attention_gnn = self.word_gate(bert_out, word_attention_gnn_mix)
            semantic_attention_gnn = self.semantic_gate(bert_out, semantic_attention_gnn_mix)
        
        gnn_out = torch.cat((word_attention_gnn, semantic_attention_gnn), dim=1)
        if self.block_pooled:
            out = gnn_out
        else:
            out = torch.cat((bert_out, gnn_out), dim=1)
        
        out = self.dropout(out)
        logits = self.fc(out)
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self