import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from transformers import BertModel


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.spmm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BERT_GCN(nn.Module):
    def __init__(self, config):
        super(BERT_GCN, self).__init__()
        self.config = config
        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        
        self.nfeat = config.get("nfeat", 768)
        self.nhid = config.get("nhid", 200)
        self.nclass = config.get("nclass", 2)
        self.dropout = config.get("dropout", 0.5)
        self.model_type = config.get("model_type", "bert-base-uncased")
        
        try:
            self.bert = BertModel.from_pretrained(self.model_type)
        except:
            self.bert = BertModel.from_pretrained(self.model_type)
        for param in self.bert.parameters():
            param.requires_grad = config.get("bert_freeze", False)
        
        self.feature_compress = nn.Linear(768, self.nhid)
        self.gc1 = GraphConvolution(self.nhid, self.nhid)
        self.gc2 = GraphConvolution(self.nhid, self.nclass)

    def forward(self, x, adj):
        if x.dtype == torch.long and x.shape[-1] <= 512:
            outputs = self.bert(x, return_dict=True)
            x = outputs.last_hidden_state
            x = torch.mean(x, dim=1)
        
        x = self.feature_compress(x)
        x = torch.relu(x)
        
        x = self.gc1(x, adj)
        x = torch.relu(x)
        x = torch.dropout(x, self.dropout, train=self.training)
        x = self.gc2(x, adj)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        return self