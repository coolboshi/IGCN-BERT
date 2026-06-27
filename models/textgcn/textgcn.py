import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


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


class TextGCN(nn.Module):
    def __init__(self, config):
        super(TextGCN, self).__init__()
        self.config = config
        self.nfeat = config.get("nfeat", 1000)
        self.nhid = config.get("nhid", 200)
        self.nclass = config.get("nclass", 2)
        self.dropout = config.get("dropout", 0.5)
        
        self.gc1 = GraphConvolution(self.nfeat, self.nhid)
        self.gc2 = GraphConvolution(self.nhid, self.nclass)

    def forward(self, x, adj):
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