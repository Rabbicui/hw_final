#!/usr/bin/env python3
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvlayer
from layers import GraphConvII
from layers import GraphAttentionLayer
from layers import JumpingKnowledge



class GCN(nn.Module):
    def __init__(self, nfeat, nclass, nlayers, nhidden, dropout):
        super(GCN, self).__init__()
        self.input_dim = nfeat
        self.output_dim = nclass
        self.nlayers = nlayers
        self.nhidden = nhidden
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.gcn1 = GraphConvlayer(self.input_dim, self.nhidden)
        self.gcn2 = GraphConvlayer(self.nhidden, self.output_dim)
        self.convs = nn.ModuleList()
        assert self.nlayers >= 2, 'nlayers must be more than or equal to 2.'
        if self.nlayers > 2:
            for _ in range(self.nlayers-2):
                self.convs.append(GraphConvlayer(self.nhidden, self.nhidden))
        # self.dropout = dropout

    def forward(self, feature, adj):
        _layers = []
        layer_inner = self.act_fn(self.gcn1(adj, feature))
        # _layers.append(layer_inner)
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        i = 1
        for con in self.convs:
            i += 1
            layer_inner = self.act_fn(con(adj, layer_inner))
            if i % 2 == 0:
                _layers.append(layer_inner)
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        logits = self.gcn2(adj, layer_inner)
        # return F.log_softmax(layer_inner, dim=1), _layers # 这个输出适用于nllloss
        return logits, _layers  # 这个输出适用于交叉熵损失函数

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x
 

class APPNP(nn.Module):

    def __init__(self, nfeat, nclass, nlayers, nhidden, dropout):
        super(APPNP, self).__init__()
        self.input_dim = nfeat
        self.output_dim = nclass
        self.nlayers = nlayers
        self.nhidden = nhidden
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(self.input_dim, self.nhidden))
        self.fcs.append(nn.Linear(self.nhidden, self.output_dim))

        # self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for _ in range(self.nlayers):
            self.convs.append(GraphConvlayer(self.nhidden, self.nhidden))
        # self.dropout = dropout

    def forward(self, feature, adj, alpha):
        _layers = []
        layer0 = self.fcs[0](feature)
        # conserve layer0
        _layers.append(layer0)
        i = 0
        for con in self.convs:
            # layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            i += 1
            if i == 1:
                layer_inner = con(adj, _layers[0], _layers[0], alpha, appnp=True)
            else:
                layer_inner = con(adj, layer_inner, _layers[0], alpha, appnp=True)
            if i % 2 == 0:
                _layers.append(layer_inner)
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        logits = self.fcs[1](layer_inner)
        # return F.log_softmax(layer_inner, dim=1), _layers # 这个输出适用于nllloss
        return logits, _layers  # 这个输出适用于交叉熵损失函数


class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvII(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        _layer = []
        x = F.dropout(x, self.dropout, training=self.training)
        # 先使用一个线性映射进行降维，降到nhidden
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        # 存储第0层
        _layer.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
            if i%2 == 0:
                _layer.append(layer_inner)
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1), _layer


class JKNet(nn.Module):
    def __init__(
        self, in_dim, hid_dim, out_dim, num_layers=1, mode="cat", dropout=0.0
    ):
        super(JKNet, self).__init__()

        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.act_fn = nn.ReLU()
        self.layers.append(GraphConvlayer(in_dim, hid_dim))
        for _ in range(num_layers):
            self.layers.append(GraphConvlayer(hid_dim, hid_dim))

        if self.mode == "lstm":
            self.jump = JumpingKnowledge(mode, hid_dim, num_layers)
        else:
            self.jump = JumpingKnowledge(mode)

        if self.mode == "cat":
            hid_dim = hid_dim * (num_layers + 1)

        self.output = nn.Linear(hid_dim, out_dim)
        self.reset_params()

    def reset_params(self):
        self.output.reset_parameters()
        for layers in self.layers:
            layers.reset_parameters()
        self.jump.reset_parameters()

    def forward(self, feats, adj):
        """
        feats: input_feature
        """
        rep_lst = []
        for layer in self.layers:
            feats = self.dropout(self.act_fn(layer(adj, feats)))
            rep_lst.append(feats)

        fin_rep = self.jump(rep_lst)
        return self.output(fin_rep)

if __name__ == '__main__':
    pass
