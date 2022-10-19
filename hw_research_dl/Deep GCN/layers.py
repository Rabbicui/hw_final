#!/usr/bin/env python3
# coding: utf-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class GraphConvlayer(nn.Module):

    def __init__(self, input_dim, output_dim, use_bias=True, residual=False):
        """图卷积：L*X*\theta

        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super(GraphConvlayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.residual = residual
        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature, h0=None, alpha=None, appnp=False):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
    
        Args: 
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
        hi = torch.spmm(adjacency, input_feature)
        if appnp:
            support = (1 - alpha) * hi + alpha * h0
        else:
            support = hi
        output = torch.mm(support, self.weight)
        if self.use_bias:
            output += self.bias
        if self.residual:
            output = output + input_feature
        return output

    """  
  def __repr__(self):
        return self.__class__.__name__ + ' ('             + str(self.input_dim) + ' -> '             + str(self.output_dim) + ')'
    """


class GraphConvII(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvII, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        # 自动转化为可训练的参数
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        # 这里的theta指的是原文中的beta
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # a^T * w:(w1||w2), w denotes concated vector, so a's size is double
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # node representation with attention
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class JumpingKnowledge(nn.Module):
    """

    The Jumping Knowledge aggregation module from `Representation Learning on
    Graphs with Jumping Knowledge Networks <https://arxiv.org/abs/1806.03536>`__

    It aggregates the output representations of multiple GNN layers with

    **concatenation**
    or **max pooling**
    or **LSTM**
    with attention scores :math:`\alpha_i^{(t)}` obtained from a BiLSTM

    Parameters
    ----------
    mode : str
        The aggregation to apply. It can be 'cat', 'max', or 'lstm',
        corresponding to the equations above in order.
    in_feats : int, optional
        This argument is only required if :attr:`mode` is ``'lstm'``.
        The output representation size of a single GNN layer. Note that
        all GNN layers need to have the same output representation size.
    num_layers : int, optional
        This argument is only required if :attr:`mode` is ``'lstm'``.
        The number of GNN layers for output aggregation.
    """
    def __init__(self, mode='cat', in_feats=None, num_layers=None):
        super(JumpingKnowledge, self).__init__()
        assert mode in ['cat', 'max', 'lstm'], \
            "Expect mode to be 'cat', or 'max' or 'lstm', got {}".format(mode)
        self.mode = mode

        if mode == 'lstm':
            assert in_feats is not None, 'in_feats is required for lstm mode'
            assert num_layers is not None, 'num_layers is required for lstm mode'
            hidden_size = (num_layers * in_feats) // 2
            self.lstm = nn.LSTM(in_feats, hidden_size, bidirectional=True, batch_first=True)
            self.att = nn.Linear(2 * hidden_size, 1)

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters. This comes into effect only for the lstm mode.
        """
        if self.mode == 'lstm':
            self.lstm.reset_parameters()
            self.att.reset_parameters()

    def forward(self, feat_list):
        r"""

        Description
        -----------
        Aggregate output representations across multiple GNN layers.

        Parameters
        ----------
        feat_list : list[Tensor]
            feat_list[i] is the output representations of a GNN layer.

        Returns
        -------
        Tensor
            The aggregated representations.
        """
        if self.mode == 'cat':
            return torch.cat(feat_list, dim=-1)
        elif self.mode == 'max':
            return torch.stack(feat_list, dim=-1).max(dim=-1)[0]
        else:
            # LSTM
            stacked_feat_list = torch.stack(feat_list, dim=1)  # (N, num_layers, in_feats)
            alpha, _ = self.lstm(stacked_feat_list)
            alpha = self.att(alpha).squeeze(-1)  # (N, num_layers)
            alpha = torch.softmax(alpha, dim=-1)
            return (stacked_feat_list * alpha.unsqueeze(-1)).sum(dim=1)

