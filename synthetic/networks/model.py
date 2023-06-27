import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from torch.nn.parameter import Parameter
from torch_scatter import scatter_add
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool, Linear

from typing import Callable, Union

from torch_geometric.typing import OptPairTensor

import torch.jit as jit
from typing import Tuple, List
import torch.nn.init as init
from collections import namedtuple


class PathNN(nn.Module):
    """
    Path Neural Networks that operate on collections of paths. Uses 1 LSTM shared across convolutional layers.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        cutoff,
        n_classes,
        dropout,
        device,
        residuals=True,
        encode_distances=False,
        l2_norm=False,
        predict=True,
    ):
        super(PathNN, self).__init__()
        self.cutoff = cutoff
        self.device = device
        self.residuals = residuals
        self.dropout = dropout
        self.encode_distances = encode_distances
        self.l2_norm = l2_norm
        self.predict = predict

        # Feature Encoder that projects initial node representation to d-dim space
        self.feature_encoder = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
        )
        conv_class = PathConv

        # 1 shared LSTM across layers
        if encode_distances:
            self.distance_encoder = nn.Embedding(cutoff, hidden_dim)
            self.lstm = nn.LSTM(
                input_size=hidden_dim * 2,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
                bias=True,
            )
        else:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
                bias=True,
            )

        self.convs = nn.ModuleList([])
        for _ in range(self.cutoff - 1):
            self.convs.append(
                conv_class(
                    hidden_dim,
                    self.lstm,
                    batch_norm=nn.Identity(),
                    residuals=self.residuals,
                    dropout=self.dropout,
                )
            )

        self.hidden_dim = hidden_dim
        if self.predict:
            self.linear1 = Linear(hidden_dim, hidden_dim)
            self.linear2 = Linear(hidden_dim, n_classes)

        self.reset_parameters()

    def reset_parameters(self):

        for c in self.feature_encoder.children():
            if hasattr(c, "reset_parameters"):
                c.reset_parameters()
        self.lstm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if self.predict:
            self.linear1.reset_parameters()
            self.linear2.reset_parameters()
        if hasattr(self, "distance_encoder"):
            nn.init.xavier_uniform_(self.distance_encoder.weight.data)

    def forward(self, data):

        # Projecting init node repr to d-dim space
        h = self.feature_encoder(data.x)

        # Looping over layers
        for i in range(self.cutoff - 1):
            if self.encode_distances:
                # distance encoding with shared distance embedding
                dist_emb = self.distance_encoder(getattr(data, f"sp_dists_{i+2}"))
            else:
                dist_emb = None

            # Euclidean normalization
            if self.l2_norm:
                h = F.normalize(h, p=2, dim=1)

            h = self.convs[i](h, getattr(data, f"path_{i+2}"), dist_emb)

        # Readout sum function
        h = global_add_pool(h, data.batch)

        # Prediction
        if self.predict:
            h = F.relu(self.linear1(h))
            h = F.dropout(h, training=self.training, p=self.dropout)
            h = self.linear2(h)
        return h


class PathConv(nn.Module):
    r"""
    The Path Aggregator module that computes result of Equation 2.
    """

    def __init__(
        self, hidden_dim, rnn: Callable, batch_norm: Callable, residuals=True, dropout=0
    ):
        super(PathConv, self).__init__()
        self.rnn = rnn
        self.bn = batch_norm
        self.hidden_dim = hidden_dim
        self.residuals = residuals
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.bn, "reset_parameters"):
            self.bn.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], paths, dist_emb=None):

        h = x[paths]

        # Add distance encoding if needed
        if dist_emb is not None:
            h = torch.cat([h, dist_emb], dim=-1)

        _, (h, _) = self.rnn(h)

        # Summing paths representations based on starting node
        h = scatter_add(
            h.squeeze(0),
            paths[:, -1],
            dim=0,
            out=torch.zeros(x.size(0), self.hidden_dim, device=x.device),
        )

        # Residual connection
        if self.residuals:
            h = self.bn(h + x)
        else:
            h = self.bn(h)

        # ReLU non linearity as the phi function
        h = F.relu(h)

        return h
