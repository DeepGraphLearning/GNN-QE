from typing import Sequence


import torch
from torch import nn

from torchdrug import core, data, utils
from torchdrug.core import Registry as R

from . import layer


@R.register("model.NBFNet")
class NeuralBellmanFordNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="distmult", aggregate_func="pna",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, dependent=True):
        super(NeuralBellmanFordNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        num_relation = int(num_relation)
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation,
                                                               self.dims[0], message_func, aggregate_func, layer_norm,
                                                               activation, dependent))

    def forward(self, graph, input, all_loss=None, metric=None):
        with graph.node():
            graph.boundary = input
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        node_query = graph.query.expand(graph.num_node, -1, -1)
        if self.concat_hidden:
            node_feature = torch.cat(hiddens + [node_query], dim=-1)
        else:
            node_feature = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": node_feature,
        }


@R.register("model.CompGCN")
class CompositionalGraphConvolutionalNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="mult", short_cut=False, layer_norm=False,
                 activation="relu", concat_hidden=False):
        super(CompositionalGraphConvolutionalNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        num_relation = int(num_relation)
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.CompositionalGraphConv(self.dims[i], self.dims[i + 1], num_relation,
                                                            message_func, layer_norm, activation))
        self.relation = nn.Embedding(num_relation, input_dim)

    def forward(self, graph, input, all_loss=None, metric=None):
        graph.relation_input = self.relation.weight
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        node_query = graph.query.expand(graph.num_node, -1, -1)
        if self.concat_hidden:
            node_feature = torch.cat(hiddens + [node_query], dim=-1)
        else:
            node_feature = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": node_feature,
        }
