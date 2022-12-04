from scipy import io, linalg, fftpack
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random
import math
import copy
import os

import torch
import torch.nn as nn

from torch_geometric.nn import ChebConv, GCNConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import LaplacianLambdaMax

# pip install torch-geometric torch


class EER_GCN(nn.Module):  # ChebNet with Batch Normalization and LeakyReLU
    def __init__(self, num_nodes, num_features, hid_channels, fc1_out_channels, out_channels, k, edge_weight,
                 batch_size, learnable=False):
        super(EER_GCN, self).__init__()

        self.lambdamax = LaplacianLambdaMax(None)

        self.in_channels = num_features
        self.cheb_out_channels = hid_channels

        self.fc1_in_channels = hid_channels * num_nodes
        self.fc1_out_channels = fc1_out_channels
        self.out_channels = out_channels

        self.edge_weight = nn.Parameter(edge_weight, requires_grad=learnable)
        self.batch_size = batch_size
        self.num_nodes = num_nodes

        self.chebconv1 = ChebConv(self.in_channels, self.cheb_out_channels, K=k, normalization=None)

        self.BN1d1 = nn.BatchNorm1d(self.in_channels)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.15)

        self.fc1 = nn.Linear(self.fc1_in_channels, self.fc1_out_channels)
        self.fc2 = nn.Linear(self.fc1_out_channels, self.out_channels)
        self.fc_module = nn.Sequential(self.fc1, self.leakyrelu, self.fc2)

    def forward(self, data, _type=None):
        data.edge_attr = self.edge_weight.data.repeat(self.batch_size)
        data = self.lambdamax(data)

        if data.x.dim() == 1:
            data.x = data.x.unsqueeze(dim=1)

        data.x = self.leakyrelu(self.BN1d1(data.x))
        cheb_layer = self.chebconv1(data.x, data.edge_index, self.leakyrelu(data.edge_attr), lambda_max=data.lambda_max)
        cheb_layer = self.leakyrelu(cheb_layer).reshape(self.batch_size, -1)
        fc_layer = self.fc_module(cheb_layer)

        if _type == 'train':
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(fc_layer.view(-1, self.out_channels), data.y.view(-1))
            logits = self.softmax(fc_layer)
            return loss, logits
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(fc_layer.view(-1, self.out_channels), data.y.view(-1))
            logits = self.softmax(fc_layer)
            return loss, logits
