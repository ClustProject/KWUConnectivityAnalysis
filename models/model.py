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


def criterion(loss, label, model, L1_regularization_scaler):
    l1_regularization = torch.tensor(0., device=device)

    for param in model.parameters():
        l1_regularization += torch.norm(param, 1)

    loss += L1_regularization_scaler * l1_regularization

    return loss


def train(loader, model, optimizer, L1_regularization_scaler, epoch, batch_size):
    model.train()
    train_acc, train_loss, count = 0., 0., 0.

    dataset_length = len(loader.dataset)
    loader_length = len(loader)
    for batch_idx, data in enumerate(loader):
        if len(data.y) == batch_size:  # dataset size가 batch size로 안나눠지면 버림
            count += 1.
            data = data.to(device)
            loss, result = model(data, 'train')
            loss = criterion(loss, data.y, model, L1_regularization_scaler)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            pred = result.max(1, keepdim=True)[1]
            acc = pred.eq(data.y.view_as(pred)).sum().item()
            acc = 100. * acc / batch_size

            train_acc += acc

    train_acc /= count
    train_loss /= count
    return train_acc, train_loss


def test(loader, model, L1_regularization_scaler, batch_size):
    model.eval()
    count, test_loss, test_acc = 0, 0, 0
    with torch.no_grad():
        for data in loader:
            if len(data.y) == batch_size:
                count += 1
                data = data.to(device)
                output, result = model(data)
                loss = output.item()
                test_loss += loss

                pred = result.max(1, keepdim=True)[1]
                acc = pred.eq(data.y.view_as(pred)).sum().item()
                acc = 100. * acc / batch_size
                test_acc += acc

    test_acc /= count
    test_loss /= count
    return test_acc, test_loss


def get_confusion_matrix(loader, model, L1_regularization_scaler, batch_size, class_n):
    model.eval()

    cfm = np.zeros((class_n, class_n), int)
    tp, tgt = [], []

    with torch.no_grad():
        for data in loader:
            if len(data.y) == batch_size:
                data = data.to(device)
                _, result = model(data)
                p = result.max(1, keepdim=True)[1]
                t = data.y.view_as(p)
                pred = p.detach().cpu().numpy()
                gt = t.detach().cpu().numpy()
                cfm += confusion_matrix(gt, pred)

                tp.extend(copy.deepcopy(pred.tolist()))
                tgt.extend(copy.deepcopy(gt.tolist()))

    tp_tgt = np.concatenate((np.array(tp), np.array(tgt)), axis=1)
    return cfm, tp_tgt

