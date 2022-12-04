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


def load_label(label_dir_path, trial_n: int):
    # label [-1, 1] --> [0, 2]
    label = io.loadmat(label_dir_path + 'label.mat')['label']
    label = [1 + label[0][i] for i in range(trial_n)]
    return label


def load_data(data_dir_path, dsf=None, t=189):
    train_data_list = []
    test_data_list = []
    session_dirs = os.listdir(data_dir_path)

    for s, dirs in enumerate(session_dirs):
        data_dirs = data_dir_path + dirs + '/'
        file_list = os.listdir(data_dirs)

        trial_list = []
        for idx, file in enumerate(file_list):
            trial = []

            df = pd.read_csv(data_dirs + file)

            if dsf:
                cols = df.columns
                df = df.loc[:dsf * t - 1, cols]

            if 'label' in df.columns:
                tdf = df.drop(['in_tm', 'label'], axis=1).T
            else:
                tdf = df.drop(['in_tm'], axis=1).T

            tdf_len = tdf.values.shape[0]

            for i in range(tdf_len):
                trial.append(list(tdf.values[i]))

            trial_list.append(trial)

        if s == 0:
            test_data_list.extend(trial_list)
        else:
            train_data_list.extend(trial_list)

    return train_data_list, test_data_list


def feature_extraction(de_path, trial: int, feature_name, eeg, eog, gsr):
    de_list, s_list = [], []
    file_list = os.listdir(de_path)

    p, f = spectral_density((eeg, eog, gsr))
    for idx, file in enumerate(file_list):
        t_list = []
        features = io.loadmat(de_path + file)

        for t_idx in range(1, trial + 1):
            t_list.append(features[feature_name + str(t_idx)][:, :, :])

        s_list.append(t_list)

        if (idx + 1) % 3 == 0:
            de_list.append(s_list)
            s_list = []

    return de_list


def load_edge_information(pdc_dir_path, sub_name, pdc_var_name, n_channels, n_trials):
    file_list = os.listdir(pdc_dir_path)

    edge_index_list = []
    for i in range(n_channels):
        for j in range(n_channels):
            edge_index_list.append([i, j])

    edge_attr_list = []
    session_edge_attr_list = []
    sub_idx = 7
    for i, file in enumerate(file_list):
        if i < 21 or i > 23: continue
        trial_edge_attr_list = []
        pdcs = io.loadmat(pdc_dir_path + file)
        pdc_name = sub_name[sub_idx] + pdc_var_name
        for trial_idx in range(1, n_trials + 1):
            edge_attr = []
            pdc = pdcs[pdc_name + str(trial_idx)][:, :]
            for k in range(n_channels):
                for l in range(n_channels):
                    if k == l:
                        edge_attr.append(0)
                    else:
                        edge_attr.append(pdc[k][l])

            trial_edge_attr_list.append(edge_attr)
        session_edge_attr_list.append(trial_edge_attr_list)
        if (i + 1) % 3 == 0:
            sub_idx += 1
            edge_attr_list.append(session_edge_attr_list)
            session_edge_attr_list = []

    return edge_index_list, edge_attr_list


# Graph Representation
def get_graph_data(eeg_data, eog_data, gsr_data, label, num_nodes, num_train_trials, batch_size):
    # Utilizing subfiles for feautures
    de_path = "./data/sub_files/de/"
    feature_name = "de_LDS"
    pdc_path = "./data/sub_files/pdcs_nodiag/"
    sub_name = ['djc', 'jl', 'jj', 'lqj', 'ly', 'mhw', 'phl', 'sxy', 'wk', 'ww', 'wsf', 'wyw', 'xyl', 'ys', 'zjy'];
    pdc_var_name = '_PDC_mean'
    s = 7
    num_subjects = 8

    subject_data = feature_extraction(de_path, trials, feature_name, eeg_data, eog_data, gsr_data)
    num_sessions = len(subject_data[0])
    num_trials = len(subject_data[0][0])

    edge_index_list, edge_attr_list = load_edge_information(pdc_path, sub_name, pdc_var_name, num_nodes, num_trials)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    train_loader, test_loader = [], []

    for subject in range(s, num_subjects):
        train_dataset, test_dataset = [], []
        for session in range(num_sessions):
            for trial in range(num_trials):
                data_list = []
                trial_data = subject_data[subject][session][trial]
                blocks = len(trial_data[1])
                edge_attr = torch.tensor(edge_attr_list[0][session][trial], dtype=torch.float)
                for block_idx in range(blocks):
                    data_sample = torch.tensor(trial_data[:, block_idx, :], dtype=torch.float)
                    data_label = torch.tensor(label[trial], dtype=torch.long)
                    data_list.append(
                        Data(x=data_sample, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=data_label))
                if trial < num_train_trials:
                    train_dataset.extend(data_list)
                else:
                    test_dataset.extend(data_list)

        random.shuffle(train_dataset)
        random.shuffle(test_dataset)

        batch_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        batch_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        train_loader.append(batch_train_loader)
        test_loader.append(batch_test_loader)
        print('loading... ' + sub_name[7] + '_graph_data')
    print("\nTrain dataset length: {}, \tTest dataset legnth: {}".format(len(train_dataset), len(test_dataset)))
    return train_loader[0], test_loader[0], train_loader[0].dataset[0].edge_attr

