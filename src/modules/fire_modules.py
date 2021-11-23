from typing import Any, List
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import AUC, ConfusionMatrix, AUROC
import torch
from torch import nn
from pathlib import Path
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
import warnings
import pandas as pd
import sys
import gc
import geopandas as gpd
import rioxarray as rxr
from shapely.geometry import box
from affine import Affine
import random
from tqdm import tqdm
from .convlstm import ConvLSTM


class SimpleCNN(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        kernel_size = 3
        self.conv1 = nn.Conv2d(input_dim, hidden_size, kernel_size=kernel_size, stride=1, padding=1)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(kernel_size * kernel_size * hidden_size * 16, 16)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 8)
        self.drop3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop1(self.fc2(x)))
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x, dim=1)

    
class SimpleConvLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # lstm part
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        # clstm part
        self.convlstm = ConvLSTM(input_dim,
                                 hidden_size,
                                 (3, 3),
                                 lstm_layers,
                                 True,
                                 True,
                                 False)

        kernel_size = 3
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=1, padding=1)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(kernel_size * kernel_size * hidden_size * 16, 16)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, 8)
        self.drop3 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor):
        _, last_states = self.convlstm(x)
        x = last_states[0][0]
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.drop1(self.fc1(x)))
        x = F.relu(self.drop1(self.fc2(x)))
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x, dim=1)


class SimpleLSTM(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # lstm part
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        lstm_layers = hparams['lstm_layers']
        self.lstm = torch.nn.LSTM(input_dim, hidden_size, num_layers=lstm_layers, batch_first=True)
        # fully-connected part
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.drop1 = torch.nn.Dropout(0.5)

        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size , hidden_size // 2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc3 = torch.nn.Linear(hidden_size // 2, 2)

        self.fc_nn = torch.nn.Sequential(
            self.fc1,
            self.relu,
            self.drop1,
            self.fc2,
            self.relu,
            self.drop2,
            self.fc3
        )

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        x = self.fc_nn(lstm_out[:, -1, :])
        return torch.nn.functional.log_softmax(x, dim=1)


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths=None):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        if lengths:
            for i, l in enumerate(lengths):  # skip the first sentence
                if l < max_len:
                    mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions


class SimpleLSTMAttention(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        # lstm part
        input_dim = len(hparams['static_features']) + len(hparams['dynamic_features'])
        hidden_size = hparams['hidden_size']
        hidden_dim = hidden_size
        lstm_layers = hparams['lstm_layers']
        lstm_layer = 2
        self.dropout = nn.Dropout(p=0.5)
        self.lstm1 = nn.LSTM(input_size=input_dim,
                             hidden_size=hidden_size,
                             num_layers=lstm_layers,
                             bidirectional=True)
        self.atten1 = Attention(hidden_dim * 2, batch_first=True)  # 2 is bidrectional
        self.lstm2 = nn.LSTM(input_size=hidden_dim * 2,
                             hidden_size=hidden_dim,
                             num_layers=1,
                             bidirectional=True)
        self.atten2 = Attention(hidden_dim * 2, batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(hidden_dim * lstm_layer * 2, hidden_dim * lstm_layer * 2),
                                 nn.BatchNorm1d(hidden_dim * lstm_layer * 2),
                                 nn.ReLU())
        self.fc2 = nn.Linear(hidden_dim * lstm_layer * 2, 2)

    def forward(self, x):
        out1, (h_n, c_n) = self.lstm1(x)
        x, _ = self.atten1(out1)  # skip connect

        out2, (h_n, c_n) = self.lstm2(out1)
        y, _ = self.atten2(out2)

        z = torch.cat([x, y], dim=1)
        z = self.fc1(self.dropout(z))
        z = self.fc2(self.dropout(z))
        return torch.nn.functional.log_softmax(z, dim=1)