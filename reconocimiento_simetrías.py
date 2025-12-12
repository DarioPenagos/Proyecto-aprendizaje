import multiprocessing as mp
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as utils
from torch import nn


class SimpleNet(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.linear1 = nn.Linear(ni, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 64)
        self.linear5 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = F.tanh(self.linear4(x))
        x = self.linear5(x)
        return x


def P1(f, X, n):
    X[:, n] *= -1
    return -f(X)


def P0(f, X, n):
    X[n] *= -1
    return f(X)


bs = 1028

sim_dict = {"P1": P1, "P0": P0}

sim_df = pd.read_csv("Invariante_por_Funcion - Hoja 1.csv")

for files in sim_df["Function"]:
    Losses = []
    for title in os.listdir(f"NN_models/{files}/"):
        Losses.append(float(re.search("loss_(.+?).pt", title).group(1)))
    file = f"loss_{sorted(Losses)[0]}.pt"

    txt = np.loadtxt(f"Feynman_with_units/{files}")

    n_variables = txt.shape[1] - 1
    variables = txt[:, :-1]
    f_dependent = txt[:, [-1]]

    factors = torch.from_numpy(variables)
    factors = factors.float()
    product = torch.from_numpy(f_dependent)
    product = product.float()
    my_dataset = utils.TensorDataset(factors, product)
    my_dataloader = utils.DataLoader(my_dataset, batch_size=bs, shuffle=True)

    model = SimpleNet(n_variables)
    model.load_state_dict(torch.load(f"NN_models/{files}/{file}")["model_state_dict"])
    model.eval()

    dist = nn.PairwiseDistance()
    for x, y in my_dataloader:
        print(dist(P1(model, x.clone(), 0), model(x)).mean())
        print(dist(P0(model, x.clone(), 0), model(x)).mean())
