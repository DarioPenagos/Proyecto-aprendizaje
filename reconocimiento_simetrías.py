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
    X[:,n] *= -1
    return f(X)

def H(f,X,t): return f(t*X)*t**(-X.shape[1])

def CCY1(f,X,n,a):
    tmp = X.clone()
    X[:,n]+=a
    tmp[:,n]=a
    return f(X)-f(tmp)

def CCY2(f,X,n,a):
    tmp = X.clone()
    X[:,n]*=a
    tmp[:,n]=a
    return f(X)/f(tmp)

def CCY3(f,X,n,a):
    tmp = X.clone()
    X[:,n]+=a
    tmp[:,n]=a
    return f(X)/f(tmp)

def CCY4(f,X,n,a):
    tmp = X.clone()
    X[:,n]*=a
    tmp[:,n]=a
    return f(X)-f(tmp)

def T(f,X,n,a):
    X[:,n]+=a
    return f(X)

def I(f,X, p): return f(X[:,p])

def P(f,X,n,t):
    X**=t
    return f(X)/t

def ESC(f,X,n,a):
    X[:,n]*=a
    return f(X)/a


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
    my_dataset = utils.TensorDataset(factors)
    my_dataloader = utils.DataLoader(my_dataset, batch_size=len(my_dataset))

    model = SimpleNet(n_variables)
    model.load_state_dict(torch.load(f"NN_models/{files}/{file}")["model_state_dict"])
    model.eval()

    dist = nn.PairwiseDistance()
    for [x] in my_dataloader:
        eval_dist = lambda a: dist(a, model(x))
        
        for f in [P0,P1]:
            print(f"sim: {f.__name__}, dist: {eval_dist(f(model,x.clone(),0)).mean()}")
        
        for f in [H]:
            print(f"sim: {f.__name__}, dist: {eval_dist(f(model,x.clone(),3)).mean()}")
        
        for f in [CCY1,CCY2,CCY3,CCY4,T,P,ESC]:
            print(f"sim: {f.__name__}, dist: {eval_dist(f(model,x.clone(),0, 0.3)).mean()}")

        for f in [I]:
            print(f"sim: {f.__name__}, dist: {eval_dist(f(model,x.clone(),[0,1])).mean()}")
        




