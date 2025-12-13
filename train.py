import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.utils.data as utils
from torch import optim
import torch.nn.functional as F
import os
import multiprocessing as mp

# Parametros del proceso de entrenamiento
epochs = 1000
bs = 2048
N_red_lr = 4
lrs = 1e-2

# Red neuronal utilizada
class SimpleNet(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.linear1 = nn.Linear(ni, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64,64)
        self.linear5 = nn.Linear(64,1)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = F.tanh(self.linear4(x))
        x = self.linear5(x)
        return x


# Leyendo los datos y escogiendo un problema
df = pd.read_csv("FeynmanEquations.csv")
names = df["Filename"].dropna()
names = names[~names.isin(os.listdir("NN_models"))]

names = names[names.isin(os.listdir("Feynman_with_units"))]


def train(proc_id, name_set):
    epochs = 1000
    bs = 2048
    N_red_lr = 4
    lrs = 1e-2
    for name in name_set:
        print(f"id: {proc_id}, working on: {name}")
        txt = np.loadtxt(f"Feynman_with_units/{name}")

        n_variables = txt.shape[1]-1
        variables = txt[:,:-1]
        f_dependent = txt[:,[-1]]

        factors = torch.from_numpy(variables)
        factors = factors.float()
        product = torch.from_numpy(f_dependent)
        product = product.float()
        my_dataset = utils.TensorDataset(factors,product)
        my_dataloader = utils.DataLoader(my_dataset, batch_size=bs, shuffle=True)

        model_feynman = SimpleNet(n_variables)
        check_es_loss = 10_000
        filename = "modelo1"
        os.makedirs(f"NN_models/{name}", exist_ok=True)


        for i_i in range(N_red_lr):
            optimizer_feynman = optim.Adam(model_feynman.parameters(), lr = lrs)
            for epoch in range(epochs):
                model_feynman.train()
                for i, data in enumerate(my_dataloader):
                    optimizer_feynman.zero_grad()
                
                    fct = data[0].float()
                    prd = data[1].float()
                    
                    loss = F.mse_loss(model_feynman(fct),prd)
                    loss.backward()
                    optimizer_feynman.step()

                print(f"Epoch: {epoch}; Loss: {loss.item()}")
                # Early stopping
                if epoch%20==0 and epoch>0:
                    if check_es_loss < loss:
                        break
                    else:
                        torch.save({
                                "model_state_dict": model_feynman.state_dict(),
                                "epoch": epoch+1,
                                "loss": loss.item(),
                            }, f"NN_models/{name}/loss_{loss.item()}.pt")
                        check_es_loss = loss
                if epoch==0:
                    if check_es_loss < loss:
                        torch.save({
                                "model_state_dict": model_feynman.state_dict(),
                                "epoch": epoch+1,
                                "loss": loss.item(),
                            }, f"NN_models/{name}/loss_{loss.item()}.pt")
                        check_es_loss = loss

            lrs = lrs/10


def main():
    num_procs = 4

    procs = []

    for i in range(num_procs):
        name_set = names.loc[i::2]
        p = mp.Process(target=train, args=(i, name_set))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

if __name__ == "__main__":
    main()

