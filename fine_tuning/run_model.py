import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch import nn
import torch.optim.adam
from torch.utils.data import TensorDataset, DataLoader
from models import simple_network

def run_model(model, epochs, train_dataloader, test_dataloader, loss_fn= nn.BCELoss(), optimizer = torch.optim.Adam, device = "cuda:0"):
    
    model = model.to(device)
    optimizer(model.parameters())

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # print(batch)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.round(), y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            # pred is now a float
            # should be converted to int
            # then update all the shit

            correct += (pred.round() == y).type(torch.float).sum().item()
            # print((pred == y).shape)
            # print(pred.shape)
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def make_dataloaders(input):
    df = pd.read_csv(input, sep="\t", header=0, index_col=0)
    df.em_x = df.em_x.apply(eval)
    df.em_y = df.em_y.apply(eval)

    samples = df.em_x + df.em_y
    y = df.similar.astype(int).values
    x = np.array(samples.values.tolist())

    x, x_val, y, y_val = train_test_split(x, y, test_size=0.30, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    train_dataloader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            ), 
        batch_size=16
        )
    
    test_dataloader = DataLoader(
        TensorDataset(
            torch.tensor(x_test, dtype=torch.float32), 
            torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
            ), 
        batch_size=16
        )
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--input", type=str, default="simple")
    parser.add_argument("--model", type=str, default="simple")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    input = Path(args.input)
    model_names = {"simple": simple_network}
    if args.model in model_names:
        model = model_names[args.model]
    else: raise ValueError(f"cannot find model with name {args.model}")
    
    epoch = args.epoch
    output_path = Path(args.output)
    make_dataloaders(input, )
    # run_model(model, )
