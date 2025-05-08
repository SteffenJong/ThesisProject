import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import numpy as np
import torch
from torch import nn
import torch.optim.adam
from torch.utils.data import TensorDataset, DataLoader
from models import simple_network

def train_model(model, epochs, train_dataloader, test_dataloader, loss_fn= nn.BCELoss(), optimizer = torch.optim.Adam, device = "cuda:0"):
    
    model = model().to(device)
    optimizer = optimizer(model.parameters())

    history = []
    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        train_a, train_l = run_train(train_dataloader, model, loss_fn, optimizer, device)
        test_acc, test_l = run_test(test_dataloader, model, loss_fn, device)
        history.append({"epoch":t, "accuracy": train_a, "loss": train_l, "phase": "train"})
        history.append({"epoch":t, "accuracy": test_acc, "loss": test_l, "phase": "test"})
        print(f"Epoch: {t}, train a/l: {train_a:>0.3f}/{train_l:>0.3f} test a/l: {test_acc:>0.3f}/{test_l:>0.3f}")


    print("Done!")
    return model, history


def run_train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    total_accuracy, total_loss = 0, 0
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)
        print(f"shapeX : {X.shape}")
        print(f"X : {X}")

        print(f"shapeX : {y.shape}")
        print(f"X : {y}")

        pred = model(X)

        print(f"shape pred: {pred.shape}")
        print(f"pred: {pred}")

        # Should I add round here aswell ?
        loss = loss_fn(pred, y)
        # print(loss.shape)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss +=  loss.item()
        # print(pred)
        
        total_accuracy += (pred.round() == y).type(torch.float).sum().item()
        
        # if batch % 20 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     accuracy = (pred.round() == y).type(torch.float).sum().item() / size
        #     print(f"loss: {loss:>7f}, accuracy:{100*accuracy:>0.1f}  [{current:>5d}/{size:>5d}]")

    total_loss /= num_batches
    total_accuracy /= size
    return total_accuracy, total_loss

def run_test(dataloader, model, loss_fn, device):

    

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
    
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.round() == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct, test_loss

def run_validation(model, dataloader,  loss_fn= nn.BCELoss(), device = "cuda:0"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    prediction = []
    labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
    
            pred = model(X)
            prediction.extend(pred.cpu())
            labels.extend(y.cpu())

            test_loss += loss_fn(pred, y).item()
            correct += (pred.round() == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"validation Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return prediction, labels
    
def plot_auc(preds, labels, out):
    viz = RocCurveDisplay.from_predictions(labels, preds, pos_label=True)
    viz.plot()
    plt.savefig(str(out)+"auc_val.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_history(hist, out):
    df = pd.DataFrame(hist)
    # print(df.head())
    plt.figure()
    lp = sns.lineplot(data=df, x="epoch", y="accuracy", hue="phase")
    lp.figure.savefig(str(out)+"accuracy.png")
    
    plt.figure()
    lp = sns.lineplot(data=df, x="epoch", y="loss", hue="phase")
    lp.figure.savefig(str(out)+"loss.png")


def make_dataloaders(input, batch_size, sample=None, name=""):
    df = pd.read_csv(input, sep="\t", header=0, index_col=0)
    if sample:
        df = df.sample(n=sample, random_state=42)
        df.to_csv(name, sep="\t")
    df.em_x = df.em_x.apply(eval)
    df.em_y = df.em_y.apply(eval)

    samples = df.em_x + df.em_y
    x = np.array(samples.values.tolist())
    y = df.similar.astype(int).values

    dl = DataLoader(
        TensorDataset(
            torch.tensor(x, dtype=torch.float32), 
            torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            ), 
        batch_size=batch_size
        )
    
    return dl

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--train", type=str)
    parser.add_argument("--test", type=str)
    parser.add_argument("--val", type=str)

    parser.add_argument("--model", type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_prefix', type=str)
    parser.add_argument('--validation', type=bool, default=True)


    args = parser.parse_args()
    train_path = Path(args.train)
    test_path = Path(args.test)
    val_path = Path(args.val)
    batch_size = args.batch_size

    model_names = {"simple": simple_network}
    if args.model in model_names:
        model = model_names[args.model]
    else: raise ValueError(f"cannot find model with name {args.model}")
    
    epochs = args.epochs
    output_prefix = Path(args.output_prefix)
    print("Making train loader")
    train_loader = make_dataloaders(train_path, batch_size, 30)

    print("Making test loader")
    test_loader = make_dataloaders(test_path, batch_size, 30)
    
    model, history = train_model(model, epochs, train_loader, test_loader)
    plot_history(history, output_prefix)

    if args.validation:
        print("Making val loader")
        val_loader = make_dataloaders(val_path, batch_size)
        pred, labels = run_validation(model, val_loader)
        plot_auc(pred, labels, output_prefix)

