import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import numpy as np
import torch
from torch import nn
import torch.optim.adam
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import simple_network, from_evo, small_network, mini_network


def train_model(model, epochs, train_dataloader, test_dataloader, checkpoint_path, loss_fn= nn.BCELoss(), optimizer = torch.optim.Adam, device = "cuda:0", verbose=False):
    
    model = model().to(device)
    optimizer = optimizer(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr=1e6)

    best_acc = 0
    history = []
    best_epoch = 0
    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        train_a, train_l = run_train(train_dataloader, model, loss_fn, optimizer, device, verbose)
        test_acc, test_l = run_test(test_dataloader, model, loss_fn, device)
        scheduler.step(test_l)
        history.append({"epoch":t, "accuracy": train_a, "loss": train_l, "phase": "train"})
        history.append({"epoch":t, "accuracy": test_acc, "loss": test_l, "phase": "test"})
        if test_acc > best_acc:
            print(f"new high! {test_acc}")
            torch.save(model.state_dict(), checkpoint_path)
            best_acc = test_acc
            best_epoch = t
        
        print(f"Epoch: {t}, train a/l: {train_a:>0.3f}/{train_l:>0.3f} test a/l: {test_acc:>0.3f}/{test_l:>0.3f}")
    print(f"Best acc: {best_acc}, best epoch: {best_epoch}")

    print("Done!")
    return model, history


def run_train(dataloader, model, loss_fn, optimizer, device, v):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    total_accuracy, total_loss = 0, 0
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)
        if v: print(f"shape X : {X.shape}")
        if v: print(f"X : {X}")

        if v:print(f"shape y : {y.shape}")
        if v:print(f"y : {y.tolist()}")

        pred = model(X)

        if v:print(f"shape pred: {pred.shape}")
        if v:print(f"pred: {pred}")
        if v:print(f"pred round: {pred.round().tolist()}")
        if v:print(f"pred accuracy: {(pred.round() == y).tolist()}")

        # Should I add round here aswell ?
        loss = loss_fn(pred, y)
        if v:print(f"loss: {loss.tolist()}")
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
    
def plot_auc(preds, labels, out, title=""):
    viz = RocCurveDisplay.from_predictions(labels, preds, pos_label=True)
    viz.ax_.set_title(title)
    viz.plot()
    plt.savefig(str(out)+"_auc_val.png", dpi=300, bbox_inches="tight")

def plot_history(hist, out, title=""):
    df = pd.DataFrame(hist)
    plt.figure()
    lp = sns.lineplot(data=df, x="epoch", y="accuracy", hue="phase")
    lp.set_title(title)
    lp.figure.savefig(str(out)+"_accuracy.png")
    
    plt.figure()
    lp = sns.lineplot(data=df, x="epoch", y="loss", hue="phase")
    lp.set_title(title)
    lp.figure.savefig(str(out)+"_loss.png")

def make_dataloaders_old(input, batch_size, sample=None, name="", v=False):
    df = pd.read_csv(input, sep="\t", header=0, index_col=0)
    print(df.shape)
    if name:
        df1 = df.query('similar == False').sample(n=int(sample/2), random_state=42)
        df2 = df.query('similar == True').sample(n=int(sample/2), random_state=42)
        df = pd.concat([df1, df2])
        df.to_csv(name, sep="\t")
    
    df.em_x = df.em_x.apply(eval)
    df.em_y = df.em_y.apply(eval)

    samples = df.em_x + df.em_y
    x = np.array(samples.values.tolist())
    scaler = StandardScaler()
    scaler.fit(x)
    x_scaled = scaler.transform(x)
    y = df.similar.astype(int).values   

    if v: print(f"len x : {len(x)}")
    if v: print(f"x : {x}")
    if v: print(f"len y : {len(y)}")
    if v: print(f"y : {y}")
    dl = DataLoader(
        TensorDataset(
            torch.tensor(x_scaled, dtype=torch.float32), 
            torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            ), 
        batch_size=batch_size, shuffle=True
        )
    
    return dl

def make_dataloaders(embeddings, batch_size, df = None, input_path = "", sample=None, name="", v=False):
    if input_path != "":
        df = pd.read_csv(input_path, sep="\t", header=0, index_col=0)
    else:
        df = df
        
    em = pd.read_csv(embeddings, sep="\t", header=0, index_col=0)
    em.embedding = em.embedding.apply(eval)
    
    # if name:
    #     df1 = df.query('similar == False').sample(n=int(sample/2), random_state=42)
    #     df2 = df.query('similar == True').sample(n=int(sample/2), random_state=42)
    #     df = pd.concat([df1, df2])
    #     df.to_csv(name, sep="\t")

    df = df.merge(em,
                    left_on="segment_id_x",
                    right_on="segment_id",
                    ).drop(columns=['segment_id']).rename({"embedding":"em_x"}, axis=1)
    df = df.merge(em,
                    left_on="segment_id_y",
                    right_on="segment_id",
                    ).drop(columns=['segment_id']).rename({"embedding":"em_y"}, axis=1)

    # display(df.head())
    samples = df.em_x + df.em_y
    samples = np.array(samples.values.tolist())
    

    scaler = StandardScaler()
    scaler.fit(samples)
    x_scaled = scaler.transform(samples)
    y = df.similar.astype(int).values   

    if v: print(f"len x : {len(x_scaled)}")
    if v: print(f"x : {x_scaled[0]}")
    if v: print(f"len y : {len(y)}")
    if v: print(f"y : {y[0]}")
    dl = DataLoader(
        TensorDataset(
            torch.tensor(x_scaled, dtype=torch.float32), 
            torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            ), 
        batch_size=batch_size, shuffle=True
        )
    
    return dl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--model", type=str, default="simple")
    parser.add_argument('--train', type=str, default="data/aar_ath_bol_chi_cpa_tha/sm7_50000_train.tsv")
    parser.add_argument('--test', type=str, default="data/aar_ath_bol_chi_cpa_tha/sm7_50000_test.tsv")
    parser.add_argument('--val', type=str, default="data/aar_ath_bol_chi_cpa_tha/sm7_50000_val.tsv")
    parser.add_argument('--embeddings', type=str, default="data/aar_ath_bol_chi_cpa_tha/sm7_50000_avg_embeddings.tsv")
    parser.add_argument('--output', type=str, default="results/testing/sm7_500000")
    parser.add_argument('--checkpoint', type=str, default="saved_models/sm7_500000")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--run_val', type=bool, default=True)
    parser.add_argument('--device', type=str, default="cpu")

    args = parser.parse_args()
    
    model_str = args.model

    model_names = {
        "simple": simple_network,
        "small": small_network,
        "mini": mini_network,
        "from_evo": from_evo
        }

    if model_str in model_names:
        model = model_names[model_str]
    else: raise ValueError(f"cannot find model with name {model_str}")

    model_str = "simple"
    train_path = Path(args.train)
    test_path = Path(args.test)
    val_path = Path(args.val)
    embeddings_path = Path(args.embeddings)

    output_prefix = Path(f"{args.output}_{model_str}")
    checkpoint_path = Path(f"{args.checkpoint}_{model_str}")

    batch_size = args.batch_size
    epochs = args.epoch
    validation = args.run_val
    device = args.device

    print("Making train loader")
    train_loader = make_dataloaders(
        input_path=train_path, 
        embeddings=embeddings_path, 
        batch_size=batch_size
        )

    print("Making test loader")
    test_loader = make_dataloaders(
        input_path=test_path, 
        embeddings=embeddings_path, 
        batch_size=batch_size
        )


    model, history = train_model(
        model=model, 
        epochs=epochs, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        checkpoint_path=checkpoint_path,
        verbose=False,
        device=device
        )
    plot_history(history, output_prefix)

    model = model_names[model_str]
    model = model().to("cpu")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()

    if validation:
        print("Making val loader")
        val_loader = make_dataloaders(
            input_path=val_path,
            embeddings=embeddings_path,
            batch_size=batch_size)
        pred, labels = run_validation(model, val_loader, device=device)
        plot_auc(pred, labels, output_prefix)