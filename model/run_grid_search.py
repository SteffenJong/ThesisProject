import argparse
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score
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
from sklearn.model_selection import ParameterGrid
from gzip import BadGzipFile
from models import modular_network


def train_model(model, epochs, train_dataloader, test_dataloader, loss_fn= nn.BCELoss(), optimizer = torch.optim.Adam, device = "cuda:0", verbose=False):
    
    model = model.to(device)
    optimizer = optimizer(model.parameters(), lr=0.0001)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr=1e6)

    best_acc = 0
    best_loss = 9999999999
    history = []
    best_epoch = 0
    counter = 0
    for t in range(epochs):
        print(f"\rCurrent epoch: {t}", end="", flush=True)
        train_a, train_l = run_train(train_dataloader, model, loss_fn, optimizer, device, verbose)
        test_acc, test_l = run_test(test_dataloader, model, loss_fn, device)
        # scheduler.step(test_l)
        history.append({"epoch":t, "accuracy": train_a, "loss": train_l, "phase": "train"})
        history.append({"epoch":t, "accuracy": test_acc, "loss": test_l, "phase": "test"})
        if test_l < best_loss:
            # print(f"New High! Epoch: {t}, train a/l: {train_a:>0.3f}/{train_l:>0.3f} test a/l: {test_acc:>0.3f}/{test_l:>0.3f}")
            best_loss = test_l
            best_acc = test_acc
            best_epoch = t
            counter = 0
        else:
            counter +=1
            if counter > 200:
                print("\nEarly stopping!")
                break

    print(f"Best loss: {best_loss}, acc: {best_acc}, epoch: {best_epoch}")

    # print("Done!")
    return model, history, best_loss, best_epoch


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

        loss = loss_fn(pred, y)
        if v:print(f"loss: {loss.tolist()}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss +=  loss.item()
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
    model = model.to(device)
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
    # print(f"validation Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return prediction, labels
    
def plot_auc_old(preds, labels, out, title=""):
    viz = RocCurveDisplay.from_predictions(labels, preds, pos_label=True)
    viz.ax_.set_title(title)
    viz.plot()
    print(viz.ax_.get_label)
    plt.savefig(str(out)+"auc_val.png", dpi=300, bbox_inches="tight")

def plot_auc(preds, labels, out, title=""):
    plt.clf()
    sns.set_theme(rc={'figure.figsize':(5,5)})
    plt.figure()
    fpr, tpr, tresh = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)

    plt.plot(fpr, tpr, label=f"AUC: {round(auc, 2)}")
    plt.legend()
    plt.title(title)
    plt.xlabel("False postive rate")
    plt.ylabel("True postive rate")
    plt.savefig(out/"auc_val.png", dpi=300)
    plt.show()
    plt.clf()
    return auc

def plot_history(df, out, title=""):
    plt.clf()
    plt.figure()
    sns.set_theme()
    plt.figure()
    lp = sns.lineplot(data=df, x="epoch", y="accuracy", hue="phase")
    lp.set_title(title)
    lp.figure.savefig(out/"accuracy.png",bbox_inches='tight')

    plt.figure()
    lp = sns.lineplot(data=df, x="epoch", y="loss", hue="phase")
    lp.set_title(title)
    lp.figure.savefig(out/"loss.png",bbox_inches='tight')
    

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
    
    try:
        em = pd.read_csv(embeddings, sep="\t", header=0, index_col=0, compression="gzip")
    except BadGzipFile as e:
        em = pd.read_csv(embeddings, sep="\t", header=0, index_col=0)

    em.embedding = em.embedding.apply(eval)

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

def make_dataloaders_old(input_path, batch_size, sample=None, v=False):
    df = pd.read_csv(input_path, sep="\t", header=0, index_col=0)
    
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

def run_grid_search(parm_grid, train_seq_path, val_seq_path, test_seq_path, results_folder, saved_models_folder, results_file, embeddings_path, device):
    best_model_name = Path("")
    best_model_auc = 0
    res = []

    for p in ParameterGrid(parm_grid):
        print("Starting:", ''.join([f'_{k[0]}_{v}' for k, v in p.items()]))
        output_suffix = f"{prefix.stem}{ ''.join([f'_{k[0]}_{v}' for k, v in p.items()])}"
        output_prefix = results_folder / output_suffix
        checkpoint_path = saved_models_folder / output_suffix
        model = modular_network(p["n_layers"], p["drop_out"])

        # print("Making train loader")
        train_loader = make_dataloaders(input_path=train_seq_path, 
                                        embeddings=embeddings_path, 
                                        batch_size=p["batch_size"])

        # print("Making test loader")
        test_loader = make_dataloaders(input_path=test_seq_path, 
                                        embeddings=embeddings_path, 
                                        batch_size=p["batch_size"])
        
        model, history, best_l, epoch = train_model(model, epochs, train_loader, test_loader, verbose=False, device=device)
        model.eval()
        
        # print("Making val loader")
        val_loader = make_dataloaders(input_path=val_seq_path, 
                                          embeddings=embeddings_path, 
                                          batch_size=p["batch_size"])
        pred, labels = run_validation(model, val_loader, device=device)
        
        auc = roc_auc_score(labels, pred)
        print(f"AUC: {auc}")

        if auc > best_model_auc:
            best_model_auc = auc
            if best_model_name.is_file():
                os.remove(best_model_name)
            torch.save(model.state_dict(), checkpoint_path)
            best_model_name = checkpoint_path
            
        r = p.copy()
        r["val_auc"] = auc
        r["t_loss"] = best_l
        r["epoch"] = epoch
        r["model_name"] = checkpoint_path
        better_hist = {}
        
        # saves a bit of storage and looks nicer in tsv file
        for d in history:
            for key, value in d.items():
                better_hist.setdefault(key, []).append(value)
        r["history"] = better_hist
        res.append(r)
        print("")

    df = pd.DataFrame(res)
    df.to_csv(results_file, sep="\t")
    return df

def plot_best(df, test_seq_path, embeddings_path, output_prefix, device):
    r = df.iloc[df['val_auc'].idxmax()]
    model = modular_network(r["n_layers"], r["drop_out"])
    model.load_state_dict(torch.load(r["model_name"], weights_only=True))
    model.eval()
    val_loader = make_dataloaders(input_path=test_seq_path, embeddings=embeddings_path, batch_size=int(r["batch_size"]))
    print("Running validation")
    pred, labels = run_validation(model, val_loader, device=device)
    plot_auc(pred, labels, output_prefix)
    df_hist = pd.DataFrame(r.history)
    plot_history(df_hist, output_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--prefix", type=str)
    parser.add_argument('--batches', nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--layers', nargs='+', default=[1, 2, 3])
    parser.add_argument('--dropout', nargs='+', default=[0.1, 0.3, 0.5])
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument('--device', type=str, default="cpu")

    args = parser.parse_args()
    prefix = Path(args.prefix)
    epochs = args.max_epochs
    device = args.device
    print(f"running on {device}")

    parm_grid = {
        "batch_size": [int(f) for f in args.batches],
        "n_layers": [int(f) for f in args.layers],
        "drop_out": [float(f) for f in args.dropout],
        }

    train_seq_path = Path(f"{prefix}_train.tsv" )
    test_seq_path = Path(f"{prefix}_test.tsv" )
    val_seq_path = Path(f"{prefix}_val.tsv" )

    results_folder = Path(f"results/{prefix.stem}")
    if not results_folder.is_dir():
        results_folder.mkdir()

    saved_models_folder = Path(f"saved_models/{prefix.stem}")
    if not saved_models_folder.is_dir():
        saved_models_folder.mkdir()

    results_file = results_folder / Path(f"{prefix.stem}_gridsearch_results.tsv")
    embeddings_path = Path(f"{prefix}_embeddings.tsv")

    df = run_grid_search(parm_grid, train_seq_path, val_seq_path, test_seq_path, results_folder, saved_models_folder, results_file, embeddings_path, device)
    plot_best(df, test_seq_path, embeddings_path, results_folder, device)

    