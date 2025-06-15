import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

from time import time
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
print("importing torch")
from sklearn.decomposition import PCA
import torch
print("importing evo2")
from evo2 import Evo2

def create_embeddings_old(df: pd.DataFrame, l_name:str = "blocks.24.mlp.l3", axis=0):
    """generate emebddings using evo2
    The embeddings are averaged over the tokens, so all will have the same length.

    Args:
        df (pd.DataFrame): dataframe containing sequence pairs, can be gotten from create training tsv
        l_name (str): layer name from wich to extract embeddings

    Returns:
        pd.Dataframe: tsv with embeddings
    """

    evo2_model = Evo2('evo2_1b_base', local_path="/home/jong505/models/evo2_1b_base.pt")
    em_x = []
    em_y = []

    # df[["seq_x", "seq_y"]] = df.apply(add_padding, axis=1)

    for seq_x, seq_y in tqdm(zip(df.seq_x, df.seq_y), total=df.shape[0]):

        tok_x = torch.tensor(evo2_model.tokenizer.tokenize(seq_x), dtype=torch.int, ).unsqueeze(0).to('cuda:0')
        _, tens_x = evo2_model(tok_x, return_embeddings=True, layer_names=[l_name])
        x = tens_x[l_name].cpu().float().numpy()

        tok_y = torch.tensor(evo2_model.tokenizer.tokenize(seq_y), dtype=torch.int, ).unsqueeze(0).to('cuda:0')
        _, tens_y = evo2_model(tok_y, return_embeddings=True, layer_names=[l_name])
        y = tens_y[l_name].cpu().float().numpy()
        
        em_x.append(np.mean(x[0], axis=axis).tolist())
        em_y.append(np.mean(y[0], axis=axis).tolist())

    df_embeddings = df.drop(["seq_x", "seq_y"], axis=1)
    df_embeddings["em_x"] = em_x
    df_embeddings["em_y"] = em_y
    return df_embeddings

def create_embeddings(df: pd.DataFrame, l_name:str = "blocks.24.mlp.l3", axis=0, reduction="avg", remove_dups=True):
    evo2_model = Evo2('evo2_1b_base', local_path="/home/jong505/models/evo2_1b_base.pt")
    x = df[["seq_x", "segment_id_x"]].copy()
    x.rename(columns={"seq_x": "seq", "segment_id_x": "segment_id"}, inplace=True)
    y = df[["seq_y", "segment_id_y"]].copy()
    y.rename(columns={"seq_y": "seq", "segment_id_y": "segment_id"}, inplace=True)
    df = pd.concat([x,y])
    df.drop_duplicates(subset="segment_id", inplace=True)
    
    results = {}

    for id, seq in tqdm(zip(df.segment_id.tolist(), df.seq.tolist()), total=df.shape[0]):
        if id in results: continue

        tok_x = torch.tensor(evo2_model.tokenizer.tokenize(seq), dtype=torch.int, ).unsqueeze(0).to('cuda:0')
        _, tens_x = evo2_model(tok_x, return_embeddings=True, layer_names=[l_name])
        x = tens_x[l_name].cpu().float().numpy()
        
        if reduction == "avg":
            results[id] = np.mean(x[0], axis=axis).tolist()
        elif reduction == "PCA":
            pca = PCA(n_components=1)
            results[id] = pca.fit_transform(x[0].T).flatten().tolist()
        elif reduction == "div3":
            split = np.array_split(x[0], 3)
            out = []
            for i in split:
                out += i.mean(axis=axis).tolist()
            results[id] = out
            
    df = pd.DataFrame(results.items(), columns=["segment_id","embedding"])
    return df

def add_padding(row:pd.Series, pad_char:str= "0"):
    max_len = max(len(row["seq_x"]), len(row["seq_y"]))
    x = row['seq_x'] + pad_char*(max_len - len(row['seq_x']))
    y = row['seq_y'] + pad_char*(max_len - len(row['seq_y']))
    return pd.Series([x, y])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--dataframes", nargs='+')
    parser.add_argument('--output', type=str)
    parser.add_argument('--l_name', type=str, default='blocks.24.mlp.l3')

    args = parser.parse_args()
    output = Path(args.output)
    l_name = args.l_name

    
    dataframes = [pd.read_csv(Path(df_path), sep="\t", header=0) for df_path in args.dataframes]
    df = pd.concat(dataframes)
    df = create_embeddings(df, reduction="PCA")
    df.to_csv(output, sep="\t")