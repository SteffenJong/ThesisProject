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
print("importing torch")
import torch
print("importing evo2")
from evo2 import Evo2

def create_embeddings(df: pd.DataFrame, l_name:str = "blocks.24.mlp.l3", axis=0):
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

def add_padding(row:pd.Series, pad_char:str= "0"):
    max_len = max(len(row["seq_x"]), len(row["seq_y"]))
    x = row['seq_x'] + pad_char*(max_len - len(row['seq_x']))
    y = row['seq_y'] + pad_char*(max_len - len(row['seq_y']))
    return pd.Series([x, y])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--train_test", type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--l_name', type=str, default='blocks.24.mlp.l3')

    args = parser.parse_args()
    input = Path(args.train_test)
    output = Path(args.output)
    l_name = args.l_name

    df = pd.read_csv(input, sep="\t", header=0)

    df = create_embeddings(df, l_name)
    df.to_csv(output, sep='\t', index=False)