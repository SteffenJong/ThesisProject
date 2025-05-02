import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
from time import time
s = time()
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
print(time()-s)

print("importing torch")
s = time()
import torch
print(time()-s)
print("importing evo2")
s = time()
from evo2 import Evo2
print(time()-s)

def main(df: pd.DataFrame, l_name):
    evo2_model = Evo2('evo2_1b_base')
    df[["seq_x", "seq_y"]] = df.apply(add_padding, axis=1)
    print(f"sim is {len(df.iloc[0]['seq_x']) == len(df.iloc[0]['seq_y'])}")
    sim = []
    sim_row = []
    sim_avg = []

    for _, r in tqdm(df.iterrows(), total=df.shape[0]):
        seq_x = r["seq_x"]
        seq_y =  r["seq_y"]
        # print(torch.cuda.memory_reserved(0))
        # print(torch.cuda.memory_allocated(0))
        # print(len(seq_x))
        
        tok_x = torch.tensor(evo2_model.tokenizer.tokenize(seq_x), dtype=torch.int, ).unsqueeze(0).to('cuda:0')
        _, tens_x = evo2_model(tok_x, return_embeddings=True, layer_names=[l_name])
        x = tens_x[l_name].cpu().float().numpy()

        tok_y = torch.tensor( evo2_model.tokenizer.tokenize(seq_y), dtype=torch.int, ).unsqueeze(0).to('cuda:0')
        _,tens_y = evo2_model(tok_y, return_embeddings=True, layer_names=[l_name])
        y = tens_y[l_name].cpu().float().numpy()
        
        sim.append(np.dot(x.ravel(),y.ravel())/(np.linalg.norm(x.ravel())*np.linalg.norm(y.ravel())))
        sim_row.append((np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))).mean())
        
        # print(x.shape)
        x_mean = np.mean(x[0], axis=1)
        y_mean = np.mean(y[0], axis=1)
        # print(x_mean.shape)
        sim_avg.append( np.dot(x_mean, y_mean)/ (np.linalg.norm(x_mean) * np.linalg.norm(y_mean)) )

    df["cosine_sim"] = sim
    df["cosine_sim_row"] = sim_row
    df["cosine_sim_avg"] = sim_avg
    return df

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

    small_df = df.sample(n=100, random_state=42)
    print(df.value_counts("similar"))

    df = main(df, l_name)
    # df[["segment_id", "Similar", "cosine_sim", "cosine_sim_row","cosine_sim_avg", "genome_x", "chr_x", "len_profile_x", "genome_y", "chr_y", "len_profile_y", "seq_x", "seq_y"]].to_csv(output, sep='\t')
    df.to_csv(output, sep='\t')
