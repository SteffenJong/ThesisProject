import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import random
from create_embeddings import create_embeddings
import numpy as np
from sklearn.model_selection import train_test_split
from Create_simple_train_tsv import get_gene_fam_per_gene, create_similar_genes, create_train_test_val

def create_segment_2(df: pd.DataFrame, flip=True):
    df.seq_x = df.seq_x.astype("string")
    df.seq_y = df.seq_y.astype("string") 

    halfa = df.sample(frac=0.5,random_state=42)
    halfb = df[~df.index.isin(halfa.index)]
    halfa = halfa.add_suffix("a")
    halfb = halfb.add_suffix("b")
    halfb.reset_index(inplace=True)
    halfb.rename(columns={"index":'segment_id_x'}, inplace=True)
    halfa.reset_index(inplace=True)
    halfa.rename(columns={"index":'segment_id_y'}, inplace=True)
    df = pd.concat([halfa, halfb], axis=1)
    
    if flip:
        df1 = df.sample(frac=0.5,random_state=42)
        df2 = df[~df.index.isin(halfa.index)]
        df1.reset_index(inplace=True)
        df2.reset_index(inplace=True)
        
        
        df1["seq_x"] = df1.seq_xa + df1.seq_xb
        df1["seq_y"] = df1.seq_ya + df1.seq_yb

        df2["seq_x"] = df2.seq_xa + df2.seq_xb
        df2["seq_y"] = df2.seq_yb + df2.seq_ya
        df = pd.concat([df1, df2]).reset_index()
        df.drop(columns=['seq_xa', 'seq_ya', 'seq_xb', 'seq_yb', "index"], inplace=True)
    else:
        df["seq_x"] = df.seq_xa + df.seq_xb
        df["seq_y"] = df.seq_ya + df.seq_yb
        df.drop(columns=['seq_xa', 'seq_ya', 'seq_xb', 'seq_yb'], inplace=True)
    
    return df


def create_negative_segment(df: pd.DataFrame):
    """Generates negative samples (not similar)
    Negatives samples are created by randomly selecting another gene from a different faimly.
    No duplicates should appear

    Args:
        df (pd.DataFrame): output from create_similar_genes

    Returns:
        pd.Dataframe: df with negative samples
    """
    df_selection = df.copy()
    groups = []
    for (fama, famb), group in df.groupby(["familya", "familyb"]):
        shuffeld = group.copy()
        # print("printing group in create negative")
        temp = df_selection.query('(familya != @fama) & (familyb != @famb)').sample(n=shuffeld.shape[0], random_state=1)
        df_selection = df_selection[~df_selection.index.isin(temp.index)]
        shuffeld[["segment_id_y", "gene_ya",  "gene_yb", "seq_y"]] = temp[["segment_id_y", "gene_ya", "gene_yb", "seq_y"]].values
        groups.append(shuffeld)
    return pd.concat(groups)

def create_train_test_val(df):
    x_index, x_val_index, _, _ = train_test_split(df.index, df.similar.astype(int).values, test_size=0.30, random_state=42)
    df_val = df.iloc[x_val_index]
    
    df = df.iloc[x_index].reset_index(drop=True)
    x_train_index, x_test_index, _, _ = train_test_split(df.index, df.similar.astype(int).values, test_size=0.20, random_state=42)
    return df.iloc[x_train_index], df.iloc[x_test_index], df_val    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--gene_fam", type=str)
    parser.add_argument('--refseqs', nargs='+')
    parser.add_argument('--output_prefix', type=str)
    parser.add_argument('--output_prefix_raw', type=str, default="")
    parser.add_argument('--samplesize_per_organism', type=int, default=500)
    parser.add_argument('--size_per_organism', type=int, default=None)

    args = parser.parse_args()

    gene_fam = Path(args.gene_fam)
    refseqs = [Path(f) for f in args.refseqs]
    output_prefix = Path(args.output_prefix)
    output_prefix_raw = Path(args.output_prefix_raw)
    sample_size = args.samplesize_per_organism

    if not output_prefix.parent.is_dir():
        raise ValueError(f"Output folder {output_prefix.parent} doesn't exist")

    df = get_gene_fam_per_gene(gene_fam, refseqs)
    df = create_similar_genes(df)
    df = create_segment_2(df, flip=False)

    df_negatives = create_negative_segment(df)
    df_negatives.insert(loc=7, column="similar", value=False)
    df.insert(loc=7, column="similar", value=True)

    df = pd.concat([df, df_negatives]).reset_index(drop=True)
    # df.reset_index().rename(columns={df.index.name:'segment_id'})

    train, test, val = create_train_test_val(df)
    train.drop(columns=["seq_x", "seq_y"]).to_csv(str(output_prefix)+"_train.tsv", sep="\t")
    test.drop(columns=["seq_x", "seq_y"]).to_csv(str(output_prefix)+"_test.tsv", sep="\t")
    val.drop(columns=["seq_x", "seq_y"]).to_csv(str(output_prefix)+"_val.tsv", sep="\t")
    
    if str(output_prefix_raw) != "":
        train.to_csv(str(output_prefix_raw)+"_train_raw.tsv", sep="\t")
        test.to_csv(str(output_prefix_raw)+"_test_raw.tsv", sep="\t")
        val.to_csv(str(output_prefix_raw)+"_val_raw.tsv", sep="\t")

    print("creating embeddings")
    embeddings = create_embeddings(df)
    embeddings.to_csv(str(output_prefix)+"_embeddings.tsv", sep="\t", compression="gzip")
    
