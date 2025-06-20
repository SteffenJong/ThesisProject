import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import random
# from create_embeddings import create_embeddings
import numpy as np
from sklearn.model_selection import train_test_split

def get_gene_fam_per_gene(gene_fam: Path, refseqs: list, sample_size:int = 1000):
    """combines the gene families from gene fam and all the genes from all refseqs into one pd.Dataframe.
    By default not all genes will be used but a 1000 randomly selected genes

    Args:
        gene_fam (Path): path to gene_fam_parsed.tsv generated by parese_gene_fam.py
        refseqs (list): list with alle genes per organism files like: annotation.all_transcripts.ath.csv.gz
        output_path (Path): outputh path

    Returns:
        pd.Datafame: datafram with all gene_ids and to witch fam they belong. 
    """

    gl = pd.read_csv(gene_fam, sep="\t",header=None, names=['gene_id', 'family_id'])
    print(f"starting with {refseqs[0]}")
    df = pd.read_csv(refseqs[0], compression='gzip', sep='\t', header=0, skiprows=8)
    df.rename(columns={"#gene_id": "gene_id"}, inplace=True)
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=42, replace=False)
    df = pd.merge(df, gl, on="gene_id", how='left')

    for path in refseqs[1:]:
        print(f"starting with {path}")
        dfn = pd.read_csv(path, compression='gzip', sep='\t', header=0, skiprows=8)
        dfn.rename(columns={"#gene_id": "gene_id"}, inplace=True)
        dfn = dfn.sample(n=sample_size, random_state=42, replace=False)
        dfn = pd.merge(dfn, gl, left_on=['gene_id'], right_on=['gene_id'], how='left')
        df = pd.concat([df, dfn])
    df.reset_index(inplace=True)
    return df

def create_similar_genes(df: pd.DataFrame):
    """creates cobinations of genes from the same gene family

    Args:
        df (pd.Dataframe): dataframe from get_gene_fam_per_gene

    Returns:
        pd.Dataframe: df with genes from the same family
    """
    out = []
    for name, group in df.groupby('family_id'):
        if group.shape[0] < 2: continue
        gc = group.copy()
        for i1, r in group.iterrows():
            for i2, j in gc.iterrows():
                if r.gene_id[:2] != j.gene_id[:2]:
                    if [r.gene_id, j.gene_id, r.seq, j.seq] not in out:
                        out.append([name , r.gene_id, j.gene_id, r.seq, j.seq])
            
            try:
                gc.drop(i1, axis=0, inplace=True)
            except:
                print("group")
                print("gc")
                break
    return pd.DataFrame(out, columns=["family", "gene_x", "gene_y", "seq_x", "seq_y"])


def create_negatives(df: pd.DataFrame):
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
    for fam, group in df.groupby("family"):
        shuffeld = group.copy()

        temp = df_selection.query('family != @fam').sample(n=shuffeld.shape[0], random_state=42)
        df_selection = df_selection[~df_selection.index.isin(temp.index)]
        shuffeld[["seq_y", "gene_y"]] = temp[["seq_y", "gene_y"]].values
        groups.append(shuffeld)
    df = pd.concat(groups)
    df.insert(loc=3, column="similar", value=False)
    return df

def create_not_similar_random(df):
    not_sim = df.copy()
    not_sim["gene_y"] = not_sim["gene_x"]
    not_sim["seq_y"] = not_sim.apply(random_seq, axis=1)
    not_sim.insert(loc=3, column="similar", value=False)
    return not_sim


def create_random_similar(df):
    df["seq_y"] = df.apply(random_similar, axis=1)
    df.insert(loc=3, column="similar", value=True)
    return df

def crop_or_padd(r: pd.Series, length):
    return pd.Series([r["seq_x"][:length] + "0"*(length-len(r["seq_x"])), r["seq_y"][:length] + "0"*(length-len(r["seq_y"]))])
    
def random_seq(r: pd.Series):
        return "".join([ random.choice(["A", "C", "T", "G"]) for i in range(len(r["seq_y"]))])

def random_similar(r: pd.Series):
    seq = r["seq_x"]
    mask = random.sample(range(len(seq)), round( len(seq)*0.3))
    new_seq = ""
    for indx, i in enumerate(seq):
        options = ["A", "C", "T", "G"]
        options.remove(i)
        if indx in mask:
            new_seq += random.choice(options)
        else:
            new_seq += i
    return new_seq

def create_train_test_val(df):
    x_index, x_val_index, _, _ = train_test_split(df.index, df.similar.astype(int).values, test_size=0.30, random_state=42)
    df_val = df.iloc[x_val_index]
    df = df.iloc[x_index].reset_index(drop=True)
    x_train_index, x_test_index, _, _ = train_test_split(df.index, df.similar.astype(int).values, test_size=0.20, random_state=42)
    return df.iloc[x_train_index], df.iloc[x_test_index], df_val

# def main(gene_fam: Path, refseqs: list, output_path, same_length):
#     gl = pd.read_csv(gene_fam, sep="\t",header=None, names=['gene_id', 'family_id'])
#     print(f"starting with {refseqs[0]}")
#     df = pd.read_csv(refseqs[0], compression='gzip', sep='\t', header=0, skiprows=8)
#     df.rename(columns={"#gene_id": "gene_id"}, inplace=True)
#     df = df.sample(n=1000, random_state=42, replace=False)
#     df = pd.merge(df, gl, on="gene_id", how='left')

#     for path in refseqs[1:]:
#         print(f"starting with {path}")
#         dfn = pd.read_csv(path, compression='gzip', sep='\t', header=0, skiprows=8)
#         dfn.rename(columns={"#gene_id": "gene_id"}, inplace=True)
#         dfn = dfn.sample(n=1000, random_state=42, replace=False)
#         dfn = pd.merge(dfn, gl, left_on=['gene_id'], right_on=['gene_id'], how='left')
#         df = pd.concat([df, dfn])
#     df.reset_index(inplace=True)
#     out = []

#     for name, group in df.groupby('family_id'):
#         if group.shape[0] < 2: continue
#         gc = group.copy()
#         for i1, r in group.iterrows():
#             for i2, j in gc.iterrows():
#                 if r.gene_id[:2] != j.gene_id[:2]:
#                     if [r.gene_id, j.gene_id, r.seq, j.seq] not in out:
#                         out.append([name , r.gene_id, j.gene_id, r.seq, j.seq])
            
#             try:
#                 gc.drop(i1, axis=0, inplace=True)
#             except:
#                 print("group")
#                 print("gc")
#                 break

#     df1 = pd.DataFrame(out, columns=["family", "gene_x", "gene_y", "seq_x", "seq_y"])
    
    
#     # df_selection = df1.copy()

#     # groups = []
#     # for fam, group in df1.groupby("family"):
#     #     shuffeld = group.copy()

#     #     temp = df_selection.query('family != @fam').sample(n=shuffeld.shape[0])
#     #     df_selection = df_selection[~df_selection.index.isin(temp.index)]
#     #     shuffeld[["seq_y", "gene_y"]] = temp[["seq_y", "gene_y"]].values

#     #     # print(shuffeld.shape[0])
#     #     # print(shuffeld.head())
#     #     groups.append(shuffeld)

#     # not_sim = pd.concat(groups)

#     not_sim = df1.copy()
#     not_sim["seq_y"] = not_sim.apply(random_seq, axis=1)
#     not_sim.insert(loc=3, column="similar", value=False)
    
#     df1["seq_y"] = df1.apply(random_similar, axis=1)
#     df1.insert(loc=3, column="similar", value=True)

#     results = pd.concat([not_sim, df1]).reset_index(drop=True)
    
#     if same_length:
#         mean = int((results.seq_y.apply(len).mean() + results.seq_x.apply(len).mean())/2)
#         results[["seq_x", "seq_y"]] =  results.apply(crop_or_padd, length = mean, axis=1)    
#     results.to_csv(output_path, sep="\t")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--gene_fam", type=str, default="data/gene_fam_parsed.tsv")
    parser.add_argument('--refseqs', nargs='+', default=["data/annotation.all_transcripts.ath.csv.gz", "data/annotation.all_transcripts.bol.csv.gz", "data/annotation.all_transcripts.aar.csv.gz"])
    parser.add_argument('--output_prefix', type=str)
    parser.add_argument('--output_prefix_raw', type=str, default="")
    # not implemented yet 
    # parser.add_argument('--same_length', type=bool, default=False)
    # parser.add_argument('--save_raw_train_test', type=bool, default=False)
    parser.add_argument('--samplesize_per_organism', type=int, default=1000)
    parser.add_argument('--size_per_organism', type=int, default=None)

    args = parser.parse_args()

    gene_fam = Path(args.gene_fam)
    refseqs = [Path(f) for f in args.refseqs]
    output_prefix = Path(args.output_prefix)
    output_prefix_raw = Path(args.output_prefix_raw)
    # same_length = args.same_length
    sample_size = args.samplesize_per_organism

    if not output_prefix.parent.is_dir():
        raise ValueError(f"Output folder {output_prefix.parent} doesn't exist")

    df = get_gene_fam_per_gene(gene_fam, refseqs)
    df = create_similar_genes(df)

    # df_not_sim = create_not_similar_random(df)
    # df_sim = create_random_similar(df)
    df_negatives = create_negatives(df)
    df.insert(loc=3, column="similar", value=True)

    df = pd.concat([df, df_negatives]).reset_index(drop=True)
    
    train, test, val = create_train_test_val(df)
    if output_prefix_raw != "":
        train.to_csv(str(output_prefix_raw)+"_train_raw.tsv", sep="\t")
        test.to_csv(str(output_prefix_raw)+"_test_raw.tsv", sep="\t")
        val.to_csv(str(output_prefix_raw)+"_val_raw.tsv", sep="\t")

    print("generating trian embeddings")
    train_em = create_embeddings(train)
    train_em.to_csv(str(output_prefix)+"_train.tsv", sep="\t")

    print("generating test embeddings")
    test_em = create_embeddings(test)
    test_em.to_csv(str(output_prefix)+"_test.tsv", sep="\t")

    print("generating val embeddings")
    val_em = create_embeddings(val)
    val_em.to_csv(str(output_prefix)+"_val.tsv", sep="\t")
    
