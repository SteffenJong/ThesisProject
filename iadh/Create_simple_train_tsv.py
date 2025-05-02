import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import gzip
from Bio import SeqIO
import math
import numpy as np
import random

def main(gene_fam: Path, refseqs: list, output_path, same_length):
    gl = pd.read_csv(gene_fam, sep="\t",header=None, names=['gene_id', 'family_id'])
    # gl1 = gl.groupby('family_id')['gene_id'].apply(list).reset_index()
    print(f"starting with {refseqs[0]}")
    # print(gl.columns)
    df = pd.read_csv(refseqs[0], compression='gzip', sep='\t', header=0, skiprows=8)
    # print(df.head())
    df.rename(columns={"#gene_id": "gene_id"}, inplace=True)
    df = df.sample(n=1000, random_state=42, replace=False)
    # print(df.columns)
    df = pd.merge(df, gl, on="gene_id", how='left')

    for path in refseqs[1:]:
        print(f"starting with {path}")
        dfn = pd.read_csv(path, compression='gzip', sep='\t', header=0, skiprows=8)
        dfn.rename(columns={"#gene_id": "gene_id"}, inplace=True)
        dfn = dfn.sample(n=1000, random_state=42, replace=False)
        dfn = pd.merge(dfn, gl, left_on=['gene_id'], right_on=['gene_id'], how='left')
        df = pd.concat([df, dfn])
    df.reset_index(inplace=True)
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

    df1 = pd.DataFrame(out, columns=["family", "gene_x", "gene_y", "seq_x", "seq_y"])
    
    
    # df_selection = df1.copy()

    # groups = []
    # for fam, group in df1.groupby("family"):
    #     shuffeld = group.copy()

    #     temp = df_selection.query('family != @fam').sample(n=shuffeld.shape[0])
    #     df_selection = df_selection[~df_selection.index.isin(temp.index)]
    #     shuffeld[["seq_y", "gene_y"]] = temp[["seq_y", "gene_y"]].values

    #     # print(shuffeld.shape[0])
    #     # print(shuffeld.head())
    #     groups.append(shuffeld)

    # not_sim = pd.concat(groups)

    not_sim = df1.copy()
    not_sim["seq_y"] = not_sim.apply(random_seq, axis=1)
    not_sim.insert(loc=3, column="similar", value=False)
    
    df1["seq_y"] = df1.apply(random_similar, axis=1)
    df1.insert(loc=3, column="similar", value=True)

    results = pd.concat([not_sim, df1]).reset_index(drop=True)
    
    if same_length:
        mean = int((results.seq_y.apply(len).mean() + results.seq_x.apply(len).mean())/2)
        results[["seq_x", "seq_y"]] =  results.apply(crop_or_padd, length = mean, axis=1)    
    results.to_csv(output_path, sep="\t")


def crop_or_padd(r: pd.Series, length):
    # print("-")
    # print(r["seq_x"][:length] + "0"*(length-len(r["seq_x"])),"-",r["seq_y"][:length] + "0"*(length-len(r["seq_y"])))
    return pd.Series([r["seq_x"][:length] + "0"*(length-len(r["seq_x"])), r["seq_y"][:length] + "0"*(length-len(r["seq_y"]))])
    

def random_seq(r: pd.Series):
        # print(r)
        return "".join([ random.choice(["A", "C", "T", "G"]) for i in range(len(r["seq_y"]))])

def random_similar(r: pd.Series):
    seq = r["seq_x"]
    mask = random.sample(range(len(seq)), round( len(seq)*0.3))
    # print(len(mask), len(seq))
    new_seq = ""
    for indx, i in enumerate(seq):
        options = ["A", "C", "T", "G"]
        options.remove(i)
        if indx in mask:
            new_seq += random.choice(options)
        else:
            new_seq += i
    return new_seq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--gene_fam", type=str, default="data/gene_fam_parsed.tsv")
    parser.add_argument('--refseqs', nargs='+', default=["data/annotation.all_transcripts.ath.csv.gz", "data/annotation.all_transcripts.bol.csv.gz", "data/annotation.all_transcripts.aar.csv.gz"])
    parser.add_argument('--output', type=str)
    parser.add_argument('--same_length', type=bool, default=False)

    args = parser.parse_args()

    gene_fam = Path(args.gene_fam)
    refseqs = [Path(f) for f in args.refseqs]
    output_path = Path(args.output)
    same_length = args.same_length

    main(gene_fam, refseqs, output_path, same_length)