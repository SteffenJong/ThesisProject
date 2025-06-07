import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import gzip
from Bio import SeqIO
import math
from sklearn.model_selection import train_test_split
from create_embeddings import create_embeddings


def create_train_df(merged: pd.DataFrame, refseq: list):
    #output dict.
    output = {}
    # loop trough reference sequences.
    for file in refseq:
        #open refseq 
        with gzip.open(file, "rt") as f:
            # loop trough files in refseq (wich are mostly chromosones).
            for record in tqdm(SeqIO.parse(f, format="fasta"), desc=f"Collecting sequences from{file}"):
                # get the name of the current name of the refseq (Wich genome we are looking at).
                file_name = file.stem.split(".")[0]
                # get the name of the current file in the refseq (Wich chromosone we are looking at).
                chr_name = record.id.split(".")[0]
                # filter the df such that we are only looking at rows that have a segment that is from the current genome and chromosone.
                # We are doing this to reduce the size of the forloop, to improve speed.
                df_focus = merged[(merged[["genome_x", "genome_y"]] == file_name).any(axis=1) & (merged[["list_x", "list_y"]] == chr_name).any(axis=1)]
                # loop over df
                for indx, r in df_focus.iterrows():
                    # for the current row see if genome_x and or genome_y is the genome that is opened right now.
                    genome = r[r==file_name]
                    # loop over genome_x or genome_y or both debending on what genome is open right now.
                    for i, _ in genome.items():
                        # set x_y to either x or y depending on wich genome we are looking at
                        x_y = i.split("_")[1]
                        # check if the current (iadh) segment is from the correct chromosone that we have opend right now.
                        if r[f"list_{x_y}"] == chr_name:
                            # see if a dict already exists for the current id(indx), if so we update it, if not we create the id and write the inital data.
                            output.setdefault(r["id"], {}).update({f"genome_{x_y}": r[f"genome_{x_y}"],
                                                                f"chr_{x_y}": r[f"list_{x_y}"],
                                                                f"segment_id_{x_y}": r[f"segment_id_{x_y}"],
                                                                f"len_profile_{x_y}": r[f"len_profile_{x_y}"],
                                                                f"seq_{x_y}": str(record.seq[r[f"start_{x_y}"]-1:r[f"stop_{x_y}"]])})
                            
    return pd.DataFrame.from_dict(output, orient='index')[["genome_x", "chr_x", "len_profile_x", "segment_id_x", "genome_y", "chr_y", "len_profile_y", "segment_id_y", "seq_x", "seq_y"]]


def create_test_df(train_df: pd.DataFrame):
    df_c = train_df.copy() 
    df_c["og_index"] = df_c.index

    grouped = df_c.groupby(["len_profile_x", "len_profile_y"])
    shuffled_parts = []

    for _, group in grouped:
        # makes sure group is not to small to shuffel
        if len(group) == 1:
            shuffled_parts.append(group)
            print("Error group to small !")
            continue
        
        print(group["seq_y"].duplicated().sum(),"-",(group.shape[0]/2))

        if group["seq_y"].duplicated(keep=False).sum() > (group.shape[0]/2):
            print("Error group to similar !")
            continue

        shuffled_group = group.copy()
        same = True
        # shuffel everything till there are no same combinations as before


        while same:
            shuffled_values = shuffled_group[["genome_y", "chr_y", "len_profile_y", "segment_id_y", "seq_y"]].sample(frac=1, random_state=None).reset_index(drop=True)
            shuffled_group[["genome_y", "chr_y", "len_profile_y","segment_id_y", "seq_y"]] = shuffled_values[["genome_y", "chr_y", "len_profile_y","segment_id_y", "seq_y"]].values
            
            if shuffled_group[shuffled_group["seq_y"] == group["seq_y"]]["seq_y"].notna().sum() == 0:
                same = False
        shuffled_parts.append(shuffled_group)
    return pd.concat(shuffled_parts).sort_values("og_index").drop(columns="og_index")

def create_train_test_val(df):
    x_index, x_val_index, _, _ = train_test_split(df.index, df.similar.astype(int).values, test_size=0.30, random_state=42)
    df_val = df.iloc[x_val_index]
    df = df.iloc[x_index].reset_index(drop=True)
    x_train_index, x_test_index, _, _ = train_test_split(df.index, df.similar.astype(int).values, test_size=0.20, random_state=42)
    return df.iloc[x_train_index], df.iloc[x_test_index], df_val

def filter_df(merged, seg_len):
    df = pd.read_csv(merged, sep="\t", header=0)
    df = df[(df["len_profile_x"]<=seg_len) & (df["len_profile_y"]<=seg_len) & (df["genome_x"] != df["genome_y"])]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--merged_iadh_tsv", type=str)
    parser.add_argument('--refseqs', nargs='+')
    parser.add_argument('--segment_length', type=int)
    parser.add_argument('--output_prefix', type=str)
    parser.add_argument('--output_prefix_seq', type=str, default="")
    parser.add_argument('--max_len_nuc', type=int, default=0)

    args = parser.parse_args()

    merged = Path(args.merged_iadh_tsv)
    refseq = [Path(f) for f in args.refseqs]
    seg_len = args.segment_length
    output_prefix = Path(args.output_prefix)
    output_prefix_seq = Path(args.output_prefix_seq)
    len_nuc = args.max_len_nuc
    filtered_df = filter_df(merged, seg_len)

    # Only put samples in there that have the same orientation. 
    # filtered_df = filtered_df[ (filtered_df["sim_orientations_x"] >= 1) & (filtered_df["sim_orientations_y"] >= 1)]
    
    train_df = create_train_df(filtered_df, refseq)
    
    if len_nuc != 0:
        print(f"Filtering for max length of {len_nuc} nucliotides")
        before = train_df.shape[0]
        train_df = train_df[train_df.apply(lambda r: max(len(r['seq_x']), len(r['seq_y'])) < len_nuc, axis=1)]
        print(f"Went fom {before} to {train_df.shape[0]} pairs")
    
    print("Creating negative samples")
    test_df = create_test_df(train_df)
    train_df["multiplicon_id"] = pd.to_numeric(train_df.index, downcast='integer')
    test_df["multiplicon_id"] = pd.NA
    train_df["similar"] = True
    test_df["similar"] = False
    df = pd.concat([train_df, test_df], ignore_index=True)
    # print(f"Writing output to {output_prefix}")
    # df[["segment_id", "similar", "genome_x", "chr_x", "len_profile_x", "genome_y", "chr_y", "len_profile_y", "seq_x", "seq_y"]].to_csv(output_path, sep='\t')
    
    train, test, val = create_train_test_val(df)
    print(f"shapes: train:{train.shape}, train:{test.shape}, train:{val.shape}")

    train.drop(columns=["seq_x", "seq_y"]).to_csv(str(output_prefix)+"_train.tsv", sep="\t")
    test.drop(columns=["seq_x", "seq_y"]).to_csv(str(output_prefix)+"_test.tsv", sep="\t")
    val.drop(columns=["seq_x", "seq_y"]).to_csv(str(output_prefix)+"_val.tsv", sep="\t")


    if output_prefix_seq != "":
        train.to_csv(str(output_prefix_seq)+"_train_seq.tsv", sep="\t")
        test.to_csv(str(output_prefix_seq)+"_test_seq.tsv", sep="\t")
        val.to_csv(str(output_prefix_seq)+"_val_seq.tsv", sep="\t")

    print("Generating embeddings")
    embeddings = create_embeddings(df)
    embeddings.to_csv(str(output_prefix_seq)+"_embeddings.tsv", sep="\t")

    # print(f"generating train embeddings")
    # train_em = create_embeddings(train)
    # print(train_em.head())
    # print(str(output_prefix)+"_train.tsv")
    # train_em.to_csv(str(output_prefix)+"_train.tsv", sep="\t")

    # print(f"generating test embeddings")
    # test_em = create_embeddings(test)
    # test_em.to_csv(str(output_prefix)+"_test.tsv", sep="\t")

    # print(f"generating val embeddings ")
    # val_em = create_embeddings(val)
    # val_em.to_csv(str(output_prefix)+"_val.tsv", sep="\t")
    
