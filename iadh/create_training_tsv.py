import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import gzip
from Bio import SeqIO


def main(merged: pd.DataFrame, refseq: list):
    counter = 0
    for file in refseq:
        with gzip.open(file, "rt") as f:
            for record in SeqIO.parse(f, format="fasta"):
                
                for i in merged.iterrows():
                    counter += 1
                    if counter >5:
                        break
                    print(i)
                



def filter_df(merged, seg_len):
    df = pd.read_csv(merged, sep="\t", header=0)
    return df[(df["len_profile_x"]<=seg_len) & (df["len_profile_y"]<=seg_len) & (df["genome_x"] != df["genome_y"])]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--merged_iadh_tsv", type=str)
    parser.add_argument('--refseqs', nargs='+')
    parser.add_argument('--segment_length', type=int)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()

    merged = Path(args.merged_iadh_tsv)
    refseq = [Path(f) for f in args.refseqs]
    seg_len = args.segment_length
    output_path = Path(args.output)
    filtered_df = filter_df(merged, seg_len)
    main(filtered_df, refseq)

    
