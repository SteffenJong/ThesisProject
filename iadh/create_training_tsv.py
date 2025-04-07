import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import gzip
from Bio import SeqIO


def main(merged: pd.DataFrame, refseq: list, output_path: Path):
    counter = 0

    output = {}
    # output = {"id": {"genome_x":"", "chr_x":"", "gene_list_x":"", "sequence_x":"", "genome_y":"", "chr_y":"", "gene_list_y":"", "sequence_y":""} }

    for file in refseq:
        with gzip.open(file, "rt") as f, open(output_path, "w+") as f_out:
            for record in SeqIO.parse(f, format="fasta"):
                # Get file name to filter df on refseq
                file_name = file.stem.split(".")[0]
                chr_name = record.id.split(".")[0]
                # print(chr_name)
                # print(merged[["genome_x", "genome_y"]].value_counts())
                df_focus = merged[(merged[["genome_x", "genome_y"]] == file_name).any(axis=1) & (merged[["list_x", "list_y"]] == chr_name).any(axis=1)]
                # print(df_focus[["genome_x", "genome_y"]].value_counts())

                # filter over every row that has a genome that is in the current opend refseq
                for indx, r in df_focus.iterrows():
                    
                    g_x = r["genome_x"]
                    g_y = r["genome_y"]
                    chr_x = r["list_x"]
                    chr_y = r["list_y"]
                    # print(g_x, chr_x, "-", g_y, chr_y)
                    
                    if r["genome_x"] == file_name and r["list_x"] == chr_x:
                        # print(r["start_x"], r["stop_x"])
                        seq_x = record.seq[r["start_x"]:r["stop_x"]]
                        # print(seq_x)
                        try:
                            output[indx].update({"genome_x": g_x, "chr_x": chr_x, "seq_x": str(seq_x)})
                        except KeyError:
                            output[indx] = {"genome_x": g_x, "chr_x": chr_x, "seq_x": str(seq_x)}
                    

                    if r["genome_y"] == file_name and r["list_y"] == chr_y:
                        # print(r["be_y"], r["end_y"])
                        seq_y = record.seq[r["start_y"]:r["stop_y"]]
                        try:
                            output[indx].update({"genome_y": g_y, "chr_y": chr_y, "seq_y": str(seq_y)})
                        except KeyError:
                            output[indx] = {"genome_y": g_y, "chr_y": chr_y, "seq_y": str(seq_y)}
                        
                        
                # print(output)
                # break
    return pd.DataFrame.from_dict(output, orient='index')[["genome_x", "chr_x", "genome_y", "chr_y", "seq_x", "seq_y"]]


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
    df = main(filtered_df, refseq, output_path)
    df.to_csv(output_path, sep='\t')
    
