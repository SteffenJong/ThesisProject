import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import gzip
from Bio import SeqIO


def create_train_df(merged: pd.DataFrame, refseq: list, output_path: Path):
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
                            output.setdefault(indx, {}).update({f"genome_{x_y}": r[f"genome_{x_y}"],
                                                                f"chr_{x_y}": r[f"list_{x_y}"],
                                                                f"len_profile_{x_y}": r[f"len_profile_{x_y}"],
                                                                f"seq_{x_y}": str(record.seq[r[f"start_{x_y}"]:r[f"stop_{x_y}"]])})      
    return pd.DataFrame.from_dict(output, orient='index')[["genome_x", "chr_x", "len_profile_x", "genome_y", "chr_y", "len_profile_y", "seq_x", "seq_y"]]


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
        
        shuffled_group = group.copy()
        same = True
        # shuffel everything till there are no same combinations as before
        while same:
            shuffled_values = shuffled_group[["genome_y", "chr_y", "len_profile_y", "seq_y"]].sample(frac=1, random_state=None).reset_index(drop=True)
            shuffled_group[["genome_y", "chr_y", "len_profile_y", "seq_y"]] = shuffled_values[["genome_y", "chr_y", "len_profile_y", "seq_y"]].values
            
            if shuffled_group[shuffled_group["seq_y"] == group["seq_y"]]["seq_y"].notna().sum() == 0:
                same = False 

        shuffled_parts.append(shuffled_group)

    return pd.concat(shuffled_parts).sort_values("og_index").drop(columns="og_index")

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
    train_df = create_train_df(filtered_df, refseq, output_path)
    print("Creating negative samples")
    test_df = create_test_df(train_df)
    train_df["segment_id"] = pd.to_numeric(train_df.index, downcast='integer')
    test_df["segment_id"] = pd.NA
    train_df["Similar"] = True
    test_df["Similar"] = False
    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Writing output to {output_path}")
    df[["segment_id", "Similar", "genome_x", "chr_x", "len_profile_x", "genome_y", "chr_y", "len_profile_y", "seq_x", "seq_y"]].to_csv(output_path, sep='\t')
    
