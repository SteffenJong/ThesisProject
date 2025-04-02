import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def add_genes(df, folder):
    sg = pd.read_csv(folder/"segments.txt", sep="\t", header=0)
    merged_df = pd.merge(
    df, sg[['first','last', "multiplicon", "genome"]][sg["order"]==0],
    left_on=['id', 'genome_x'],
    right_on=['multiplicon', 'genome'],
    how='left'  
    )
    merged_df.drop(["multiplicon", "genome"], axis=1, inplace=True)
    merged_df.rename(columns={'first': 'first_x', 'last': 'last_x'}, inplace=True)
    
    merged_df = pd.merge(
        merged_df, sg[['first','last', "multiplicon", "genome"]][sg["order"]==1],
        left_on=['id', 'genome_y'],
        right_on=['multiplicon', 'genome'],
        how='left'  
    )
    merged_df.drop(["multiplicon", "genome"], axis=1, inplace=True)
    merged_df.rename(columns={'first': 'first_y', 'last': 'last_y'}, inplace=True)
    return merged_df

def add_start_stop(df, anno_files, decompress):
    names = ["gene_id",	"species", "transcript", "coord_cds", "start", "stop", "coord_transcript", "seq", "strand", "chr", "type", "check_transcript", "check_protein", "transl_table"]
    
    df["start_x"] = pd.NA
    df["stop_x"] = pd.NA
    df["start_y"] = pd.NA
    df["stop_y"] = pd.NA
    merged_df = df.copy()

    for file in anno_files:
        if decompress:
            anno = pd.read_csv(file, compression='gzip', sep="\t", skiprows = 9, names=names)
        else:    
            anno = pd.read_csv(file, sep="\t", skiprows = 9, names=names)
        anno = anno.drop_duplicates(subset=['gene_id'], keep='first')

        merged_df = pd.merge(
        merged_df, anno[['gene_id','start']],
        left_on=['first_x'],
        right_on=['gene_id'],
        how='left',
        )
        merged_df = pd.concat([merged_df[merged_df.columns.difference(["start_x", "start"])], merged_df["start_x"].combine_first(merged_df["start"])], axis=1)

        # df["start_x"] = df.groupby(np.where(df.columns == 'start_x', 'start'), axis=1).first()
        merged_df.drop(["gene_id"], axis=1, inplace=True)
        # merged_df.rename(columns={'start': 'start_x'}, inplace=True)

        merged_df = pd.merge(
        merged_df, anno[['gene_id','stop']],
        left_on=['last_x'],
        right_on=['gene_id'],
        how='left'  
        )
        merged_df = pd.concat([merged_df[merged_df.columns.difference(["stop_x", "stop"])], merged_df["stop_x"].combine_first(merged_df["stop"])], axis=1)
        # df["stop_x"] = df.groupby(np.where(df.columns == 'stop_x', 'stop'), axis=1).first()
        merged_df.drop(["gene_id"], axis=1, inplace=True)
        # merged_df.rename(columns={'stop': 'stop_x'}, inplace=True)

        merged_df = pd.merge(
        merged_df, anno[['gene_id','start']],
        left_on=['first_y'],
        right_on=['gene_id'],
        how='left',
        )
        merged_df = pd.concat([merged_df[merged_df.columns.difference(["start_y", "start"])], merged_df["start_y"].combine_first(merged_df["start"])], axis=1)
        # df["start_y"] = df.groupby(np.where(df.columns == 'start_y', 'start'), axis=1).first()
        merged_df.drop(["gene_id"], axis=1, inplace=True)
        # merged_df.rename(columns={'start': 'start_y'}, inplace=True)

        merged_df = pd.merge(
        merged_df, anno[['gene_id','stop']],
        left_on=['last_y'],
        right_on=['gene_id'],
        how='left'  
        )
        merged_df = pd.concat([merged_df[merged_df.columns.difference(["stop_y", "stop"])], merged_df["stop_y"].combine_first(merged_df["stop"])], axis=1)
        # df["stop_y"] = df.groupby(np.where(df.columns == 'stop_y', 'stop'), axis=1).first()
        merged_df.drop(["gene_id"], axis=1, inplace=True)
        # merged_df.rename(columns={'stop': 'stop_y'}, inplace=True)

    return merged_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--iadhdir", type=str)
    parser.add_argument('--annofiles', nargs='+')
    parser.add_argument("--output", type=str, default="IADH_results.tsv")
    parser.add_argument("--decompress", type=bool, default=True)
    
    args = parser.parse_args()

    iadhdir = Path(args.iadhdir)
    output = Path(args.output)
    annofiles = [Path(f) for f in args.annofiles]

    print("reading csv")
    mp = pd.read_csv(iadhdir/"multiplicons.txt", sep="\t", header=0)
    print("adding genes")
    df = add_genes(mp, iadhdir)
    print("adding start stop")
    df = add_start_stop(df, annofiles, args.decompress)
    print("write to csv")
    df.to_csv(output, sep='\t')

