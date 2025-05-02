import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def add_genes(df, folder):
    sg = pd.read_csv(folder/"segments.txt", sep="\t", header=0)
    sg.rename(columns={'id': 'segment_id'}, inplace=True)

    merged_df = pd.merge(
    df, sg[['first','last', "multiplicon", "genome", "segment_id"]][sg["order"]==0],
    left_on=['id', 'genome_x'],
    right_on=['multiplicon', 'genome'],
    how='left'  
    )
    merged_df.drop(["multiplicon", "genome"], axis=1, inplace=True)
    merged_df.rename(columns={'first': 'first_x', 'last': 'last_x', "segment_id": "segment_id_x"}, inplace=True)
    
    merged_df = pd.merge(
        merged_df, sg[['first','last', "multiplicon", "genome", "segment_id"]][sg["order"]==1],
        left_on=['id', 'genome_y'],
        right_on=['multiplicon', 'genome'],
        how='left'  
    )
    merged_df.drop(["multiplicon", "genome"], axis=1, inplace=True)
    merged_df.rename(columns={'first': 'first_y', 'last': 'last_y', "segment_id": "segment_id_y"}, inplace=True)
    
    le = pd.read_csv(folder/"list_elements.txt", sep="\t", header=0)
    le_grouped = le.groupby("segment")[["gene", "orientation"]].agg(list).reset_index()

    merged_df = pd.merge(merged_df, le_grouped, left_on="segment_id_x", right_on="segment", how="left")
    merged_df.drop(["segment"], axis=1, inplace=True)
    merged_df.rename(columns={'gene': 'genes_x', 'orientation': 'orientations_x'}, inplace=True)

    merged_df = pd.merge(merged_df, le_grouped, left_on="segment_id_y", right_on="segment", how="left")
    merged_df.drop(["segment"], axis=1, inplace=True)
    merged_df.rename(columns={'gene': 'genes_y', 'orientation': 'orientations_y'}, inplace=True)

    # merged_df["orientations_x"] = merged_df["orientations_x"].apply(eval)
    # merged_df["orientations_y"] = merged_df["orientations_y"].apply(eval)
    merged_df[["sim_orientations_x", "sim_orientations_y"]] = merged_df.apply(sim_orientation, axis=1)

    return merged_df

def sim_orientation(s: pd.Series):
    lx = s["orientations_x"]
    ly = s["orientations_y"]
    return pd.Series([max(lx.count("+"), lx.count("-")) / len(lx), max(ly.count("+"), ly.count("-")) / len(ly)])

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

def add_extra_collums(df):
    df[["start_x","stop_x", "start_y", "stop_y"]] = df[["start_x","stop_x", "start_y", "stop_y"]].apply(pd.to_numeric)

    df["len_x"] =  df["stop_x"] - df["start_x"]
    df["len_y"] =  df["stop_y"] - df["start_y"]

    df["len_profile_x"] = df["end_x"] - df["begin_x"]+1
    df["len_profile_y"] = df["end_y"] - df["begin_y"]+1
    df["max_profile_length"] = df[["len_profile_x", "len_profile_y"]].max(axis=1)

    df["percentage_similar"] = df["number_of_anchorpoints"] / df["max_profile_length"]
    return df


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
    print("adding some collums")
    df = add_extra_collums(df)
    print(f"writing to {output}")
    df.to_csv(output, sep='\t', index=False)

