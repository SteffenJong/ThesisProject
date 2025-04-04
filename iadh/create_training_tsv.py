import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def main(merged, refseq, seg_len):
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--merged_iadh_tsv", type=str)
    parser.add_argument('--refseqs', nargs='+')
    parser.add_argument('--segment_length', type=int)

    args = parser.parse_args()

    merged = Path(args.merged_iadh_tsv)
    refseq = [Path(f) for f in args.annofiles]
    seg_len = args.segment_length
    main(merged, refseq, seg_len)
    
