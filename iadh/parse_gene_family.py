import argparse
from pathlib import Path
import gzip
from tqdm import tqdm

def main(input: Path, output: Path):
    with open(input, "r") as f_in:
        with open(output, "w+") as f_out:
            for line in f_in:
                if line.startswith("#"): continue
                items = line.split()
                f_out.write(f"{items[2]}\t{items[0]}\n")

def main_unziped(input: Path, output: Path):
    with gzip.open(input, 'rt') as f_in:
        with open(output, "w+") as f_out:
            for line in tqdm(f_in, desc=f"creating {output}"):
                if line.startswith("#"): continue
                items = line.split()
                f_out.write(f"{items[2]}\t{items[0]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str, default="gene_fam_parsed.tsv")
    parser.add_argument("--decompress", type=bool, default=True)
    args = parser.parse_args()

    input = Path(args.input)
    output = Path(args.output)
    if args.decompress:
        main_unziped(input, output)
    else:
        main(input, output)