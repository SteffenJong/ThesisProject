import argparse
from pathlib import Path

def main(input: Path, output: Path):
    with open(input, "r") as f_in:
        with open(output, "w+") as f_out:
            for line in f_in:
                if line.startswith("#"): continue
                items = line.split()
                f_out.write(f"{items[2]}\t{items[0]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str, default="gene_fam_parsed.tsv")
    
    args = parser.parse_args()

    input = Path(args.input)
    output = Path(args.output)
    main(input, output)



