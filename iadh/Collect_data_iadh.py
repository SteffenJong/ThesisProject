import argparse
from pathlib import Path

def main(input_folder: Path, output: Path):
    with open(input_folder+"multiplicons.txt", "r") as ml, open(input_folder+"multiplicons.txt", "r"), open(output, "w+") as out:





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--input_folder", type=str)
    parser.add_argument("--output", type=str, default="iadh_results.tsv")
    
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output = Path(args.output)
    main(input_folder, output)



names = ["gene_id",	"species", "transcript", "coord_cds", "start", "stop", "coord_transcript", "seq", "strand", "chr", "type", "check_transcript", "check_protein", "transl_table"]