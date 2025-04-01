import argparse
from pathlib import Path
import gzip
from tqdm import tqdm

def main(inputf: Path, output: Path, remove_more: bool, decompress: bool):
    if decompress:
        with gzip.open(inputf, "rt")as f_in:
            annotation = read_annotation(f_in)
    else:
        with open(inputf, "r") as f_in:
            annotation = read_annotation(f_in)
    
    ressults = remove_dups(annotation, remove_more=remove_more)

    if not output.is_dir():
        output.mkdir()

    for chr in tqdm(ressults.keys(), desc=f"creating gene lists from {inputf}"):
        output_f = output.joinpath(f"{chr}.lst")
        with open(output_f, "w+") as f_out:
            for line in ressults[chr]:
                f_out.write(line+"\n")


def read_annotation(file):
        ressults = {}
        for line in tqdm(file, desc=f"Reading {inputf}"):
            if line.startswith("#"): continue
            items = line.split()
            gen = items[0]
            start = items[4]
            end = items[5]
            direction = items[8]
            chr = items[9]
            if chr not in ressults.keys():
                ressults[chr] = [[gen+direction, int(start), int(end)]]
            else: ressults[chr].append([gen+direction, int(start), int(end)])
        return ressults


def remove_dups(annotation, remove_more):
    ordered_results = {}
    for chr in tqdm(annotation.keys(), desc="Removing duplicates"):
        previous_line = [[],[],[]]
        gene_list = []
        for x in sorted(annotation[chr], key=lambda x: x[1], reverse=False):
            # maybe check if this if statement actually finds dups?            
            if remove_more:
                if x[0] == previous_line[0]:
                    continue
            else:
                if x == previous_line:
                    continue
            gene_list.append(x[0])
            previous_line = x
        ordered_results[chr] = gene_list
    return ordered_results

def remove_dups_old(annotation):
    ordered_results = {}
    for chr in annotation.keys():
        previous_line = [[],[],[]]
        gene_list = []
        for x in sorted(annotation[chr], key=lambda x: x[1], reverse=False):
            # maybe check if this if statement actually finds dups?
            if x == previous_line:
                # print("found dup!")
                counter +=1
                continue
            
            if remove_more:
                if x[0] == previous_line[0]:
                    counter +=1
                else: counter = 0
                if counter >0:
                    # print("found dup!")
                    continue

            gene_list.append(x[0])
            previous_line = x
        ordered_results[chr] = gene_list
    return ordered_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--input", type=str)
    parser.add_argument("--output_dir", type=str, default="gene_list")
    parser.add_argument("--remove_more", type=bool, default=False)
    parser.add_argument("--decompress", type=bool, default=True)
    
    args = parser.parse_args()

    inputf = Path(args.input)
    output = Path(args.output_dir)
    remove_more = args.remove_more
    main(inputf, output, remove_more, args.decompress)



