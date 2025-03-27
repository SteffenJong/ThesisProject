import argparse
from pathlib import Path

def main(inputf: Path, output: Path, remove_more: bool):
    ressults = {}
    with open(inputf, "r") as f_in:
        for line in f_in:
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

    ordered_results = {}
    for chr in ressults.keys():
        previous_line = [[],[],[]]
        gene_list = []
        for x in sorted(ressults[chr], key=lambda x: x[1], reverse=False):
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
                    print("found dup!")
                    continue

            gene_list.append(x[0])
            previous_line = x
        ordered_results[chr] = gene_list
    ressults = ordered_results

    if not output.is_dir():
        output.mkdir()


    for chr in ressults.keys():
        output_f = output.joinpath(f"{chr}.lst")
        with open(output_f, "w+") as f_out:
            for line in ressults[chr]:
                f_out.write(line+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--input", type=str)
    parser.add_argument("--output_dir", type=str, default="gene_list")
    parser.add_argument("--remove_more", type=bool, default=False)
    
    args = parser.parse_args()

    inputf = Path(args.input)
    output = Path(args.output_dir)
    remove_more = args.remove_more
    main(inputf, output, remove_more)



