import argparse
from pathlib import Path

# def main(g_dir: Path, g_list: Path):
#     with open(input, "r") as f_in:
#         with open(output, "w+") as f_out:
#             for line in f_in:
#                 if line.startswith("#"): continue
#                 items = line.split()
#                 f_out.write(f"{items[2]}\t{items[0]}\n")


def get_genes(g_dir:Path):
    genes = []
    for file in g_dir.glob("**/*"):
        with open(file, "r") as f_in:
            for gene in f_in:
                genes.append(gene.strip("\n")[:-1])
    
    return genes

def get_missing_genes(genes, g_list):
    f_genes = []
    with open(g_list, "r+") as f:
        for line in f:
            f_genes.append(line.split()[0])
    return set(genes) - set(f_genes)


def add_singeltons(singeltons, g_list, g_dir):
    with open(g_list, "a") as f:
        counter = 0
        for gene in singeltons:
            f.write(f"{gene}\t{g_dir.name}{counter}\n")
            counter+=1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--gene_list_dir", type=str)
    parser.add_argument("--gene_fam_list", type=str, default="gene_fam_parsed.tsv")
    
    args = parser.parse_args()

    g_dir = Path(args.gene_list_dir)
    g_list = Path(args.gene_fam_list)
    genes = get_genes(g_dir)
    missing = get_missing_genes(genes, g_list)
    print(f"missing genes: {len(missing)}")
    add_singeltons(missing, g_list, g_dir)  


