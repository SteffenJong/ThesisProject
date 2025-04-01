import argparse
from pathlib import Path
from tqdm import tqdm
import gzip

def get_genes(g_dir:Path):
    genes = []
    for file in tqdm(g_dir.glob("**/*"), desc=f"retrieving all genes from {g_dir}"):
        with open(file, "r") as f_in:
            for gene in f_in:
                genes.append(gene.strip("\n")[:-1])
    
    return genes

def get_missing_genes(genes, g_list):
    f_genes = []
    with open(g_list, "r") as f:
        for line in tqdm(f, desc=f"retrieving genes from {g_list}"):
            f_genes.append(line.split()[0])

    return set(genes) - set(f_genes)


def add_singeltons(singeltons, g_list, g_dir):
    with open(g_list, "a") as f:
        counter = 0
        for gene in tqdm(singeltons, desc=f"Adding singeltons to {g_list}"):
            f.write(f"{gene}\t{g_dir.name}{counter}\n")
            counter+=1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Huh')
    parser.add_argument("--gene_list_dir", type=str)
    parser.add_argument("--gene_fam", type=str, default="data/gene_fam_parsed.tsv")
    
    args = parser.parse_args()

    g_dir = Path(args.gene_list_dir)
    g_list = Path(args.gene_fam)
    genes = get_genes(g_dir)
    missing = get_missing_genes(genes, g_list)
    if len(missing) > 0:
        print(f"missing genes: {len(missing)}")
        add_singeltons(missing, g_list, g_dir)
    else: print("No missing singeltons found")  


