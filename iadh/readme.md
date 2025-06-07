# I adhore
Short explenation of how all the code works in this folder.

## Code
### Python files
parse_gene_family.py: re structers everything in genefamily_data.HOMFAM.csv.gz so that it can be used by IADH.

anno_to_genelist.py: converts a annotation.all_transcripts.ath.csv.gz file into a folder with genelists

collect_results_iadh.py: Collects all the results of IADH and combines the most important ones into one tsv

### Bash Files
run_preprocess.sh: Used to run all the preprocessing for iadh, an example can be found in this repo

run_iad.sh: Can be used to run IADH

run_res_iadh.sh: Collects all results of iadh 

data/create_ini.sh: Can be used to write all gene list paths to a .ini file that iadh uses. especially usefull if there are a lot of gene lists.

## Folder structures
There are 2 main folders
- data
    - Here all the lists will be stored pre organism
- iadh_out
    - Here all the output of IADH will be stored per run.