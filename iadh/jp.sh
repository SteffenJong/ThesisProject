#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=cseduproject
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --output=/home/sdejong/thesis/iadh/logs/jp.out
#SBATCH --error=/home/sdejong/thesis/iadh/logs/jp.err

source /home/sdejong/.bashrc
source activate jp
jupyter lab --ip=0.0.0.0 --port=8080 --no-browser

