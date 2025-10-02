#!/bin/bash

#SBATCH --job-name="llm-legal-interp"
#SBATCH --output="%x.o%j"
#SBATCH --time=8:00:00
#SBATCH --gpus-per-node=4
#SBATCH --mem=128000
#SBATCH --account=def-annielee
#SBATCH --mail-user=jm3743@georgetown.edu
#SBATCH --mail-type=END,FAIL

source env-ca.sh
python main.py --seed 1 >> infer-1.log