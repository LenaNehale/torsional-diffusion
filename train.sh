#!/bin/bash
#SBATCH --partition=long                      
#SBATCH --cpus-per-task=8                   
#SBATCH --gres=gpu:1                         
#SBATCH --mem=32G                             
#SBATCH --time=50:00:00                        
#SBATCH --output=job_output%j.txt
#SBATCH --error=job_error%j.txt
# Load the necessary modules
module load anaconda/3 

# Activate a virtual environment, if needed:
conda activate torsional_diffusion

python train.py