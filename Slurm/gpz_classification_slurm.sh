#!/bin/bash

#SBATCH -t 30
#SBATCH --mem=1024
#SBATCH --tmp=4096MB
#SBATCH -n 4
#SBATCH --job-name GPz_Class
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.luken@westernsydney.edu.au
#SBATCH --output slurm_logs/gpz_class_%A_%a.out
#SBATCH --error slurm_logs/gpz_class_%A_%a.err
#SBATCH --array=0-99

source /fred/oz237/kluken/redshift_pipeline_adacs/Slurm/hpc_profile_setup.sh

mkdir seed_${SLURM_ARRAY_TASK_ID}
cd seed_${SLURM_ARRAY_TASK_ID}

singularity exec $container_path/GPz_Python.simg python3 $script_path/GPz_GMM_HPC_Class.py -s ${SLURM_ARRAY_TASK_ID}
