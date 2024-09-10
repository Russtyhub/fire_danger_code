#!/bin/bash

#SBATCH -N 12                                     # Number of nodes
#SBATCH -C gpu                                   # Use GPU nodes
#SBATCH -q regular                               # Queue type
#SBATCH -J fire_danger_tuning                    # Job name
#SBATCH -o ../Log_Output/tuning/%x_%j.out         # Standard output file
#SBATCH -e ../Log_Output/tuning/%x_%j.err         # Standard error file
#SBATCH -t 20:00:00                              # Time limit
#SBATCH -A m2467                                 # Allocation account
#SBATCH -G 48                                    # Total number of GPUs requested
#SBATCH --gpus-per-node=4                        # Number of GPUs per node

module load tensorflow/2.15.0

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Keras tuner environment vars
export KERASTUNER_ORACLE_PORT="8008"
export KERASTUNER_ORACLE_IP=$(hostname)

# Use srun to distribute tasks across nodes
srun -N 12 --ntasks-per-node=1 --cpus-per-task=128 --gpus-per-task=4 --cpu-bind=cores --gpu-bind=single:4 python3 ../transformer_tuning.py