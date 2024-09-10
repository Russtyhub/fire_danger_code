#!/bin/bash

#SBATCH -N 1                                     # Number of nodes
#SBATCH -C cpu                                   # Use CPU nodes (not GPU)
#SBATCH -q debug                                 # Queue type
#SBATCH -J fire_danger_evaluation                # Job name
#SBATCH -o ../Log_Output/evaluation/%x_%j.out    # Standard output file
#SBATCH -e ../Log_Output/evaluation/%x_%j.err    # Standard error file
#SBATCH -t 00:30:00                              # Time limit
#SBATCH -A m2467                                 # Allocation account

module load tensorflow/2.15.0

# Use srun to distribute tasks across nodes
srun -N 1 --ntasks=4 --cpus-per-task=32 --cpu-bind=cores python3 ../evaluation/evaluation_V3.py
