#!/bin/bash
#SBATCH -p RM
#SBATCH -t 6:30:00
#SBATCH -N 19
#SBATCH --ntasks-per-node 28

export OMP_NUM_THREADS=24

#run MPI program
mpirun -np 19 ./ivfpq_test siftbig 24 3200000000
