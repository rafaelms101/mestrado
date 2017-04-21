#!/bin/bash
#SBATCH -p RM
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 28


export OMP_NUM_THREADS=20

#run MPI program
mpirun -np 4 ./ivfpq_test siftbig 20 5000000
