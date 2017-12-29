#!/bin/bash
#SBATCH -p RM
#SBATCH -t 2:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node 2


mpiexec -n 4 ./ivfpq_test siftbig 2 10000000
