#!/bin/bash
#SBATCH -J ivfpq64           
#SBATCH -o ivfpq64.o%j  
#SBATCH -N 35
#SBATCH -n 560                 
#SBATCH -p normal     
#SBATCH -t 03:30:00        
#SBATCH --mail-user=andreff@outlook.com
#SBATCH --mail-type=begin  
#SBATCH --mail-type=end    

ibrun -n 35 -o 0 ./ivfpq_test siftbig 14 64000000
