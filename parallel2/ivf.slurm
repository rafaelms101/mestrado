#!/bin/bash
#SBATCH -J ivfpq2e200           
#SBATCH -o ivfpq2e200.o%j  
#SBATCH -N 103
#SBATCH -n 103                
#SBATCH -p normal     
#SBATCH -t 3:00:00        
#SBATCH --mail-user=andreff@outlook.com
#SBATCH --mail-type=begin  
#SBATCH --mail-type=end    

ibrun -n 103 -o 0 ./ivfpq_test siftbig 14 200000000
