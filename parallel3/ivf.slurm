#!/bin/bash
#SBATCH -J ivfpq100          
#SBATCH -o ivfpq100.o%j  
#SBATCH -N 53
#SBATCH -n 53    
#SBATCH -p normal     
#SBATCH -t 3:00:00        
#SBATCH --mail-user=andreff@outlook.com
#SBATCH --mail-type=begin  
#SBATCH --mail-type=end    

ibrun -n 53 -o 0 ./ivfpq_test siftbig 1 100000000
