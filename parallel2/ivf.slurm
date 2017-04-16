#!/bin/bash
#SBATCH -J ivfpq1000          
#SBATCH -o ivfpq1000.o%j  
#SBATCH -N 103
#SBATCH -n 103    
#SBATCH -p normal     
#SBATCH -t 1:00:00        
#SBATCH --mail-user=andreff@outlook.com
#SBATCH --mail-type=begin  
#SBATCH --mail-type=end    

ibrun -n 103 -o 0 ./ivfpq_test siftbig 2 1000000000
