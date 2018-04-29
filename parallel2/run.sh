#!/usr/bin/bash
tput reset
#optirun mpiexec -n 4   ./ivfpq_test siftsmall 4  10000 32 32    3   1
optirun mpiexec -n 4   nvprof ./ivfpq_test sift         1        1000000    256         8     4   8
#mpiexec -n                   ./ivfpq_test <dataset> <threads> <tam>   <coarsek> <nsq> <w> <threads_training>

