#!/usr/bin/bash
tput reset
#optirun mpiexec -n 4   ./ivfpq_test siftsmall 4  10000 32 32    3   1
mpiexec -n 3 nvprof  ./ivfpq_test sift         8        1000000 10000    32         32     4 gpu
#mpiexec -n                   ./ivfpq_test <dataset> <threads> <tam>  <queries> <coarsek> <nsq> <w> <threads_training>

