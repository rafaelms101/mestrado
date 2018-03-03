#!/usr/bin/bash
mpiexec -n 4 ./ivfpq_test siftsmall 4         10000 8         2    3   1
#mpiexec -n  ./ivfpq_test <dataset> <threads> <tam> <coarsek> <nsq> <w> <threads_training>
