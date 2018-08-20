nsq=32
coarsek=32
w=4
database=sift
size=1000000
queries=10000
threads=8

#./train $database $size $coarsek $nsq $threads
#./genivf $database $size $coarsek $nsq $threads
mpiexec -n 3 nvprof ./ivfpq_test $database $threads $size $queries $coarsek $nsq $w
