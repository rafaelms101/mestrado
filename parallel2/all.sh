nsq=32
coarsek=32
w=4
database=siftsmall
size=10000
queries=100
threads=8

./train $database $size $coarsek $nsq $threads
./genivf $database $size $coarsek $nsq $threads
mpiexec -n 3 nvprof ./ivfpq_test $database $threads $size $queries $coarsek $nsq $w
