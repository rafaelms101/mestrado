#include <stdio.h>
#include <mpi.h>
#include <omp.h>
extern "C"{
	#include "../yael_needs/nn.h"
}
#include "../pq-utils/pq_new.h"
#include "../pq-utils/pq_test_load_vectors.h"
#include "../pq-utils/pq_test_compute_stats.h"
#include "myIVF.h"
#include "k_min.h"

void parallel_aggregator(int k, int w, int my_rank, int comm_sz, int tam_base, int nqueries, int threads, char* dataset);
