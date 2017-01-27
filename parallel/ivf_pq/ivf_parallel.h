#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <pthread.h>
extern "C"{
	#include "../yael_needs/nn.h"
}
#include "../pq-utils/pq_new.h"
#include "../pq-utils/pq_test_load_vectors.h"
#include "../pq-utils/pq_test_compute_stats.h"
#include "ivf_assign.h"
#include "ivf_new.h"
#include "ivf_search.h"
#include "myIVF.h"

void set_last (int comm_sz, int num_threads);
void parallel_training (char *dataset, int coarsek, int nsq, int tam);
void parallel_assign (char *dataset, int w);
void parallel_search (int nsq, int my_rank, int k, char *arquivo);
void parallel_aggregator(int k, int w, int my_rank, char *arquivo);
void *search_threads(void *ivf_threads_recv);
