#include <stdio.h>
#include <mpi.h>
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

void parallel_training (char *dataset, int coarsek, int nsq, int last_search, int k, float *dis, int *ids, int w, int last_agregator);
void parallel_search(int nsq, int last_search, int my_rank, int last_agregator);
void parallel_agregator(int k, int w, int my_rank, int last_agregator, int last_search);