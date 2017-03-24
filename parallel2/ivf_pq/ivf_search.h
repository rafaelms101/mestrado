#ifndef H_IVFSEARCH
#define H_IVFSEARCH

	#include <stdio.h>
	#include <stdlib.h>
	#include <mpi.h>
	#include <pthread.h>
	extern "C"{
		#include "../yael_needs/vector.h"
		#include "../yael_needs/nn.h"
		#include "../yael_needs/kmeans.h"
	}
	#include "../pq-utils/pq_test_load_vectors.h"
	#include "../pq-utils/pq_new.h"
	#include "../pq-utils/pq_search.h"
	#include "myIVF.h"
	#include "k_min.h"

	typedef struct ivf_threads{
		ivf_t *ivf;
		ivfpq_t ivfpq;
		int threads;
		int thread;
		mat residual;
	}ivf_threads_t;

	void parallel_search (int nsq, int my_rank, int k, int comm_sz, int threads, MPI_Comm search_comm);
	dis_t ivfpq_search(ivf_t *ivf, float *residual, pqtipo pq, int centroid_idx);
	int min(int a, int b);
	float * sumidxtab2(mat D, matI ids, int offset);
	int* imat_new_transp (const int *a, int ncol, int nrow);

#endif
