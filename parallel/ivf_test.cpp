#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include "pq-utils/pq_new.h"
#include "pq-utils/pq_test_load_vectors.h"

#include "ivf_pq/ivf_assign.h"
#include "ivf_pq/ivf_new.h"
#include "ivf_pq/ivf_search.h"
#include "ivf_pq/ivf_parallel.h"

int main(int argv, char **argc){

	if(argv < 2){
		cout << "Usage: ./ivfpq_test <dataset>" << endl;
		return -1;
	}

	int k,
		nsq,
		coarsek,
		w,
		comm_sz,
		my_rank,
		last_agregator,
		last_search;

	char* dataset;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	dataset = argc[1];
	k = 100;
	nsq = 8;
	coarsek = 256;
	w = 4;
	last_search=comm_sz-2;
	last_agregator=comm_sz-1;

	if (my_rank==0){
		int *ids;
		float *dis;

		parallel_training(dataset, coarsek, nsq, last_search, k, dis, ids, w, last_agregator);
	}
	if(my_rank>0 && my_rank<=last_search){
		parallel_search(nsq, last_search, my_rank, last_agregator);
	}
	if(my_rank>last_search){
		parallel_agregator(k, w, my_rank, last_agregator, last_search);
	}
	MPI_Finalize();
	return 0;
}