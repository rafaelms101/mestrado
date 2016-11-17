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
		last_aggregator,
		last_search,
		last_assign;

	char* dataset;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	dataset = argc[1];
	k = 100;
	nsq = 8;
	coarsek = 256;
	w = 4;
	last_assign=1;
	last_search=comm_sz-2;
	last_aggregator=comm_sz-1;

	if (my_rank==0){
		double start=0, finish=0;
		start = MPI_Wtime();
		parallel_training (dataset, coarsek, nsq, last_search, last_aggregator, last_assign);
		MPI_Recv(&finish, 1, MPI_DOUBLE, last_aggregator, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("%g milissegundos\n", finish*1000-start*1000);
	}
	else if(my_rank<=last_assign){
		parallel_assign (dataset, last_search, last_assign, w, last_aggregator);
	}
	else if(my_rank<=last_search){
		parallel_search (nsq, last_search, my_rank, last_aggregator, k, last_assign);
	}
	else{
		parallel_aggregator(k, w, my_rank, last_aggregator, last_search, last_assign);
		double finish = MPI_Wtime();
		MPI_Send(&finish, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}