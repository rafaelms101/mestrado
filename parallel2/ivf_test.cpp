#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include "pq-utils/pq_new.h"
#include "pq-utils/pq_test_load_vectors.h"

#include "ivf_pq/ivf_assign.h"
#include "ivf_pq/ivf_training.h"
#include "ivf_pq/ivf_search.h"
#include "ivf_pq/ivf_aggregator.h"

int main(int argc, char **argv){

	if(argc < 6){
		cout << "Usage: mpiexec -n ./ivfpq_test <dataset> <threads> <tam> <coarsek> <nsq> <w>" << endl;
		return -1;
	}

	int nsq, coarsek, tam, comm_sz, threads, w;
	char* dataset;

	dataset = argv[1];
	threads  = atoi(argv[2]);
	tam  = atoi(argv[3]);
	coarsek = atoi(argv[4]);
	nsq = atoi(argv[5]);
	w = atoi(argv[6]);
	comm_sz = 1;

	#ifdef TRAIN

		parallel_training (dataset, coarsek, nsq, tam, comm_sz);

	#else
		int k, tamt, my_rank;

		k = 1000;

		MPI_Group world_group, search_group;
		MPI_Comm search_comm;

		MPI_Init(&argc, &argv);
		MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
		MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		int last_assign, last_search, last_aggregator;

		last_assign=1;
		last_search=comm_sz-2;
		last_aggregator=comm_sz-1;
		tamt = tam/(last_search-last_assign);

		int n = comm_sz-2;
		int ranks[n];
		for(int i=0; i<n; i++){
			ranks[i]=i+1;
		}

		MPI_Group_incl(world_group, n, ranks, &search_group);
		MPI_Comm_create_group(MPI_COMM_WORLD, search_group, 0, &search_comm);

		if (my_rank<last_assign){
			parallel_training (dataset, coarsek, nsq, tam, comm_sz);
		}
		else if(my_rank<=last_assign){
			parallel_assign (dataset, w, comm_sz,search_comm, threads);
		}
		else if(my_rank<=last_search){
			parallel_search (nsq, k, comm_sz, threads, tamt, search_comm, dataset, w);
		}
		else{
			parallel_aggregator(k, w, my_rank, comm_sz, tam, threads);
		}
		MPI_Finalize();

	#endif

	return 0;
}
