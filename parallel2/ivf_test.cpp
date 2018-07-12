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
		cout << "Usage: mpiexec -n ./ivfpq_test <dataset> <threads> <tam> <num_queries> <coarsek> <nsq> <w> <threads_training>" << endl;
		return -1;
	}

	int nsq, coarsek, tam, num_queries, comm_sz, threads, w, k, tamt, threads_training, provided;
	char* dataset;

	dataset = argv[1];
	threads  = atoi(argv[2]);
	tam  = atoi(argv[3]);
	num_queries = atoi(argv[4]);
	coarsek = atoi(argv[5]);
	nsq = atoi(argv[6]);
	w = atoi(argv[7]);
	threads_training = atoi(argv[8]);
	comm_sz = 1;
	k = 100;

	#ifdef TRAIN

		struct timeval start, end;
		gettimeofday(&start, NULL);
		parallel_training (dataset, coarsek, nsq, tam, comm_sz, threads_training);
		gettimeofday(&end, NULL);
		double time = ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec));
		cout << time << endl;		
	
	#else
		int my_rank;

		MPI_Group world_group, search_group;
		MPI_Comm search_comm;

		MPI_Init_thread(&argc, &argv,MPI_THREAD_SERIALIZED,&provided);
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
			parallel_training (dataset, coarsek, nsq, tam, comm_sz, threads_training);
		}
		else if(my_rank<=last_assign){
			parallel_assign (dataset, w, comm_sz,search_comm, threads, num_queries);
		}
		else if(my_rank<=last_search){
			parallel_search (nsq, k, comm_sz, threads, tamt, search_comm, dataset, w);
		}
		else{
			parallel_aggregator(k, w, my_rank, comm_sz, tam, num_queries, threads, dataset);
		}
		MPI_Finalize();
	#endif

	return 0;
}
