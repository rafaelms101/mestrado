#include <boost/filesystem.hpp>

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

#include "ivf_pq/debug.h"




//TODO: it might be possible to read some of these arguments from the ivf files itself rather than passing them.
int main(int argc, char **argv){

	if (argc != 9) {
		cout << "Usage: mpiexec -n ./ivfpq_test <dataset> <threads> <tam> <num_queries> <coarsek> <nsq> <w> <gpu/cpu>"
			 << endl;
		return -1;
	}


	char* dataset = argv[1];
	int threads  = atoi(argv[2]);
	int tam  = atoi(argv[3]);
	int num_queries = atoi(argv[4]);
	int coarsek = atoi(argv[5]);
	int nsq = atoi(argv[6]);
	int w = atoi(argv[7]);
	int k = 100;
	bool gpu = ! std::strcmp("gpu", argv[8]);

	
	
	MPI_Group world_group;
	

	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
	
	int comm_sz;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	debug("comm_sz: %d", comm_sz);
	
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	debug("rank: %d", my_rank);
	
	MPI_Comm_group(MPI_COMM_WORLD, &world_group);

	int last_assign, last_search, last_aggregator;
	last_assign = 0;
	last_search = comm_sz - 2;
	last_aggregator = comm_sz - 1;

	int search_nodes_qty = last_search - last_assign;
	int ranks[search_nodes_qty+1];

	for (int i = 0; i < search_nodes_qty + 1; i++) {
		ranks[i] = i;
	}
	
	MPI_Group search_group;
	MPI_Comm search_comm;
	MPI_Group_incl(world_group, search_nodes_qty + 1, ranks, &search_group);
	MPI_Comm_create_group(MPI_COMM_WORLD, search_group, 0, &search_comm);

	char* train_path;
	asprintf(&train_path, "%s/%s/train/%d_%d_%d", BASE_DIR, dataset, tam, coarsek, nsq);

	if (! boost::filesystem::exists(train_path)) {
		std::printf("You have to train first\n");
		exit(-1);
	}
	
	char* ivf_path;
	asprintf(&ivf_path, "%s/%s/ivf/%d_%d_%d", BASE_DIR, dataset, tam, coarsek, nsq);

	if (! boost::filesystem::exists(ivf_path)) {
		std::printf("You have to generate the IVF first\n");
		exit(-1);
	}
	
	if (my_rank <= last_assign) {
		parallel_assign(dataset, w, comm_sz - 1, search_comm, threads, num_queries, train_path);
	} else if (my_rank <= last_search) {
		parallel_search(nsq, k, threads, tam / search_nodes_qty, comm_sz - 1, search_comm, dataset, w, train_path, ivf_path, gpu);
	} else {
		parallel_aggregator(k, w, my_rank, 0, search_nodes_qty, tam, num_queries, threads, dataset);
	}

	free(train_path);
	free(ivf_path);
	
	
	MPI_Finalize();


	return 0;
}
