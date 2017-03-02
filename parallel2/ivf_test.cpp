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

int main(int argc, char **argv){

	if(argc < 2){
		cout << "Usage: mpiexec -n ./ivfpq_test <dataset> <threads> <tam>" << endl;
		return -1;
	}

	int 	k,
		nsq,
		coarsek,
		w,
		tam=0,
		comm_sz,
		my_rank,
		threads;

	char* dataset;
	char arquivo[] = "testes.txt";

	int last_assign,
		last_search,
		last_aggregator;

	#ifndef TRAIN
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	#endif

	last_assign=1;
	last_search=comm_sz-2;
	last_aggregator=comm_sz-1;

	dataset = argv[1];
	threads  = atoi(argv[2]);
	tam  = atoi(argv[3]);
	k = 100;
	nsq = 8;
	coarsek = 256;
	w = 4;

	set_last (comm_sz, threads);
	
	#ifdef TRAIN
	
		parallel_training (dataset, coarsek, nsq, tam);

	#else

		if (my_rank<last_assign){
			double start=0, finish=0;
			FILE *fp;
			if(my_rank==0){
				fp = fopen(arquivo, "a");
				fprintf(fp,"Teste com a base %s, coarsek=%d, w=%d, ", dataset, coarsek, w);
				if(tam!=0)fprintf(fp,"tamanho%d\n\n", tam);
				fclose(fp);
				start = MPI_Wtime();
			}
			parallel_training (dataset, coarsek, nsq, tam);
	
			if(my_rank==0){
				MPI_Recv(&finish, 1, MPI_DOUBLE, last_aggregator, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				fp = fopen(arquivo, "a");
				fprintf(fp,"Tempo total: %g milissegundos\n\n", finish*1000-start*1000);
				fclose(fp);
			}
		}
		else if(my_rank<=last_assign){
			parallel_assign (dataset, w);
		}
		else if(my_rank<=last_search){
			parallel_search (nsq, my_rank, k, arquivo);
		}
		else{
			parallel_aggregator(k, w, my_rank, arquivo);
			double finish = MPI_Wtime();
			MPI_Send(&finish, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
		MPI_Finalize();
	
	#endif

	return 0;
}
