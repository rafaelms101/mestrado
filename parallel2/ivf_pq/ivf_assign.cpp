#include "ivf_assign.h"
#include "ivf_training.h"


void parallel_assign(char *dataset, int w, int aggregator_id, MPI_Comm search_comm,int threads, int nqueries, char* train_path){
	mat vquery, residual;
	int *coaidx, dest, rest,id, search_rank;
	float *coadis;

	vquery = pq_test_load_query(dataset, threads, nqueries);
	printf("g1");

	char* header;
	asprintf(&header, "%s/header", train_path);
	char* cent;
	asprintf(&cent, "%s/pq_centroids", train_path);
	char* coa;
	asprintf(&coa, "%s/coa_centroids", train_path);
	
	ivfpq_t ivfpq;
	read_cent(header, cent, coa, &ivfpq);

	printf("g2");

	//Calcula o residuo de cada vetor da query para os processos de busca
	residual.d = vquery.d;
	residual.n = vquery.n*w;
	residual.mat = (float*)malloc(sizeof(float)*residual.d*residual.n);

	coaidx = (int*)malloc(sizeof(int)*vquery.n*w);
	coadis = (float*)malloc(sizeof(float)*vquery.n*w);
	knn_full(2, vquery.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd, w, ivfpq.coa_centroids, vquery.mat, NULL, coaidx, coadis);
	
	free(coadis);

	for(int i=0;i<vquery.n; i++){
		for(int j=0; j<w; j++){
			id = i * w + j;
			bsxfunMINUS(&residual.mat[residual.d*id], vquery, ivfpq.coa_centroids, i, &coaidx[id], 1);
		}
	}

	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);

	//TODO: Im not 100% sure if this comment is correct
	//Send the number of queries to the aggregator
	MPI_Send(&vquery.n, 1, MPI_INT, aggregator_id, 0, MPI_COMM_WORLD);
	free(vquery.mat);


	MPI_Barrier(search_comm);
	double start = MPI_Wtime();
	MPI_Send(&start, 1, MPI_DOUBLE, aggregator_id, 0, MPI_COMM_WORLD);
//
	//Envia o resíduo para o processo de busca
	MPI_Bcast(&residual.n, 1, MPI_INT, 0, search_comm);
	MPI_Bcast(&residual.d, 1, MPI_INT, 0, search_comm);
	MPI_Bcast(&residual.mat[0], residual.d * residual.n, MPI_FLOAT, 0, search_comm);
	MPI_Bcast(&coaidx[0], residual.n, MPI_INT, 0, search_comm);

	free(coaidx);
	free(residual.mat);	
}

void bsxfunMINUS(float *mout, mat vin, float* vin2, int nq, int* qcoaidx, int ncoaidx){
	for (int i = 0; i < vin.d; i++) {
		for (int j = 0; j < ncoaidx; j++) {
			mout[j*vin.d+i] = vin.mat[(vin.d*nq) + i] - vin2[(qcoaidx[j]*vin.d)+i];
		}
	}
}
