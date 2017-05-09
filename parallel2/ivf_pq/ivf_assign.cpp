#include "ivf_assign.h"

void parallel_assign (char *dataset, int w, int comm_sz, MPI_Comm search_comm){
	mat vquery, residual;
	ivfpq_t ivfpq;
	int *coaidx, dest, rest,id, search_rank;
	float *coadis;
	static int last_assign, last_search, last_aggregator;

	set_last (comm_sz, &last_assign, &last_search, &last_aggregator);

	vquery = pq_test_load_query(dataset);

	//Recebe os centroides
	MPI_Recv(&ivfpq, sizeof(ivfpq_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.pq.centroids = (float*)malloc(sizeof(float)*ivfpq.pq.centroidsn*ivfpq.pq.centroidsd);
	MPI_Recv(&ivfpq.pq.centroids[0], ivfpq.pq.centroidsn*ivfpq.pq.centroidsd, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.coa_centroids=(float*)malloc(sizeof(float)*ivfpq.coa_centroidsd*ivfpq.coa_centroidsn);
	MPI_Recv(&ivfpq.coa_centroids[0], ivfpq.coa_centroidsn*ivfpq.coa_centroidsd, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	//Calcula o residuo de cada vetor da query para os processos de busca
	residual.d = vquery.d;
	residual.n = vquery.n*w;
	residual.mat = (float*)malloc(sizeof(float)*residual.d*residual.n);

	coaidx = (int*)malloc(sizeof(int)*vquery.n*w);
	coadis = (float*)malloc(sizeof(float)*vquery.n*w);
	knn_full(2, vquery.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd, w, ivfpq.coa_centroids, vquery.mat, NULL, coaidx, coadis);
	
	for(int i=0;i<vquery.n; i++){
		for(int j=0;j<w;j++){
			id=i*w+j;
			bsxfunMINUS(&residual.mat[residual.d*id], vquery, ivfpq.coa_centroids, i, &coaidx[id], 1);
		}
	}
	//Envia os identificadores dos centroides correspondentes a cada vetor da query para o agregador
	for(int i=last_search+1; i<=last_aggregator; i++){
		MPI_Send(&vquery.n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
		MPI_Send(&coaidx[0], vquery.n*w, MPI_INT, i, 0, MPI_COMM_WORLD);
	}
	
	int i=0;

	//Envia o resíduo para o processo de busca
	MPI_Barrier(search_comm);
	double start = MPI_Wtime();
    MPI_Bcast(&residual.n, 1, MPI_INT, 0, search_comm);
    MPI_Bcast(&residual.d, 1, MPI_INT, 0, search_comm);
	
    MPI_Bcast(&residual.mat[0]+i*residual.d, residual.d*residual.n, MPI_FLOAT, 0, search_comm);
    MPI_Bcast(&coaidx[0]+i, residual.n, MPI_INT, 0, search_comm);	
	
	MPI_Send(&start, 1, MPI_DOUBLE, last_aggregator, 0, MPI_COMM_WORLD);

	free(coaidx);
	free(coadis);
	free(residual.mat);
	free(vquery.mat);
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);
}

void bsxfunMINUS(float *mout, mat vin, float* vin2, int nq, int* qcoaidx, int ncoaidx){
	for (int i = 0; i < vin.d; i++) {
		for (int j = 0; j < ncoaidx; j++) {
			mout[j*vin.d+i] = vin.mat[(vin.d*nq) + i] - vin2[(qcoaidx[j]*vin.d)+i];
		}
	}
}
