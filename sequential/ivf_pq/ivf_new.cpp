#include "ivf_new.h"
#include "../pq-utils/pq_new.h"

ivfpq_t ivfpq_new(int coarsek, int nsq, mat vtrain){

	ivfpq_t ivfpq;
	ivfpq.coarsek = ivfpq.coa_centroidsn = coarsek;
	ivfpq.coa_centroidsd = vtrain.d;
	ivfpq.coa_centroids = (float*)malloc(sizeof(float)*ivfpq.coa_centroidsn*ivfpq.coa_centroidsd);

	float * dis = (float*)malloc(sizeof(float)*vtrain.n);

	//definicao de variaveis
	int flags = 0;
	flags = flags | KMEANS_INIT_BERKELEY;
	flags |= 1;
	flags |= KMEANS_QUIET;
	int* assign = (int*)malloc(sizeof(int)*vtrain.n);

	kmeans(vtrain.d, vtrain.n, coarsek, 50, vtrain.mat, flags, 2, 1, ivfpq.coa_centroids, NULL, NULL, NULL);

	//calculo do vetores residuais
	knn_full(L2, vtrain.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd, 1, ivfpq.coa_centroids, vtrain.mat, NULL, assign, dis);
	subtract(vtrain, ivfpq.coa_centroids, assign, ivfpq.coa_centroidsd);

	//aprendizagem do produto residual
	ivfpq.pq = pq_new(nsq, vtrain, coarsek);

	free(assign);
	free(dis);

	return ivfpq;
}


void subtract(mat v, float* v2, int* idx, int c_d){
	for (int i = 0; i < v.d; i++) {
		for (int j = 0; j < v.n; j++) {
			v.mat[j*v.d + i] = v.mat[j*v.d + i] - v2[idx[j]*c_d + i];
		}
	}
}

void subtract2(mat v, float* v2, int* idx, int c_d, float* vout){
	for (int i = 0; i < v.d; i++) {
		for (int j = 0; j < v.n; j++) {
			vout[j*v.d + i] = v.mat[j*v.d + i] - v2[idx[j]*c_d + i];
		}
	}
}

void printMat(float* mat, int n, int d){

	printf("[");
		for (int i = 0; i < d; i++) {
			for (int j = 0; j < n; j++) {
				printf("%.3f ", mat[j*d + i]);
			}
			printf(";\n");
		}
	printf("]\n");

	printf("rows= %d cols =%d\n", d, n);
	return;
}

void printMatI(int* mat, int n, int d){

	printf("[");
		for (int i = 0; i < d; i++) {
			for (int j = 0; j < n; j++) {
				printf("%d ", mat[j*d + i]);
			}
			printf(";\n");
		}
	printf("]\n");

	printf("rows= %d cols =%d\n", d, n);
	return;
}

//query -> n do vetor a ser copiado,
//nq ????????????
void copySubVectorsI(int* qcoaidx, int* coaidx, int query, int nq, int w){
	for (int i = 0; i < w; i++) {
			qcoaidx[i] = coaidx[query*w + i];
	}
}

void copySubVectors2(float* vout, float* vin, int dim, int nvec, int subn){
	for (int i = 0; i < dim; i++) {
			vout[i] = vin[nvec*dim + i + subn*dim];
	}
}
