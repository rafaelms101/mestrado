#include "pq_search.h"

void pq_search(pqtipo pq, int *cbase, mat vquery, int k){
	int i,
		j,
		*ids;

	float 	*distab,
			*dis;

	mat vsub;

	for (i=0;i<vquery.n;i++){

		for (j=0;j<pq.nsq;j++){
			copySubVectors(vsub.mat,vquery ,pq.ds,j,i,i);
			compute_cross_distances (vsub.d, vsub.n, pq.centroids.n, vsub.mat, pq.centroids.mat, distab);
			//criar sumidxtab
		}


	}
}

float* sumidxtab(float* D, mat v, int offset){
	//aloca o vetor de a ser retornado
	static float *dis = (float*)malloc(sizeof(float)*v.n*v.d);
	float dist_tmp = 0;
	int j;

	//soma as distancias para cada vetor
	for (int i = 0, j = 0 ; i < v.n*v.d ; i++) {

		//caso o resto seja 0 houve a mudança de vetor
		if(i % v.d == 0){
			dis[j] = dist_tmp;
			dist_tmp = 0;
			j++;
			dist_tmp += D[i];
		}
		else{
			dist_tmp += D[i];
		}
	}

	return dis;
}
