#include "pq_search.h"

void pq_search(pqtipo pq, matI codebook, mat vquery, int k, float *dis, int *ids){
	int i,
		j,
		*ids1;

	float 	*dis1,
			*distab_temp;

	mat vsub,
		distab,
		disquerybase;

	vsub.mat= (float*)malloc(sizeof(float)*pq.ds*vquery.n);
	vsub.n=vquery.n;
	vsub.d=pq.ds;	

	distab.mat = (float*)malloc(sizeof(float)*vquery.n*pq.ks*pq.nsq);
	distab.n = pq.nsq;
	distab.d = vquery.n*pq.ks;

	distab_temp = (float*)malloc(sizeof(float)*vquery.n*pq.ks);

	disquerybase.mat= (float*)malloc(sizeof(float)*codebook.n);
	disquerybase.n=codebook.n;
	disquerybase.d=1;

	dis1= (float*)malloc(sizeof(float)*k);
	ids1= (int*)malloc(sizeof(int)*k);


	for (i=0;i<vquery.n;i++){

		for (j=0;j<pq.nsq;j++){
			copySubVectors(vsub.mat,vquery ,pq.ds,j,i,i);
			compute_cross_distances (vsub.d, vsub.n, pq.ks, vsub.mat, pq.centroids[j], distab_temp);
			memcpy(distab.mat+j*pq.ks*vsub.n, distab_temp, sizeof(float)*pq.ks*vsub.n);
		}
		disquerybase.mat = sumidxtab(distab, codebook);
		k_min(disquerybase, k, dis1, ids1);

		memcpy(dis+i*k, dis1, sizeof(float)*k);
		memcpy(ids+i*k, ids1, sizeof(int)*k);
	}
}

float* sumidxtab(mat distab, matI codebook){
	//aloca o vetor a ser retornado
	static float *dis = (float*)malloc(sizeof(float)*codebook.n);
	float dis_tmp = 0;
	int i, j;

	//soma as distancias para cada vetor
	for (i = 0; i < codebook.n ; i++) {
		dis_tmp = 0;
		for(j=0; j<distab.n*distab.d; j+=distab.d){
			dis_tmp += distab.mat[*(codebook.mat++) + j];
		}
		dis[i]=dis_tmp;
	}

	return dis;
}

void k_min (mat disquerybase, int k, float *dis, int *ids){
	int i,
		j,
		d,
		n;

	if(disquerybase.d==1 && disquerybase.n>1){
		d=disquerybase.n;
		n=1;
	}	


	for (i=0; i<n; i++){
		fvec_k_min(disquerybase.mat, d, ids, k);
		for(j=0; j<k; j++){
			dis[j] = disquerybase.mat[ids[j]];
			ids[j]++;
		}
		ids += k;
		dis += k;
		disquerybase.mat += d;
	}
}