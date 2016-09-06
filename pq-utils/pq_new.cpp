#include "pq_new.h"

/*
* nsq: numero de subquantizadores (m no paper)
* vtrain: estrutura de vetores, contendo os dados de treinamento
*/
pqtipo pq_new(int nsq, mat vtrain){
	int	i,
		j,
		ds,
		ks,
		flags,
		*assign,
		seed=1;

	float	*centroids_tmp,
			*dis,
			*vs;

	pqtipo pq;

	flags = flags | KMEANS_INIT_RANDOM;
	flags |= 1;
	flags |= KMEANS_QUIET;
	ds=vtrain.d/nsq;
	ks=2^nsq;

	vs=fmat_new (ds, vtrain.n);

	pq.nsq = nsq;
	pq.ks = ks;
	pq.ds = ds;
	pq.centroids.mat = fvec_new(nsq);
	pq.centroids.n=ks;

	for(i=0;i<nsq;i++){
		for(j=0;j<ds*vtrain.n;j++){
			vs[j]=vtrain.mat[(i*vtrain.n*ds)+j];
		}
		kmeans(ds, vtrain.n, ks, 100, vs, flags, seed/*numero aleatorio dif de 0*/, 1, centroids_tmp , dis, assign, NULL);
		fvec_concat(pq.centroids.mat, pq.centroids.n, centroids_tmp, ks);
		pq.centroids.n += ks;
	}

	return pq;
}

void check_new(){
  cout << ":::PQ_NEW OK:::" << endl;
}

void fvec_concat(float* vinout, int vinout_n, float* vin, int vin_n){
	if (vinout == NULL) {
		vinout = (float*)malloc(sizeof(float)*vin_n);
		vinout_n = 0;
		memcpy(vinout + vinout_n, vin, sizeof(float)*vin_n);
	}
	else{
		fvec_resize(vinout, vinout_n + vin_n);
		memcpy(vinout + vinout_n, vin, sizeof(float)*vin_n);
	}
}

void ivec_concat(int* vinout, int vinout_n, int* vin, int vin_n){
	if (vinout == NULL) {
		vinout = (int*)malloc(sizeof(int)*vin_n);
		vinout_n = 0;
		memcpy(vinout + vinout_n, vin, sizeof(int)*vin_n);
	}
	else{
		ivec_resize(vinout, vinout_n + vin_n);
		memcpy(vinout + vinout_n, vin, sizeof(int)*vin_n);
	}
}
