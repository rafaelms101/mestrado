#include "pq_new.h"

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

	flags = flags & KMEANS_INIT_RANDOM;
	ds=vtrain.d/nsq;
	ks=2^nsq;

	vs=fmat_new (ds, vtrain.n);

	pq.nsq = nsq;
	pq.ks = ks;
	pq.ds = ds;
	pq.centroids.mat=fvec_new(nsq);

	for(i=0;i<nsq;i++){
		for(j=0;j<ds;j++){
			vs[j]=vtrain.mat[(i+j-1)*ds];
		}
		kmeans(ds, vtrain.n, ks, 100, vs, flags, seed/*numero aleatorio dif de 0*/, 1, centroids_tmp , dis, assign, NULL);
		//pq.centroids=centroids_tmp
		//fvec_concat(pq.centroids.mat, pq.centroids.n, pq.centroids.d, centroids_tmp, ks, ds);
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
