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
		//pq.centroids=centroids_tmp;
	}

	return pq;
}

void check_new(){
  cout << "::PQ_NEW OK::" << endl;
}
